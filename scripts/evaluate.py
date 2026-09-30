#!/usr/bin/env python3
"""Evaluate systems A/B/C on fixed per-hop samples.

  A: LoRA direct-answer model -> Hits@1
  B: LoRA DSL model, unconstrained -> executable-query rate, parse-fail share, Hits@1
  C: LoRA DSL model + xgrammar schema/entity constraint -> same + trie compile time

Writes per-item JSONL to data/runs/<run-id>/.
Usage:
  PYTHONPATH=src python scripts/evaluate.py --system A --adapter data/adapters/direct
  PYTHONPATH=src python scripts/evaluate.py --system B --adapter data/adapters/dsl
  PYTHONPATH=src python scripts/evaluate.py --system C --adapter data/adapters/dsl
"""
import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from graphproof.dsl.parser import parse_query
from graphproof.executor.engine import execute, render_proof
from graphproof.executor.graph import KnowledgeGraph

SYSTEM_DIRECT = ("Answer the movie question with the correct entity name(s), "
                 "separated by ' | ' if there are several. Reply with nothing else.")
SYSTEM_DSL = ("Translate the movie question into a path-query. Reply with nothing else.\n"
              "Grammar: START \"<entity>\" (->|<-) <relation> ... RETURN ?x "
              "[WHERE ?x.value <op> <value>] [EXCEPT \"<entity>\"]")

EVAL_SPECS = [
    (1, "data/metaqa/raw/1-hop/vanilla/qa_test.txt", 2000),
    (2, "data/metaqa/raw/2-hop/vanilla/qa_test.txt", 2000),
    (3, "data/metaqa/raw/3-hop/vanilla/qa_test.txt", 2000),
]


def load_sample(path: str, n: int) -> list[dict]:
    import random
    rng = random.Random(42)
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            q, _, ans = line.rstrip("\n").partition("\t")
            rows.append({"id": i, "question": q,
                         "answers": [a for a in ans.split("|") if a]})
    rng.shuffle(rows)
    return rows[:n]


def _escape_ebnf(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def build_matcher(tokenizer, relations: set[str], entities: set[str]):
    """Compile xgrammar matcher: DSL shape + relation vocabulary + entity trie.

    The entity alternation (~43k names) compiles into a token trie inside
    xgrammar; its compile time is measured and reported (spec requires it).
    """
    import xgrammar as xgr

    rel_alt = "|".join(f'"{r}"' for r in sorted(relations))
    ent_alt = "|".join(f'"{_escape_ebnf(e)}"' for e in sorted(entities))
    grammar_ebnf = (
        'root ::= "START " entity hop+ "RETURN ?x" ("WHERE ?x.value " compop " " value)? ("EXCEPT " entity)?\n'
        f'entity ::= ({ent_alt})\n'
        f'hop ::= ("->" | "<-") " " ({rel_alt})\n'
        'compop ::= ">=" | "<=" | "==" | "!=" | ">" | "<"\n'
        'value ::= [0-9]+ ("." [0-9]+)? | "\\"" [^"]+ "\\""\n'
    )
    t0 = time.time()
    import xgrammar as xgr
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(grammar_ebnf)
    compile_s = time.time() - t0
    return compiled, compile_s, len(tokenizer)


def generate_constrained(model, tokenizer, compiled, vocab_size, messages, max_tokens=256):
    """Greedy generation with xgrammar token mask applied per step."""
    import xgrammar as xgr

    matcher = xgr.GrammarMatcher(compiled)
    bitmask = xgr.allocate_token_bitmask(1, vocab_size)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    for tok in prompt_ids:
        if not matcher.accept_token(tok):
            break
    generated = list(prompt_ids)
    device = next(model.parameters()).device
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([generated], device=device)).logits[0, -1]
        matcher.fill_next_token_bitmask(bitmask)
        xgr.apply_token_bitmask_inplace(logits.unsqueeze(0), bitmask)
        nxt = int(torch.argmax(logits))
        if nxt == tokenizer.eos_token_id:
            break
        if not matcher.accept_token(nxt):
            break
        generated.append(nxt)
        if matcher.is_terminated():
            break
    full = tokenizer.decode(generated, skip_special_tokens=True)
    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    return full[len(prompt_text):] if full.startswith(prompt_text) else full


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["A", "B", "C"])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model", default="data/models/Qwen--Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out-dir", default="data/runs")
    args = ap.parse_args()

    print("Loading KB...", flush=True)
    kg = KnowledgeGraph.from_kb_file("data/metaqa/raw/kb.txt")
    print(f"KB: {len(kg)} triples, {len(kg.entities)} entities", flush=True)

    print(f"Loading base + adapter {args.adapter}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    matcher_compiled, vocab_size, compile_s = None, 0, 0.0
    if args.system == "C":
        print("Compiling xgrammar schema+entity matcher...", flush=True)
        matcher_compiled, compile_s, vocab_size = build_matcher(tok, kg.relations, kg.entities)
        print(f"Trie+grammar compiled in {compile_s:.1f}s", flush=True)

    run_id = f"sys{args.system}_{int(time.time())}"
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    totals = {"hits": 0, "n": 0, "parse_fail": 0, "exec_empty": 0}
    sys_prompt = SYSTEM_DIRECT if args.system == "A" else SYSTEM_DSL

    for hop, path, n in EVAL_SPECS:
        sample = load_sample(path, n)
        print(f"Hop {hop}: {len(sample)} questions", flush=True)
        with open(run_dir / f"hop{hop}.jsonl", "w", encoding="utf-8") as f:
            for it in sample:
                messages = [{"role": "system", "content": sys_prompt},
                            {"role": "user", "content": it["question"]}]
                if args.system == "A":
                    prompt = tok.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
                    inputs = tok(prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                    response = tok.decode(out[0][inputs.input_ids.shape[1]:],
                                          skip_special_tokens=True)
                    pred = {p.strip() for p in response.split("|") if p.strip()}
                    hit = bool(pred & set(it["answers"]))
                    rec = {"id": it["id"], "hop": hop, "question": it["question"],
                           "gold": it["answers"], "response": response[:500],
                           "hit": hit, "parse_ok": None}
                else:
                    if args.system == "C":
                        response = generate_constrained(
                            model, tok, matcher_compiled, vocab_size, messages)
                    else:
                        prompt = tok.apply_chat_template(messages, tokenize=False,
                                                         add_generation_prompt=True)
                        inputs = tok(prompt, return_tensors="pt").to(model.device)
                        with torch.no_grad():
                            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                        response = tok.decode(out[0][inputs.input_ids.shape[1]:],
                                              skip_special_tokens=True)
                    try:
                        q = parse_query(response.strip())
                        parse_ok = True
                    except Exception:
                        parse_ok = False
                    if not parse_ok:
                        totals["parse_fail"] += 1
                        rec = {"id": it["id"], "hop": hop, "question": it["question"],
                               "gold": it["answers"], "response": response[:500],
                               "hit": False, "parse_ok": False}
                    else:
                        r = execute(q, kg)
                        pred = set(r.answers)
                        if not pred:
                            totals["exec_empty"] += 1
                        hit = bool(pred & set(it["answers"]))
                        rec = {"id": it["id"], "hop": hop, "question": it["question"],
                               "gold": it["answers"], "response": response[:500],
                               "dsl": str(q), "proof": render_proof(r, q)[:1000],
                               "hit": hit, "parse_ok": True}
                totals["hits"] += rec["hit"]
                totals["n"] += 1
                f.write(json.dumps(rec) + "\n")
        hop_hits = sum(1 for _ in open(run_dir / f"hop{hop}.jsonl") if json.loads(_)["hit"])
        print(f"  hop{hop} Hits@1: {hop_hits}/{len(sample)} ({hop_hits/len(sample):.1%})", flush=True)

    report = {"system": args.system, "adapter": args.adapter, "run_id": run_id,
              "hits": totals["hits"], "n": totals["n"],
              "hits_at_1": totals["hits"] / totals["n"],
              "parse_fail": totals["parse_fail"], "exec_empty": totals["exec_empty"],
              "trie_compile_s": compile_s}
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDone: Hits@1={report['hits_at_1']:.1%} ({totals['hits']}/{totals['n']}) "
          f"parse_fail={totals['parse_fail']} exec_empty={totals['exec_empty']}", flush=True)
    print(f"Report -> {run_dir}/report.json", flush=True)


if __name__ == "__main__":
    main()

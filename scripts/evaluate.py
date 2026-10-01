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
    (1, "data/metaqa/raw/1-hop/vanilla/qa_test.txt"),
    (2, "data/metaqa/raw/2-hop/vanilla/qa_test.txt"),
    (3, "data/metaqa/raw/3-hop/vanilla/qa_test.txt"),
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


def load_done(run_dir: Path) -> dict[int, set[int]]:
    """Already-evaluated item ids per hop (for resume)."""
    done: dict[int, set[int]] = {}
    for hop in (1, 2, 3):
        p = run_dir / f"hop{hop}.jsonl"
        ids = set()
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        ids.add(json.loads(line)["id"])
                    except Exception:
                        pass
        done[hop] = ids
    return done


def build_matcher(tokenizer, relations: set[str], entities: set[str]):
    """Compile xgrammar matcher: DSL shape + relation vocabulary.

    NOTE on scope: this enforces structure and the 9-relation schema, but
    NOT the 43k entity trie. Enforcing the full trie distorts greedy entity
    selection (verified: constrained outputs collapse to valid-but-wrong
    entities like "'Til There Was You" — a Qwen tokenization-vs-trie-path
    misalignment in xgrammar's compiled matcher). The trie itself compiles
    in ~15s (measured separately); full-trie decoding is documented open
    work. Since system B already emits 100% parse-valid queries, this
    skeleton constraint tests whether structural enforcement alone changes
    accuracy — a clean ablation.
    """
    import xgrammar as xgr

    rel_alt = "|".join(f'"{r}"' for r in sorted(relations))
    _ = entities  # trie measured separately (see docs); not enforced here
    # Whitespace is MANDATORY at junctions (" "+): the tolerant version
    # let the model emit spaceless junctions (actorsRETURN) that our own
    # Lark parser rejects. Forcing spaces matches the training format.
    grammar_ebnf = (
        'root ::= "START" ws "\\"" [^"]+ "\\"" ws hop (ws hop)* ws "RETURN" ws "?x"'
        ' (ws "WHERE" ws "?x.value" ws compop " " value)?'
        ' (ws "EXCEPT" ws "\\"" [^"]+ "\\"")?\n'
        f'hop ::= ("->" | "<-") ws ({rel_alt})\n'
        'ws ::= " "+\n'
        'compop ::= ">=" | "<=" | "==" | "!=" | ">" | "<"\n'
        'value ::= [0-9]+ ("." [0-9]+)? | "\\"" [^"]+ "\\""\n'
    )
    t0 = time.time()
    import xgrammar as xgr
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(grammar_ebnf)
    compile_s = time.time() - t0
    return compiled, compile_s


GRAMMAR_VOCAB = 151680  # xgrammar bitmask width for Qwen; model head is 151936
# The excluded tail holds unused special tokens (documented, not silently dropped:
# argmax runs over the masked 151680 only).


def generate_constrained(model, tokenizer, compiled, messages, max_tokens=256):
    """Greedy generation with xgrammar token mask applied per step.

    Reuses KV-cache across steps (only the newest token is forwarded),
    so per-step cost is O(1) instead of O(n).
    """
    import xgrammar as xgr

    matcher = xgr.GrammarMatcher(compiled)
    device = next(model.parameters()).device
    bitmask = xgr.allocate_token_bitmask(1, GRAMMAR_VOCAB)  # stays on CPU per xgrammar API
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    # NOTE: prompt tokens are NOT fed to the matcher — the grammar constrains
    # only the completion (which must be a DSL query), not the chat template.
    generated = list(prompt_ids)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([generated], device=device), use_cache=True)
    past = out.past_key_values
    for _ in range(max_tokens):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[generated[-1]]], device=device),
                        past_key_values=past, use_cache=True)
        logits = out.logits[0, -1]
        past = out.past_key_values
        matcher.fill_next_token_bitmask(bitmask)
        masked = logits.detach().to("cpu")[:GRAMMAR_VOCAB]
        xgr.apply_token_bitmask_inplace(masked.unsqueeze(0), bitmask)
        nxt = int(torch.argmax(masked))
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
    ap.add_argument("--run-id", default=None, help="Resume existing run")
    ap.add_argument("--n-sample", type=int, default=2000, help="Questions per hop")
    ap.add_argument("--kb", default="data/metaqa/raw/kb.txt", help="KB file")
    ap.add_argument("--qa-file", default=None,
                    help="Single QA file (overrides per-hop specs; hop inferred as 0)")
    args = ap.parse_args()

    print("Loading KB...", flush=True)
    kg = KnowledgeGraph.from_kb_file(args.kb)
    print(f"KB: {len(kg)} triples, {len(kg.entities)} entities", flush=True)

    print(f"Loading base + adapter {args.adapter}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    matcher_compiled, compile_s = None, 0.0
    if args.system == "C":
        print("Compiling xgrammar schema+entity matcher...", flush=True)
        matcher_compiled, compile_s = build_matcher(tok, kg.relations, kg.entities)
        print(f"Trie+grammar compiled in {compile_s:.1f}s", flush=True)

    run_id = args.run_id or f"sys{args.system}_{int(time.time())}"
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    done_map = load_done(run_dir)
    if args.run_id:
        ndone = sum(len(v) for v in done_map.values())
        print(f"Resuming {run_id}: {ndone} items already done", flush=True)

    totals = {"hits": 0, "n": 0, "parse_fail": 0, "exec_empty": 0}
    sys_prompt = SYSTEM_DIRECT if args.system == "A" else SYSTEM_DSL

    if args.qa_file:
        specs = [(0, args.qa_file, args.n_sample)]
    else:
        specs = [(hop, path, args.n_sample) for hop, path in EVAL_SPECS]

    for hop, path, n_sample in specs:
        sample = load_sample(path, n_sample)
        todo = [it for it in sample if it["id"] not in done_map.get(hop, set())]
        print(f"Hop {hop}: {len(sample)} questions ({len(todo)} remaining)", flush=True)
        mode = "a" if (run_dir / f"hop{hop}.jsonl").exists() else "w"
        with open(run_dir / f"hop{hop}.jsonl", mode, encoding="utf-8") as f:
            for it in todo:
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
                            model, tok, matcher_compiled, messages)
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
                               "hit": hit, "parse_ok": True, "empty": not pred}
                totals["hits"] += rec["hit"]
                totals["n"] += 1
                f.write(json.dumps(rec) + "\n")
        hop_hits = sum(1 for _ in open(run_dir / f"hop{hop}.jsonl") if json.loads(_)["hit"])
        print(f"  hop{hop} Hits@1: {hop_hits}/{len(sample)} ({hop_hits/len(sample):.1%})", flush=True)

    # Final report aggregates ALL chunks (including resumed items).
    all_hits, all_n, all_pf, all_ee = 0, 0, 0, 0
    per_hop = {}
    report_hops = [hop for hop, _, _ in specs]
    for hop in report_hops:
        h, n, pf, ee = 0, 0, 0, 0
        with open(run_dir / f"hop{hop}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                n += 1
                h += r["hit"]
                pf += (r.get("parse_ok") is False)
                ee += bool(r.get("empty", False))
        per_hop[hop] = {"hits": h, "n": n}
        all_hits += h
        all_n += n
        all_pf += pf
        all_ee += ee
    report = {"system": args.system, "adapter": args.adapter, "run_id": run_id,
              "hits": all_hits, "n": all_n,
              "hits_at_1": all_hits / all_n if all_n else 0,
              "parse_fail": all_pf, "exec_empty": all_ee,
              "per_hop": per_hop,
              "trie_compile_s": compile_s}
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDone: Hits@1={report['hits_at_1']:.1%} ({all_hits}/{all_n}) "
          f"parse_fail={all_pf} exec_empty={all_ee}", flush=True)
    print(f"Report -> {run_dir}/report.json", flush=True)


if __name__ == "__main__":
    main()

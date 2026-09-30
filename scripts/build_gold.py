#!/usr/bin/env python3
"""Build gold queries for a full MetaQA train split (resumable by shard).

Saves JSONL: {question, dsl, answers, topic} per line.
Usage:
  PYTHONPATH=src python scripts/build_gold.py --hop 3 --num-shards 4 --shard 0
"""
import argparse
import json
import time
from pathlib import Path

from graphproof.data.gold import generate_gold, verify_gold
from graphproof.executor.graph import KnowledgeGraph

FILES = {
    1: "data/metaqa/raw/1-hop/vanilla/qa_train.txt",
    2: "data/metaqa/raw/2-hop/vanilla/qa_train.txt",
    3: "data/metaqa/raw/3-hop/vanilla/qa_train.txt",
}


def count_lines(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out-dir", default="data/gold")
    args = ap.parse_args()

    print("Loading KB...", flush=True)
    kg = KnowledgeGraph.from_kb_file("data/metaqa/raw/kb.txt")
    print(f"KB: {len(kg)} triples", flush=True)

    src = FILES[args.hop]
    total_lines = count_lines(src)
    chunk = total_lines // args.num_shards
    start = args.shard * chunk
    end = total_lines if args.shard == args.num_shards - 1 else start + chunk
    print(f"Shard {args.shard + 1}/{args.num_shards}: lines {start}-{end} of {total_lines}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gold_{args.hop}hop_s{args.shard}.jsonl"

    from graphproof.data import gold as goldmod
    index = goldmod.build_entity_index(kg)

    t0 = time.time()
    n_ok = n_no_path = n_ambig_missing = n_verified = 0
    with open(src, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= end:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            q, _, ans = line.partition("\t")
            answers = set(a for a in ans.split("|") if a)
            import re
            mentions = re.findall(r"\[([^\]]+)\]", q)
            if len(mentions) != 1:
                n_ambig_missing += 1
                continue
            cands = goldmod.candidates_in_order(mentions[0], index)
            if not cands:
                n_ambig_missing += 1
                continue
            found = None
            for n, topic in enumerate(cands):
                path, needs_except = goldmod.find_path(topic, answers, args.hop, kg)
                if path is not None:
                    found = (topic, path, needs_except)
                    break
            if found is None:
                n_no_path += 1
                continue
            topic, path, needs_except = found
            dsl = goldmod.to_dsl(topic, path, except_entity=topic if needs_except else None)
            out.write(json.dumps({"question": q, "dsl": dsl,
                                  "answers": sorted(answers), "topic": topic}) + "\n")
            n_ok += 1
            if n_ok % 5000 == 0:
                print(f"  ...{n_ok} gold written ({time.time()-t0:.0f}s)", flush=True)

    dt = time.time() - t0
    # verify the shard
    from graphproof.dsl.parser import parse_query
    from graphproof.executor.engine import execute
    rep = tot = 0
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            q = parse_query(g["dsl"])
            r = execute(q, kg)
            tot += 1
            if set(r.answers) == set(g["answers"]):
                rep += 1
    print(f"Shard done: ok={n_ok} no_path={n_no_path} skipped={n_ambig_missing} "
          f"verified={rep}/{tot} in {dt:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

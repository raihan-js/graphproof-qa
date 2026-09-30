#!/usr/bin/env python3
"""Build LoRA training pairs from gold queries.

Two formats from the same data (same base, same data — the ablation):
  System A (direct):   question -> answer string
  System B (DSL):      question -> DSL query string

Samples --n-per-hop per hop (default 10k -> 30k total), seeded shuffle.
Writes data/train/{direct,DSL}.jsonl with {prompt, completion} rows.
Usage: PYTHONPATH=src python scripts/build_train.py [--n-per-hop 10000] [--seed 42]
"""
import argparse
import json
import random
from pathlib import Path


def load_gold() -> dict[int, list[dict]]:
    gold: dict[int, list[dict]] = {}
    for hop in (1, 2, 3):
        rows = []
        for path in sorted(Path("data/gold").glob(f"gold_{hop}hop_*.jsonl")):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    rows.append(json.loads(line))
        gold[hop] = rows
    return gold


SYSTEM_PROMPT_DIRECT = (
    "Answer the movie question with the correct entity name(s), "
    "separated by ' | ' if there are several. Reply with nothing else."
)
SYSTEM_PROMPT_DSL = (
    "Translate the movie question into a path-query. Reply with nothing else.\n"
    "Grammar: START \"<entity>\" (->|<-) <relation> ... RETURN ?x "
    "[WHERE ?x.value <op> <value>] [EXCEPT \"<entity>\"]"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-hop", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/train")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gold = load_gold()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    counts = {}
    with open(out / "direct.jsonl", "w", encoding="utf-8") as fa, \
         open(out / "dsl.jsonl", "w", encoding="utf-8") as fb:
        for hop in (1, 2, 3):
            rows = gold[hop]
            rng.shuffle(rows)
            sample = rows[:args.n_per_hop]
            counts[hop] = len(sample)
            for r in sample:
                answer = " | ".join(r["answers"])
                fa.write(json.dumps({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_DIRECT},
                        {"role": "user", "content": r["question"]},
                        {"role": "assistant", "content": answer},
                    ],
                    "hop": hop,
                }) + "\n")
                fb.write(json.dumps({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_DSL},
                        {"role": "user", "content": r["question"]},
                        {"role": "assistant", "content": r["dsl"]},
                    ],
                    "hop": hop,
                }) + "\n")

    total = sum(counts.values())
    print(f"Wrote {total} pairs per format ({counts}) -> {out}/", flush=True)
    with open(out / "manifest.json", "w") as f:
        json.dump({"n_per_hop": args.n_per_hop, "seed": args.seed,
                   "counts": counts, "total": total}, f, indent=2)


if __name__ == "__main__":
    main()

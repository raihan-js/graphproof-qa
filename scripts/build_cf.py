#!/usr/bin/env python3
"""Build MetaQA-CF counterfactual conditions (seeded, reproducible).

Three conditions, written to data/cf/:
  1. swapped: 500 triples with objects swapped within the same relation.
     Executor follows the edit by construction (demonstration).
  2. renamed: 200 entities replaced by novel names unseen in training,
     in both the KB and the test questions that mention them.
     Tests grounding vs memorisation (the real result).
  3. deleted: 300 triples removed; affected test questions get the
     correct answer "not in graph". Tests abstention (the real result).

Usage: python scripts/build_cf.py [--seed 123]
"""
import argparse
import json
import random
from pathlib import Path

from graphproof.data.gold import build_entity_index
from graphproof.executor.graph import KnowledgeGraph


def load_kb_triples(path: str) -> list[tuple[str, str, str]]:
    triples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


def build_swapped(triples: list, n: int, rng: random.Random) -> tuple[list, list]:
    """Swap objects within same-relation pairs. Returns (new_triples, swaps)."""
    by_rel: dict[str, list[int]] = {}
    for i, (_, r, _) in enumerate(triples):
        by_rel.setdefault(r, []).append(i)
    # candidate pairs: same relation, different subject and object
    candidates = []
    for r, idxs in by_rel.items():
        rng.shuffle(idxs)
        for a, b in zip(idxs[::2], idxs[1::2]):
            sa, _, oa = triples[a]
            sb, _, ob = triples[b]
            if sa != sb and oa != ob:
                candidates.append((a, b))
    rng.shuffle(candidates)
    chosen = candidates[:n]
    swapped_idx = set()
    new_triples = list(triples)
    swaps = []
    for a, b in chosen:
        sa, ra, oa = triples[a]
        sb, rb, ob = triples[b]
        new_triples[a] = (sa, ra, ob)
        new_triples[b] = (sb, rb, oa)
        swapped_idx.add(a)
        swapped_idx.add(b)
        swaps.append({"a": list(triples[a]), "b": list(triples[b])})
    return new_triples, swaps


def build_renamed(triples: list, test_questions: list[str], n: int,
                  rng: random.Random) -> tuple[list, dict, list]:
    """Rename n entities appearing in test questions to novel names.
    Returns (new_triples, rename_map, rewritten_questions)."""
    import re
    # entities mentioned in test questions
    mentioned: dict[str, int] = {}
    for q in test_questions:
        for m in re.findall(r"\[([^\]]+)\]", q):
            mentioned[m.lower()] = mentioned.get(m.lower(), 0) + 1
    # map to KB entities (unique case-insensitive matches preferred)
    index: dict[str, list[str]] = {}
    for s, r, o in triples:
        index.setdefault(s.lower(), []).append(s)
        index.setdefault(o.lower(), []).append(o)
    cands = [m for m in mentioned if len(index.get(m, [])) == 1]
    cands.sort(key=lambda m: -mentioned[m])
    chosen = cands[:n]
    rename_map = {}
    used = set(index.keys())
    for m in chosen:
        real = index[m][0]
        i = 0
        while True:
            new = f"ENTITY_{i:05d}"
            if new.lower() not in used:
                break
            i += 1
        used.add(new.lower())
        rename_map[real] = new
    new_triples = [(rename_map.get(s, s), r, rename_map.get(o, o)) for s, r, o in triples]
    # rewrite questions (case-insensitive bracket replacement)
    rewritten = []
    for q in test_questions:
        def repl(mo):
            for real, new in rename_map.items():
                if mo.group(1).lower() == real.lower():
                    return f"[{new}]"
            return mo.group(0)
        rewritten.append(re.sub(r"\[([^\]]+)\]", repl, q))
    return new_triples, rename_map, rewritten


def build_deleted(triples: list, n: int, rng: random.Random) -> tuple[list, list]:
    """Remove n random triples. Returns (new_triples, deleted)."""
    idxs = rng.sample(range(len(triples)), n)
    deleted = [list(triples[i]) for i in idxs]
    drop = set(idxs)
    new_triples = [t for i, t in enumerate(triples) if i not in drop]
    return new_triples, deleted


def write_kb(triples: list, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s, r, o in triples:
            f.write(f"{s}|{r}|{o}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out-dir", default="data/cf")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading KB...", flush=True)
    triples = load_kb_triples("data/metaqa/raw/kb.txt")
    print(f"KB: {len(triples)} triples", flush=True)

    # 1. swapped
    swapped, swaps = build_swapped(triples, 500, rng)
    write_kb(swapped, out / "kb_swapped.txt")
    with open(out / "swaps.json", "w") as f:
        json.dump(swaps, f)
    print(f"Swapped: {len(swaps)} pairs -> kb_swapped.txt", flush=True)

    # 2. renamed (needs test questions per hop)
    test_qs: list[str] = []
    for hop in (1, 2, 3):
        with open(f"data/metaqa/raw/{hop}-hop/vanilla/qa_test.txt", encoding="utf-8") as f:
            for line in f:
                q = line.rstrip("\n").split("\t")[0]
                if q:
                    test_qs.append(q)
    renamed_triples, rename_map, rewritten = build_renamed(triples, test_qs, 200, rng)
    write_kb(renamed_triples, out / "kb_renamed.txt")
    with open(out / "renames.json", "w") as f:
        json.dump({"map": rename_map, "n_questions": len(test_qs)}, f, indent=2)
    with open(out / "qa_test_renamed.txt", "w", encoding="utf-8") as f:
        for q in rewritten:
            f.write(q + "\n")
    print(f"Renamed: {len(rename_map)} entities -> kb_renamed.txt", flush=True)

    # 3. deleted
    deleted_triples, deleted = build_deleted(triples, 300, rng)
    write_kb(deleted_triples, out / "kb_deleted.txt")
    with open(out / "deleted.json", "w") as f:
        json.dump(deleted, f)
    print(f"Deleted: {len(deleted)} triples -> kb_deleted.txt", flush=True)

    with open(out / "manifest.json", "w") as f:
        json.dump({"seed": args.seed, "swapped_pairs": len(swaps),
                   "renamed_entities": len(rename_map),
                   "deleted_triples": len(deleted)}, f, indent=2)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()

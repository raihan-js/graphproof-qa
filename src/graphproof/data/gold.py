"""Gold query generation: (question, answers) -> DSL query via KB search.

For each question, resolve the [entity] mention, then BFS for paths of
exactly `hops` length from the topic entity that reach the gold answers.
The shortest covering path becomes the gold DSL query.

Questions whose entity is ambiguous or missing are skipped with a recorded
reason (they form the explained remainder and the ambiguity failure class).
"""

import json
import re
from collections import defaultdict

from ..executor.graph import KnowledgeGraph


def build_entity_index(kg: KnowledgeGraph) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for e in kg.entities:
        index[e.lower()].append(e)
    return index


def resolve_entity(mention: str, index: dict[str, list[str]]) -> tuple[str | None, str]:
    cands = index.get(mention.lower(), [])
    if len(cands) == 1:
        return cands[0], "ok"
    if len(cands) > 1:
        return None, f"ambiguous:{len(cands)}"
    return None, "missing"


def candidates_in_order(mention: str, index: dict[str, list[str]]) -> list[str]:
    """Candidate entities for a mention: exact-case match first, then the rest.
    Surface casing is evidence (tags are lowercase in MetaQA); the gold
    answers disambiguate among the rest (see generate_gold)."""
    cands = index.get(mention.lower(), [])
    exact = [c for c in cands if c == mention]
    rest = [c for c in cands if c != mention]
    return exact + rest


def find_path(topic: str, answers: set[str], hops: int, kg: KnowledgeGraph) -> tuple[list | None, bool]:
    """BFS over relation sequences: state = (current node set, sequence).
    At most (2*|R|)^hops distinct sequences — tractable for hops <= 3.
    Endpoint sets are computed WITH topic-exclusion at every step (matching
    the EXCEPT semantics the executor applies). Requires exact equality
    with gold. This rejects spurious covering paths (e.g. routing via genre
    when the question asks about shared actors).
    Returns (hops, needs_except): needs_except tells whether the topic
    actually appeared in any raw result (i.e. the gold DSL must carry
    the EXCEPT clause)."""
    rels = sorted(kg.relations)
    # frontier: list of (node_set, seq). The topic is excluded from every
    # intermediate set, mirroring the executor's EXCEPT semantics.
    frontier: list[tuple[set[str], list[tuple[str, bool]]]] = [({topic}, [])]
    for _ in range(hops):
        nxt: dict[tuple, set[str]] = {}
        for nodes, seq in frontier:
            for rel in rels:
                for fwd in (True, False):
                    reached: set[str] = set()
                    for n in nodes:
                        reached.update(kg.neighbors(n, rel, fwd))
                    reached.discard(topic)
                    if not reached:
                        continue
                    key = tuple(seq + [(rel, fwd)])
                    if key in nxt:
                        nxt[key] |= reached
                    else:
                        nxt[key] = set(reached)
        frontier = [(nodes, list(seq)) for seq, nodes in nxt.items()]
        if not frontier:
            return None, False
    for nodes, seq in frontier:
        if answers == nodes:
            # needs_except: did the topic appear in any RAW intermediate
            # set? Pollution enters mid-path (e.g. the topic film reappears
            # in step 2, contributing its own directors in step 3).
            raw: set[str] = {topic}
            needs_except = False
            for rel, fwd in seq:
                step: set[str] = set()
                for n in raw:
                    step.update(kg.neighbors(n, rel, fwd))
                raw = step
                if topic in raw:
                    needs_except = True
            return [{"relation": rel, "forward": fwd} for rel, fwd in seq], needs_except
    return None, False


def to_dsl(topic: str, hops: list[dict], except_entity: str | None = None) -> str:
    parts = [f'START "{topic}"']
    for h in hops:
        arrow = "->" if h["forward"] else "<-"
        parts.append(f"{arrow} {h['relation']}")
    parts.append("RETURN ?x")
    if except_entity is not None:
        parts.append(f'EXCEPT "{except_entity}"')
    return " ".join(parts)


def verify_gold(gold: list[dict], kg: KnowledgeGraph) -> tuple[int, int, list[dict]]:
    """Execute each gold query and check exact reproduction.

    The executor applies each query's EXCEPT clause, so comparison is
    direct: executed answers vs gold answers. Returns (reproduced, total, failures).
    """
    from ..dsl.parser import parse_query
    from ..executor.engine import execute

    reproduced = 0
    failures: list[dict] = []
    for g in gold:
        q = parse_query(g["dsl"])
        r = execute(q, kg)
        if set(r.answers) == set(g["answers"]):
            reproduced += 1
        else:
            failures.append({"question": g["question"], "dsl": g["dsl"],
                             "gold": sorted(g["answers"]),
                             "got": sorted(r.answers)})
    return reproduced, len(gold), failures


def generate_gold(qa_path: str, hops: int, kg: KnowledgeGraph,
                  limit: int | None = None) -> tuple[list[dict], dict]:
    """Generate gold (question, dsl, answers) triples. Returns (gold, stats).

    Entity resolution is answer-guided: candidates are tried exact-case
    first, and the first candidate yielding an exact path wins. This
    disambiguates MetaQA's case-variant duplicates (e.g. tag "ginger
    rogers" vs person "Ginger Rogers") using the gold answers — valid
    for training-data construction, and every emitted query is still
    verified by execution."""
    index = build_entity_index(kg)
    gold: list[dict] = []
    stats = {"total": 0, "ok": 0, "ambiguous": 0, "missing": 0, "no_path": 0,
             "disambiguated": 0}
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            stats["total"] += 1
            if limit is not None and stats["total"] > limit:
                break
            q, _, ans = line.partition("\t")
            answers = set(a for a in ans.split("|") if a)
            mentions = re.findall(r"\[([^\]]+)\]", q)
            if len(mentions) != 1:
                stats["missing"] += 1
                continue
            cands = candidates_in_order(mentions[0], index)
            if not cands:
                stats["missing"] += 1
                continue
            if len(cands) > 1:
                stats["ambiguous"] += 1
            found = None
            for n, topic in enumerate(cands):
                path, needs_except = find_path(topic, answers, hops, kg)
                if path is not None:
                    found = (topic, path, needs_except, n > 0)
                    break
            if found is None:
                stats["no_path"] += 1
                continue
            topic, path, needs_except, was_ambiguous = found
            stats["ok"] += 1
            if was_ambiguous:
                stats["disambiguated"] += 1
            dsl = to_dsl(topic, path, except_entity=topic if needs_except else None)
            gold.append({"question": q, "dsl": dsl,
                         "answers": sorted(answers), "topic": topic})
    return gold, stats

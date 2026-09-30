"""Executor: run a Query AST over a KnowledgeGraph, with proof traces."""

from dataclasses import dataclass, field

from ..dsl.parser import Filter, Query
from .graph import KnowledgeGraph


@dataclass
class ProofStep:
    node: str
    relation: str
    forward: bool
    reached: list[str]


@dataclass
class ExecutionResult:
    answers: list[str]
    proof: list[ProofStep]
    ok: bool
    error: str = ""


def _apply_filter(values: list[str], filt: Filter | None) -> list[str]:
    if filt is None:
        return values
    out = []
    for v in values:
        try:
            lhs = float(v)
            rhs = float(filt.value)
        except (ValueError, TypeError):
            lhs, rhs = str(v), str(filt.value)
        ok = {
            ">=": lhs >= rhs,
            "<=": lhs <= rhs,
            "==": lhs == rhs,
            "!=": lhs != rhs,
            ">": lhs > rhs,
            "<": lhs < rhs,
        }[filt.op]
        if ok:
            out.append(v)
    return out


def execute(query: Query, kg: KnowledgeGraph) -> ExecutionResult:
    """Execute a query, returning answers plus the traversed path as proof.

    If the query carries EXCEPT <entity>, that entity is removed from every
    intermediate node set and from the final answers. This encodes MetaQA's
    answer convention ("movies like X" never lists X, paths may not route
    back through X) as an explicit, checkable part of the query.
    """
    if query.start_entity not in kg.entities:
        return ExecutionResult(answers=[], proof=[], ok=False,
                               error=f"unknown start entity: {query.start_entity!r}")

    excluded = query.except_entity
    current = [query.start_entity]
    proof: list[ProofStep] = []

    for hop in query.hops:
        nxt: list[str] = []
        for node in current:
            reached = kg.neighbors(node, hop.relation, hop.forward)
            if excluded is not None:
                reached = [r for r in reached if r != excluded]
            proof.append(ProofStep(node=node, relation=hop.relation,
                                   forward=hop.forward, reached=list(reached)))
            nxt.extend(reached)
        # de-duplicate, preserve order
        current = list(dict.fromkeys(nxt))
        if not current:
            break

    current = _apply_filter(current, query.filter)
    if excluded is not None:
        current = [c for c in current if c != excluded]
    return ExecutionResult(answers=current, proof=proof, ok=True)


def render_proof(result: ExecutionResult, query: Query) -> str:
    """Render the executed path as a human-readable proof trace."""
    lines = [f"QUERY: START {query.start_entity!r}"]
    if query.except_entity:
        lines.append(f"EXCEPT {query.except_entity!r}")
    for step in result.proof:
        arrow = "->" if step.forward else "<-"
        reached = ", ".join(step.reached) if step.reached else "(nothing)"
        lines.append(f"  {step.node!r} {arrow} [{step.relation}] => {reached}")
    lines.append(f"ANSWER: {', '.join(result.answers) if result.answers else '(not in graph)'}")
    return "\n".join(lines)

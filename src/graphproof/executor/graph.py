"""Knowledge-graph index: MetaQA kb.txt -> adjacency dicts."""

from collections import defaultdict


class KnowledgeGraph:
    """Directed multigraph with forward and backward adjacency.

    Triples are (subject, relation, object) strings.
    """

    def __init__(self) -> None:
        self.forward: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.backward: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.entities: set[str] = set()
        self.relations: set[str] = set()
        self.n_triples: int = 0

    @classmethod
    def from_kb_file(cls, path: str) -> "KnowledgeGraph":
        kg = cls()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) != 3:
                    continue
                kg.add(*parts)
        return kg

    def add(self, subject: str, relation: str, obj: str) -> None:
        self.forward[(subject, relation)].append(obj)
        self.backward[(obj, relation)].append(subject)
        self.entities.add(subject)
        self.entities.add(obj)
        self.relations.add(relation)
        self.n_triples += 1

    def neighbors(self, node: str, relation: str, forward: bool) -> list[str]:
        index = self.forward if forward else self.backward
        return list(index.get((node, relation), []))

    def __len__(self) -> int:
        return self.n_triples

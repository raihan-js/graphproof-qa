"""DSL parser: Lark parse tree -> typed AST."""

from dataclasses import dataclass, field
from pathlib import Path

from lark import Lark, Transformer

_GRAMMAR = Path(__file__).with_name("grammar.lark").read_text()


@dataclass(frozen=True)
class Hop:
    relation: str
    forward: bool  # True for ->, False for <-


@dataclass(frozen=True)
class Filter:
    var: str
    attr: str
    op: str
    value: str | float


@dataclass(frozen=True)
class Except:
    entity: str


@dataclass(frozen=True)
class Query:
    start_entity: str
    hops: tuple[Hop, ...]
    return_var: str
    filter: Filter | None = None
    except_entity: str | None = None


class _ToAST(Transformer):
    def start(self, items):
        return items[0]

    def query(self, items):
        # items: [entity, Hop..., var, (Filter)?, (Except)?] in any order after entity
        entity = items[0]
        hops = tuple(i for i in items[1:] if isinstance(i, Hop))
        var = next(i for i in items[1:] if isinstance(i, str))
        filt = next((i for i in items[1:] if isinstance(i, Filter)), None)
        exc = next((i for i in items[1:] if isinstance(i, Except)), None)
        return Query(start_entity=entity, hops=hops, return_var=var,
                     filter=filt, except_entity=exc.entity if exc else None)

    def entity(self, items):
        return items[0][1:-1]  # strip quotes

    def hop(self, items):
        arrow, relation = items
        return Hop(relation=str(relation), forward=(str(arrow) == "->"))

    def ARROW(self, tok):
        return str(tok)

    def filter_clause(self, items):
        var, attr, op, value = items
        return Filter(var=str(var), attr=str(attr), op=str(op), value=value)

    def except_clause(self, items):
        return Except(entity=items[0])

    def var(self, items):
        return "?" + str(items[0])

    def value(self, items):
        tok = items[0]
        try:
            return float(tok)
        except (ValueError, TypeError):
            s = str(tok)
            return s[1:-1] if s.startswith('"') else s

    def RELATION(self, tok):
        return str(tok)

    def ATTRNAME(self, tok):
        return str(tok)

    def COMPOP(self, tok):
        return str(tok)


_parser = Lark(_GRAMMAR, parser="lalr", transformer=_ToAST())


def parse_query(text: str) -> Query:
    """Parse DSL source into a Query AST. Raises on syntax error."""
    return _parser.parse(text)

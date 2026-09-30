"""Tests for the DSL parser (hand-made cases)."""

import pytest
from graphproof.dsl.parser import parse_query, Query, Hop


class TestParseQueries:
    def test_one_hop(self):
        q = parse_query('START "Forrest Gump" -> directed_by RETURN ?x')
        assert q == Query(start_entity="Forrest Gump",
                          hops=(Hop(relation="directed_by", forward=True),),
                          return_var="?x", filter=None)

    def test_two_hop_mixed_direction(self):
        q = parse_query('START "Tom Hanks" -> acted_in <- starred_actors RETURN ?x')
        assert q.start_entity == "Tom Hanks"
        assert q.hops == (Hop(relation="acted_in", forward=True),
                          Hop(relation="starred_actors", forward=False))
        assert q.return_var == "?x"

    def test_filter_numeric(self):
        q = parse_query('START "Forrest Gump" -> release_year RETURN ?x WHERE ?x.value >= 1990')
        assert q.filter is not None
        assert (q.filter.var, q.filter.attr, q.filter.op, q.filter.value) == ("?x", "value", ">=", 1990.0)

    def test_filter_string(self):
        q = parse_query('START "X" -> genre RETURN ?x WHERE ?x.value == "Drama"')
        assert q.filter is not None
        assert q.filter.value == "Drama"

    def test_syntax_error(self):
        with pytest.raises(Exception):
            parse_query('START "X" RETURN ?x')  # no hops

    def test_unknown_relation_chars_rejected(self):
        with pytest.raises(Exception):
            parse_query('START "X" -> Directed-By RETURN ?x')

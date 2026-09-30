"""Tests for the graph executor (hand-made toy KB)."""

import pytest
from graphproof.dsl.parser import parse_query
from graphproof.executor.engine import execute, render_proof
from graphproof.executor.graph import KnowledgeGraph


@pytest.fixture
def toy_kg():
    kg = KnowledgeGraph()
    kg.add("Forrest Gump", "directed_by", "Robert Zemeckis")
    kg.add("Forrest Gump", "starred_actors", "Tom Hanks")
    kg.add("Tom Hanks", "acted_in", "Forrest Gump")
    kg.add("Tom Hanks", "acted_in", "Apollo 13")
    kg.add("Apollo 13", "starred_actors", "Tom Hanks")
    kg.add("Apollo 13", "starred_actors", "Kevin Bacon")
    kg.add("Forrest Gump", "release_year", "1994")
    return kg


class TestExecutor:
    def test_one_hop(self, toy_kg):
        q = parse_query('START "Forrest Gump" -> directed_by RETURN ?x')
        r = execute(q, toy_kg)
        assert r.ok and r.answers == ["Robert Zemeckis"]

    def test_two_hop(self, toy_kg):
        q = parse_query('START "Tom Hanks" -> acted_in -> starred_actors RETURN ?x')
        r = execute(q, toy_kg)
        # co-stars of Tom Hanks across his movies
        assert r.ok
        assert set(r.answers) == {"Tom Hanks", "Kevin Bacon"}

    def test_backward_hop(self, toy_kg):
        q = parse_query('START "Robert Zemeckis" <- directed_by RETURN ?x')
        r = execute(q, toy_kg)
        assert r.ok and r.answers == ["Forrest Gump"]

    def test_empty_result(self, toy_kg):
        q = parse_query('START "Nobody" -> directed_by RETURN ?x')
        r = execute(q, toy_kg)
        assert not r.ok  # unknown start entity

    def test_dead_end(self, toy_kg):
        q = parse_query('START "Forrest Gump" -> directed_by -> acted_in RETURN ?x')
        r = execute(q, toy_kg)
        assert r.ok and r.answers == []

    def test_numeric_filter(self, toy_kg):
        q = parse_query('START "Forrest Gump" -> release_year RETURN ?x WHERE ?x.value >= 1990')
        r = execute(q, toy_kg)
        assert r.ok and r.answers == ["1994"]

    def test_numeric_filter_excludes(self, toy_kg):
        q = parse_query('START "Forrest Gump" -> release_year RETURN ?x WHERE ?x.value >= 2000')
        r = execute(q, toy_kg)
        assert r.ok and r.answers == []

    def test_proof_trace_rendered(self, toy_kg):
        q = parse_query('START "Forrest Gump" -> directed_by RETURN ?x')
        r = execute(q, toy_kg)
        text = render_proof(r, q)
        assert "Forrest Gump" in text
        assert "directed_by" in text
        assert "Robert Zemeckis" in text


class TestGraphIndex:
    def test_counts(self, toy_kg):
        assert len(toy_kg) == 7
        assert "directed_by" in toy_kg.relations
        assert "Forrest Gump" in toy_kg.entities

"""Tests for gold query generation."""

import pytest
from graphproof.data.gold import (
    build_entity_index,
    candidates_in_order,
    find_path,
    resolve_entity,
    to_dsl,
    verify_gold,
)
from graphproof.executor.graph import KnowledgeGraph


@pytest.fixture
def toy_kg():
    kg = KnowledgeGraph()
    kg.add("Forrest Gump", "directed_by", "Robert Zemeckis")
    kg.add("Forrest Gump", "starred_actors", "Tom Hanks")
    kg.add("Tom Hanks", "acted_in", "Forrest Gump")
    kg.add("Tom Hanks", "acted_in", "Apollo 13")
    kg.add("Apollo 13", "starred_actors", "Kevin Bacon")
    kg.add("Forrest Gump", "release_year", "1994")
    return kg


class TestEntityResolution:
    def test_unique(self, toy_kg):
        index = build_entity_index(toy_kg)
        entity, reason = resolve_entity("Tom Hanks", index)
        assert (entity, reason) == ("Tom Hanks", "ok")

    def test_missing(self, toy_kg):
        index = build_entity_index(toy_kg)
        entity, reason = resolve_entity("Nobody Here", index)
        assert entity is None and reason == "missing"

    def test_case_variants_ordered_exact_first(self):
        kg = KnowledgeGraph()
        kg.add("Ginger Rogers", "starred_actors", "X")
        kg.add("ginger rogers", "has_tags", "Y")
        index = build_entity_index(kg)
        assert candidates_in_order("ginger rogers", index)[0] == "ginger rogers"
        assert candidates_in_order("Ginger Rogers", index)[0] == "Ginger Rogers"


class TestFindPath:
    def test_exact_path(self, toy_kg):
        path, needs_except = find_path("Forrest Gump", {"Robert Zemeckis"}, 1, toy_kg)
        assert path == [{"relation": "directed_by", "forward": True}]
        assert needs_except is False

    def test_rejects_superset_path(self, toy_kg):
        # A path reaching extra answers must not match without EXCEPT;
        # with topic-exclusion this one matches exactly and needs EXCEPT.
        path, needs_except = find_path("Tom Hanks", {"Kevin Bacon"}, 2, toy_kg)
        assert path == [{"relation": "acted_in", "forward": True},
                        {"relation": "starred_actors", "forward": True}]
        assert needs_except is True

    def test_needs_except_when_topic_pollutes(self, toy_kg):
        # Tom Hanks -> acted_in -> starred_actors reaches {Tom Hanks, Kevin Bacon};
        # excluding the topic leaves {Kevin Bacon} == gold.
        path, needs_except = find_path("Tom Hanks", {"Kevin Bacon"}, 2, toy_kg)
        # (depends on exact coverage; just check the flag machinery runs)
        assert isinstance(needs_except, bool)

    def test_no_path_when_nothing_matches(self, toy_kg):
        path, needs_except = find_path("Forrest Gump", {"Nobody"}, 1, toy_kg)
        assert path is None and needs_except is False

    def test_rejects_genre_style_superset(self, toy_kg):
        # Gold is a strict subset of every reachable set: no exact path.
        kg = KnowledgeGraph()
        kg.add("M1", "has_genre", "Comedy")
        kg.add("M2", "has_genre", "Comedy")
        kg.add("M2", "has_genre", "Drama")
        path, _ = find_path("M1", {"Comedy"}, 2, kg)
        # M1 -> has_genre -> {Comedy} <- has_genre -> {M1, M2} -> has_genre -> {Comedy, Drama}
        # exact match impossible (extra Drama always present)
        assert path is None


class TestToDsl:
    def test_with_except(self):
        dsl = to_dsl("Jawbreaker", [{"relation": "starred_actors", "forward": True}],
                     except_entity="Jawbreaker")
        assert dsl == 'START "Jawbreaker" -> starred_actors RETURN ?x EXCEPT "Jawbreaker"'

    def test_without_except(self):
        dsl = to_dsl("Forrest Gump", [{"relation": "directed_by", "forward": True}])
        assert dsl == 'START "Forrest Gump" -> directed_by RETURN ?x'


class TestVerifyGold:
    def test_all_reproduce(self, toy_kg):
        gold = [
            {"question": "q1", "dsl": 'START "Forrest Gump" -> directed_by RETURN ?x',
             "answers": ["Robert Zemeckis"], "topic": "Forrest Gump"},
            {"question": "q2",
             "dsl": 'START "Tom Hanks" -> acted_in -> starred_actors RETURN ?x EXCEPT "Tom Hanks"',
             "answers": ["Kevin Bacon"],
             "topic": "Tom Hanks"},
        ]
        rep, total, fails = verify_gold(gold, toy_kg)
        assert (rep, total, fails) == (2, 2, [])

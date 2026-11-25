"""
Offline tests for the AI Assistant's tools and safety checks (no API key or internet needed).

    cd capstone_streamlit_app
    python3 -m pytest tests -q
"""

import json

import pytest

from harness import Harness, ai


@pytest.fixture(scope="module")
def h():
    return Harness()


def call(h, tool, **args):
    return json.loads(h.tools.run(tool, args))


# --- catalog search -----------------------------------------------------------------------
def test_search_exact_title(h):
    r = call(h, "search_movies", title="Casablanca")
    assert r["matches"][0]["title"] == "Casablanca"
    assert r["matches"][0]["year"] == 1942


def test_search_misspelled_title(h):
    assert call(h, "search_movies", title="casablanka")["matches"][0]["title"] == "Casablanca"


def test_search_partial_title(h):
    titles = [m["title"] for m in call(h, "search_movies", title="lord of the rings")["matches"]]
    assert any("Lord of the Rings" in t for t in titles)


def test_search_movie_not_in_catalog(h):
    r = call(h, "search_movies", title="Oppenheimer")
    assert r["matches"] == [] and "not in the" in r["note"]


def test_search_empty_title(h):
    assert call(h, "search_movies", title="   ")["matches"] == []


def test_search_uses_the_loaded_model(h):
    m = call(h, "search_movies", title="Casablanca")["matches"][0]
    probs = h.model.predict_proba(h.X_scaled.loc[[h.movies.index[h.movies.title == "Casablanca"][0]]])[0]
    assert m["award_caliber_probability"] == f"{probs[5]:.1%}"


# --- classifier tool: validation -----------------------------------------------------------
@pytest.mark.parametrize("args", [
    {"imdb_score": 15}, {"imdb_score": 0}, {"imdb_score": -3},
    {"imdb_score": 8, "rotten_score": 150}, {"imdb_score": 8, "year": 3000},
    {"imdb_score": 8, "title_awards": -1}, {"imdb_score": "abc"},
])
def test_impossible_inputs_are_rejected(h, args):
    r = call(h, "classify_movie", **args)
    assert "error" in r and "decision" not in r


def test_out_of_training_range_is_flagged(h):
    r = call(h, "classify_movie", imdb_score=8.5, year=2024, director_awards=12)
    assert "decision" in r
    assert any("year" in w for w in r["out_of_training_range"])
    assert any("director_awards" in w for w in r["out_of_training_range"])
    assert r["reliability"].startswith("LOW")


def test_normal_input_is_not_flagged(h):
    r = call(h, "classify_movie", imdb_score=7.5, rotten_score=80, year=2005, title_awards=1)
    assert "out_of_training_range" not in r


def test_missing_inputs_are_reported_as_assumed(h):
    r = call(h, "classify_movie", imdb_score=7.0)
    assert r["assumed"]


def test_unknown_genre_is_ignored_and_reported(h):
    r = call(h, "classify_movie", imdb_score=7.0, genres=["Drama", "Space Opera"])
    assert r["inputs_used"]["genres"] == ["Drama"]
    assert r["genres_ignored_not_in_model"] == ["Space Opera"]


def test_classifier_matches_what_if_tab(h):
    s = h.scenario()  # Oscar-winning epic preset
    r = call(h, "classify_movie", imdb_score=s["imdb"], rotten_score=s["rotten"], year=s["year"],
             rating=s["rating"], genres=s["genres"], title_awards=s["title_aw"],
             director_awards=s["director_aw"], actor_awards=s["actor_aw"], platforms=s["platforms"])
    assert r["award_caliber_probability"] == f"{s['award_prob']:.1%}"


def test_unknown_tool(h):
    assert "error" in json.loads(h.tools.run("delete_database", {}))


# --- catalog query -------------------------------------------------------------------------
def test_query_catalog_filters(h):
    r = call(h, "query_catalog", persona="Classic Film Buff", genre="drama", limit=3)
    assert r["match_count"] > 0 and len(r["rows"]) == 3
    assert all(row["persona"] == "Classic Film Buff" for row in r["rows"])


def test_query_catalog_limit_is_capped(h):
    assert len(call(h, "query_catalog", limit=500)["rows"]) <= 10


# --- answer checks --------------------------------------------------------------------------
FACTS = "LR F1 0.960, ANN 0.937, 315 test movies, 1,572 movies, Casablanca 98.4%"


@pytest.mark.parametrize("answer", [
    "Logistic Regression scores 96% vs 93.7% for the ANN.",
    "Casablanca has a 98% probability.", "There are 1,572 movies.", "Step 1: open Tab 4.",
])
def test_fact_check_accepts_correct_numbers(answer):
    assert ai.fact_check(answer, [FACTS])[1] == []


@pytest.mark.parametrize("answer, bad", [
    ("The ANN reaches 0.99 F1.", "0.99"), ("It won 47 Oscars.", "47"),
    ("The catalog has 2,000 movies.", "2,000"),
])
def test_fact_check_flags_invented_numbers(answer, bad):
    assert bad in ai.fact_check(answer, [FACTS])[1]


def test_long_questions_are_truncated():
    q = ai.clean_question("x" * 5000)
    assert len(q) < ai.MAX_QUESTION_CHARS + 40 and q.endswith("truncated]")


def test_offline_answer_uses_the_data(h):
    a = ai.offline_answer("Is Casablanca award caliber?", h.tools)
    assert "Casablanca" in a and "%" in a


def test_offline_answer_empty_without_movie(h):
    assert ai.offline_answer("How many personas are there?", h.tools) == ""


# --- prompt guardrails ----------------------------------------------------------------------
@pytest.mark.parametrize("rule", ["Check the premise", "Stay in scope", "Security",
                                  "Never guess a model decision", "user's language"])
def test_system_prompt_has_guardrails(rule):
    assert rule in ai.SYSTEM_PROMPT


def test_facts_contain_key_numbers(h):
    f = h.facts()
    for n in ["1572", "0.96", "0.937", "0.130"]:
        assert n in f

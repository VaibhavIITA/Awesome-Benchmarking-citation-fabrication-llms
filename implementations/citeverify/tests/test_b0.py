"""Hand-made inputs only (not real model output)."""
from src import b0_rules as b0

GOOD = {"doi": "10.1145/3442188.3445922", "year": 2021,
        "authors": "Bender, E. and Gebru, T.", "title": "On the Dangers of Stochastic Parrots"}


def test_clean_citation_has_no_flags():
    assert b0.flags(GOOD) == []
    assert b0.predict(GOOD, 2) == "not_fabricated"


def test_each_rule_fires():
    assert "no_doi" in b0.flags({**GOOD, "doi": None})
    assert "bad_doi" in b0.flags({**GOOD, "doi": "doi:abc"})
    assert "no_year" in b0.flags({**GOOD, "year": None})
    assert "bad_year" in b0.flags({**GOOD, "year": 2031})
    assert "no_authors" in b0.flags({**GOOD, "authors": ""})
    assert "short_title" in b0.flags({**GOOD, "title": "Parrots"})


def test_threshold_and_score():
    c = {"doi": None, "year": None, "authors": None, "title": "x"}
    assert len(b0.flags(c)) == 4
    assert b0.predict(c, 2) == "fabricated"
    assert b0.score(c) == 4 / len(b0.RULES)

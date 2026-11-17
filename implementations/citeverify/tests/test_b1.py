"""Hand-made inputs only (not real model output)."""
import numpy as np

from src import b1_detector as b1


def test_no_database_features():
    assert not b1.FORBIDDEN & set(b1.NAMES)


def test_feats_on_empty_and_full():
    assert all(v == 0 for v in b1.feats({}).values())
    f = b1.feats({"raw": "Smith, J. et al. (2020). A Study of X: Y. Journal of Z, 12(3), 1-10. "
                        "https://doi.org/10.1145/12345",
                  "title": "A Study of X: Y", "authors": "Smith, J. et al.",
                  "doi": "10.1145/12345", "year": 2020})
    assert f["has_doi"] == 1 and f["doi_known_prefix"] == 1 and f["doi_suffix_len"] == 5
    assert f["et_al"] == 1 and f["venue_word"] == 1 and f["has_pages"] == 1
    assert f["has_volume"] == 1 and f["title_colon"] == 1 and f["age"] == 6


def test_matrix_shape_and_fit():
    rows = [{"doi": "10.1145/1", "year": 2020, "title": "a b c d", "raw": "x"}] * 10 + \
           [{"doi": None, "year": None, "title": "a", "raw": "y"}] * 10
    X = b1.matrix(rows)
    assert X.shape == (20, len(b1.NAMES))
    y = np.array([0] * 10 + [1] * 10)
    p = b1.make().fit(X, y).predict_proba(X)[:, 1]
    assert p[10:].min() > p[:10].max()


def test_best_threshold_prefers_higher_on_tie():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert b1.best_threshold(y, p) == 0.8

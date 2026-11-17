"""B1: string-only fabrication detector (self-built, after change notice D08).

Follows the idea described by Naser (2026): predict whether a citation is
fabricated from its bibliographic string alone, with no database lookup.
The exact features of Naser's classifier are NOT public (his code is not
accessible), so these features are my own choice and are listed in full.

Training labels come from B2 (database matching). Evaluation is against
human gold labels, so B1 is never scored against the labels it learned from.
"""
import re

import numpy as np

KNOWN = ("10.1145", "10.1109", "10.1007", "10.1016", "10.18653", "10.48550",
         "10.1038", "10.1126", "10.1371", "10.3390", "10.1093", "10.1002", "10.1111")
VENUE = re.compile(r"\b(journal|proceedings|conference|transactions|review|letters|"
                   r"arxiv|symposium|workshop|magazine)\b", re.I)
PAGES = re.compile(r"\b\d+\s*[-–]\s*\d+\b")
VOL = re.compile(r"\b\d+\s*\(\d+\)|\bvol\.?\s*\d+", re.I)

# Features that would leak a database lookup are forbidden here.
FORBIDDEN = {"best_sim", "surname_ok", "year_ok", "doi_ok", "doi_resolves", "doi_sim"}


def feats(c, now=2026):
    raw = c.get("raw") or ""
    t = c.get("title") or ""
    a = c.get("authors") or ""
    d = c.get("doi") or ""
    y = c.get("year") or 0
    letters = [ch for ch in t if ch.isalpha()]
    return {
        "n_chars": len(raw),
        "n_title_words": len(t.split()),
        "n_authors": (a.count(",") // 2 + 1 + a.count(" and ") + a.count("&")) if a else 0,
        "et_al": int("et al" in raw.lower()),
        "has_doi": int(bool(d)),
        "doi_known_prefix": int(d.startswith(KNOWN)),
        "doi_suffix_len": len(d.split("/", 1)[1]) if "/" in d else 0,
        "has_year": int(bool(y)),
        "age": (now - y) if y else 0,
        "has_pages": int(bool(PAGES.search(raw))),
        "has_volume": int(bool(VOL.search(raw))),
        "venue_word": int(bool(VENUE.search(raw))),
        "title_colon": int(":" in t),
        "title_upper": (sum(ch.isupper() for ch in letters) / len(letters)) if letters else 0.0,
        "has_url": int("http" in raw.lower()),
        "n_digits": sum(ch.isdigit() for ch in raw),
    }


NAMES = list(feats({}).keys())
assert not FORBIDDEN & set(NAMES), "database-derived feature leaked into B1"


DOI_FEATS = ["has_doi", "doi_known_prefix", "doi_suffix_len"]


def matrix(rows, names=None):
    names = names or NAMES
    return np.array([[feats(r)[k] for k in names] for r in rows], dtype=float)


def make(C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=2000))


def best_threshold(y, p):
    """Threshold maximising F1 on TRAINING predictions (ties -> higher threshold)."""
    best, bt = -1.0, 0.5
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        pr = p >= t
        tp = int((pr & (y == 1)).sum())
        fp = int((pr & (y == 0)).sum())
        fn = int((~pr & (y == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 >= best:
            best, bt = f1, float(t)
    return bt

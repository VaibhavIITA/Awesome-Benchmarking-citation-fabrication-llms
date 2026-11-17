"""B0: rule baseline. Flags a citation from its own text only (no database, no model).

Each rule is one red flag. Score = flags / number of rules (for ROC-AUC).
Decision: fabricated if flags >= cfg["b0"]["min_flags"] (frozen before testing).
"""
import re

DOI_OK = re.compile(r"^10\.\d{4,9}/\S{3,}$")
RULES = ["no_doi", "bad_doi", "no_year", "bad_year", "no_authors", "short_title"]


def flags(c, now=2026):
    f = []
    d = c.get("doi")
    if not d:
        f.append("no_doi")
    elif not DOI_OK.match(d):
        f.append("bad_doi")
    y = c.get("year")
    if not y:
        f.append("no_year")
    elif y > now or y < 1950:
        f.append("bad_year")
    if not c.get("authors"):
        f.append("no_authors")
    t = c.get("title") or ""
    if len(t.split()) < 4:
        f.append("short_title")
    return f


def score(c):
    return len(flags(c)) / len(RULES)


def predict(c, min_flags):
    return "fabricated" if len(flags(c)) >= min_flags else "not_fabricated"

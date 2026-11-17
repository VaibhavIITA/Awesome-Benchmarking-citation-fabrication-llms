"""Check each parsed citation against CrossRef and OpenAlex.

For every citation we store the best-matching candidate and its similarity,
plus surname / year / DOI checks. The label is then a pure function of the
stored evidence and the threshold t (see `label_at`), so the threshold
sensitivity analysis needs no new API calls.

Labels: exists | metadata_error | fabricated | unparseable

Run: python -m src.verify --mode main
Output: data/labels/auto_<mode>.jsonl
"""
import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import requests
from rapidfuzz import fuzz

from src.common import load_cfg, read_jsonl, write_jsonl

CR = "https://api.crossref.org/works"
OA = "https://api.openalex.org/works"
DC = "https://api.datacite.org/dois"   # arXiv (10.48550) and Zenodo (10.5281) DOIs live here, not in CrossRef


def norm(s):
    # strip accents first ("Mądry" -> "madry", "Dollár" -> "dollar"); pilot finding, 17 Nov 2026
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).replace("ł", "l").replace("Ł", "L")
    s = s.lower()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def sim(a, b):
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(norm(a), norm(b)) / 100.0


def surname(full):
    t = norm(full).split()
    return t[-1] if t else ""


class Client:
    """Cached HTTP GET with retry. `get` is injectable for tests."""

    def __init__(self, cache, mail="", okey="", get=requests.get, pause=0.1, tries=5):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.mail, self.okey, self.get, self.pause, self.tries = mail, okey, get, pause, tries
        self.calls = 0          # real (non-cached) API calls, for the cost comparison

    def fetch(self, url, params):
        pub = {k: v for k, v in sorted(params.items()) if k != "api_key"}
        h = hashlib.sha1(json.dumps([url, pub]).encode()).hexdigest()
        f = self.cache / f"{h}.json"
        if f.exists():
            return json.loads(f.read_text())
        for i in range(self.tries):
            self.calls += 1
            r = self.get(url, params=params, timeout=30)
            if r.status_code == 404:
                data = None
                break
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** i)
                continue
            r.raise_for_status()
            data = r.json()
            break
        else:
            raise RuntimeError(f"gave up after {self.tries} tries: {url}")
        f.write_text(json.dumps(data))
        time.sleep(self.pause)
        return data

    def crossref(self, title, auth):
        p = {"query.bibliographic": f"{title} {auth or ''}".strip(), "rows": 5}
        if self.mail:
            p["mailto"] = self.mail
        d = self.fetch(CR, p)
        out = []
        for it in (d or {}).get("message", {}).get("items", []):
            yr = None
            for k in ("published-print", "published-online", "issued"):
                dp = it.get(k, {}).get("date-parts", [[None]])
                if dp and dp[0] and dp[0][0]:
                    yr = dp[0][0]
                    break
            out.append({"src": "crossref", "title": (it.get("title") or [""])[0],
                        "surnames": [norm(a.get("family", "")) for a in it.get("author", [])],
                        "year": yr, "doi": (it.get("DOI") or "").lower() or None})
        return out

    def openalex(self, title):
        p = {"search": title, "per_page": 5}
        if self.okey:
            p["api_key"] = self.okey
        if self.mail:
            p["mailto"] = self.mail
        d = self.fetch(OA, p)
        out = []
        for it in (d or {}).get("results", []):
            doi = (it.get("doi") or "").lower().replace("https://doi.org/", "") or None
            out.append({"src": "openalex", "title": it.get("display_name") or "",
                        "surnames": [surname(a.get("author", {}).get("display_name", ""))
                                     for a in it.get("authorships", [])],
                        "year": it.get("publication_year"), "doi": doi})
        return out

    def by_doi(self, doi):
        """Resolve a DOI: CrossRef first, then DataCite (pilot finding, 17 Nov 2026)."""
        p = {"mailto": self.mail} if self.mail else {}
        d = self.fetch(f"{CR}/{doi}", p)
        if not d:
            return self.datacite(doi)
        it = d.get("message", {})
        return {"src": "crossref_doi", "title": (it.get("title") or [""])[0],
                "surnames": [norm(a.get("family", "")) for a in it.get("author", [])],
                "year": (it.get("issued", {}).get("date-parts", [[None]])[0] or [None])[0],
                "doi": doi}


def _dc(self, doi):
    d = self.fetch(f"{DC}/{doi}", {})
    if not d:
        return None
    at = (d.get("data") or {}).get("attributes", {})
    return {"src": "datacite_doi", "title": ((at.get("titles") or [{}])[0]).get("title", ""),
            "surnames": [norm(a.get("familyName") or a.get("name", "").split(",")[0])
                         for a in at.get("creators", [])],
            "year": at.get("publicationYear"), "doi": doi}


Client.datacite = _dc


def evidence(c, cands, doi_rec, ytol):
    """Best candidate and checks. Independent of the threshold."""
    ev = {"citation_id": c["citation_id"], "n_cand": len(cands), "best_sim": 0.0,
          "best_src": None, "best_title": None, "surname_ok": None, "year_ok": None,
          "doi_ok": None, "doi_resolves": None, "doi_sim": None}
    if c.get("doi"):
        ev["doi_resolves"] = doi_rec is not None
        if doi_rec:
            ev["doi_sim"] = sim(c["title"], doi_rec["title"])
    if not cands:
        return ev
    sc = [(sim(c["title"], x["title"]), x) for x in cands]
    best = max(v for v, _ in sc)
    b = next(x for v, x in sc if v == best)
    # Several records can hold the same work (conference and journal version, CrossRef and
    # OpenAlex copies). Checks pass if ANY candidate within 0.02 of the best title score passes.
    top = [x for v, x in sc if v >= best - 0.02]
    ev.update(best_sim=best, best_src=b["src"], best_title=b["title"], n_top=len(top))
    fs = norm(c.get("first_surname") or "")
    if fs and any(x["surnames"] for x in top):
        ev["surname_ok"] = any(fs in x["surnames"] for x in top)
    if c.get("year") and any(x["year"] for x in top):
        ev["year_ok"] = any(abs(int(c["year"]) - int(x["year"])) <= ytol for x in top if x["year"])
    if c.get("doi") and any(x["doi"] for x in top):
        ev["doi_ok"] = any(c["doi"].lower() == x["doi"].lower() for x in top if x["doi"])
    return ev


def label_at(c, ev, t):
    """Label from stored evidence at threshold t."""
    if c["parse_status"] == "fail":
        return "unparseable"
    if ev["best_sim"] < t:
        return "fabricated"
    bad = [ev["surname_ok"] is False, ev["year_ok"] is False, ev["doi_ok"] is False,
           ev["doi_resolves"] is False,
           ev["doi_sim"] is not None and ev["doi_sim"] < t]
    return "metadata_error" if any(bad) else "exists"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol_v1.yaml")
    ap.add_argument("--mode", default="main", choices=["main", "pilot", "robust", "quant"])
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    t = cfg["matcher"]["threshold"]
    if a.mode != "pilot" and t is None:
        raise SystemExit("matcher.threshold is null: fix it from the pilot first (analyse --pilot)")
    rps = cfg.get("crossref_max_rps", 3)
    cl = Client(cfg["paths"]["cache"], os.environ.get("CONTACT_EMAIL", ""),
                os.environ.get("OPENALEX_API_KEY", ""), pause=1.0 / rps)
    cits = read_jsonl(Path(cfg["paths"]["parsed"]) / f"citations_{a.mode}.jsonl")
    rows = []
    for i, c in enumerate(cits, 1):
        t0, n0 = time.time(), cl.calls
        if c["parse_status"] == "fail":
            ev = evidence(c, [], None, cfg["matcher"]["year_tolerance"])
        else:
            cands = cl.crossref(c["title"], c.get("authors")) + cl.openalex(c["title"])
            drec = cl.by_doi(c["doi"]) if c.get("doi") else None
            ev = evidence(c, cands, drec, cfg["matcher"]["year_tolerance"])
        ev["label"] = label_at(c, ev, t) if t is not None else None
        ev["seconds"] = round(time.time() - t0, 3)   # wall time incl. cache hits
        ev["api_calls"] = cl.calls - n0              # 0 if fully cached
        rows.append(ev)
        if i % 50 == 0:
            print(f"{i}/{len(cits)}")
    out = Path(cfg["paths"]["labels"]) / f"auto_{a.mode}.jsonl"
    write_jsonl(out, rows)
    print(f"wrote {len(rows)} rows -> {out} (threshold={t})")


if __name__ == "__main__":
    main()

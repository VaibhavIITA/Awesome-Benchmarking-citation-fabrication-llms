"""Split a model response into citation records and extract fields.

Deterministic regex parser: no LLM is used, so parsing cannot add or remove
citations. Every citation gets parse_status ok / partial / fail; nothing is
hand-fixed silently.

Run: python -m src.parse --mode main
Output: data/parsed/citations_<mode>.jsonl
"""
import argparse
import re
from pathlib import Path

from src.common import load_cfg, read_jsonl, write_jsonl

ITEM = re.compile(r"^\s*(?:\[(\d{1,2})\]|(\d{1,2})[.)]|[-*•])\s+")
NUM = re.compile(r"^\s*(?:\*\*)?(?:\[\d{1,2}\]|\d{1,2}[.)])(?:\*\*)?\s+")
# "Title: ...", "**Authors**: ..." style answers (seen in the pilot, 17 Nov 2026)
LAB = re.compile(r"\b(author\(s\)|authors?|year|title|venue|journal|conference|published in|doi|link|url|"
                 r"abstract|summary|keywords|pages|volume)\s*:\s*", re.I)
DOI = re.compile(r"10\.\d{4,9}/[^\s\"<>\]]+", re.I)
YPAR = re.compile(r"\((19[5-9]\d|20[0-3]\d)[a-z]?\)")
YANY = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")
QUOTE = re.compile(r"[\"“]([^\"”]{8,})[\"”]")
URL = re.compile(r"https?://\S+")


def clean(s):
    s = re.sub(r"\*\*|__|`", "", s)
    s = re.sub(r"(?<!\w)\*(?!\s)|(?<!\s)\*(?!\w)", "", s)
    return re.sub(r"\s+", " ", s).strip()


def split_items(text):
    """Numbered lines start a new item. Bullets start one only if the answer has no
    numbered lines (otherwise they are sub-fields of the current item).
    Other non-empty lines continue the current item."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    pat = NUM if any(NUM.match(ln) for ln in lines) else ITEM
    items, cur = [], None
    for ln in lines:
        if pat.match(ln):
            if cur:
                items.append(cur)
            cur = pat.sub("", ln, count=1)
        elif cur is not None:
            cur += " " + ln.strip()
    if cur:
        items.append(cur)
    return [clean(i) for i in items]


def first_surname(auth):
    if not auth:
        return None
    a = re.sub(r"\bet\s*al\.?", "", auth, flags=re.I)
    a = re.split(r";|\band\b|&", a)[0]
    c0 = a.split(",")[0].strip()          # "Walters" | "P. Lewis" | "Smith J"
    ini = re.compile(r"(?:[A-Z]\.?){1,3}(?:-[A-Z]\.?)?")
    toks = [t for t in c0.split() if not ini.fullmatch(t)]
    s = toks[-1] if toks else ""
    s = re.sub(r"[^A-Za-zÀ-ɏ'\-]", "", s)
    return s or None


def fields(txt):
    """Split a labelled item ("Authors: .. Year: .. Title: ..") into a dict, or None."""
    ms = list(LAB.finditer(txt))
    if not any(m.group(1).lower() == "title" for m in ms):
        return None
    f = {}
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(txt)
        k = m.group(1).lower().replace("(s)", "")
        k = "author" if k.startswith("author") else k
        f.setdefault(k, txt[m.end():end].strip(" -.,;"))
    return f


def parse_item(txt):
    """Return dict of fields for one citation string."""
    f = fields(txt)
    if f is not None:
        m = DOI.search(f.get("doi", "")) or DOI.search(txt)
        doi = m.group(0).rstrip(".,;)").lower() if m else None
        ya = YANY.search(f.get("year", ""))
        title = f.get("title") or None
        title = title.strip(" \"“”") if title else None
        if title and len(title) < 8:
            title = None
        auth = f.get("author") or None
        year = int(ya.group(1)) if ya else None
        status = "ok" if title and (year or doi) else "partial" if title else "fail"
        return {"raw": txt, "title": title, "authors": auth, "first_surname": first_surname(auth),
                "year": year, "doi": doi, "parse_status": status}
    doi = None
    m = DOI.search(txt)
    if m:
        doi = m.group(0).rstrip(".,;)").lower()
    body = URL.sub(" ", txt)
    body = re.sub(r"(?i)\bdoi:?\s*10\.\S+", " ", body)
    body = DOI.sub(" ", body)
    body = re.sub(r"\s+", " ", body).strip()

    ym = YPAR.search(body)
    year = int(ym.group(1)) if ym else None
    if year is None:
        ya = YANY.search(body)
        year = int(ya.group(1)) if ya else None

    title, auth = None, None
    q = QUOTE.findall(body)
    if q:
        title = max(q, key=len).strip(" .,")
        auth = body[:body.find(q[0]) - 1].strip(" .,\"“")
    elif ym:
        auth = body[:ym.start()].strip(" .,")
        rest = body[ym.end():].lstrip(" .,")
        title = re.split(r"\.\s", rest, maxsplit=1)[0].strip(" .,")
    else:
        segs = [s.strip(" ,") for s in re.split(r"\.\s", body) if s.strip(" ,")]
        if len(segs) >= 2:
            auth, title = segs[0], segs[1]
        elif segs:
            title = segs[0]
    if title and len(title) < 8:
        title = None
    status = "ok" if title and (year or doi) else "partial" if title else "fail"
    return {"raw": txt, "title": title, "authors": auth or None,
            "first_surname": first_surname(auth), "year": year, "doi": doi,
            "parse_status": status}


def parse_generation(g):
    out = []
    for i, it in enumerate(split_items(g["text"]), 1):
        r = parse_item(it)
        r["citation_id"] = f"{g['gen_id']}|c{i:02d}"
        for k in ("gen_id", "model", "family", "size_b", "topic_id", "topic", "band",
                  "is_pilot", "prompt_type", "seed"):
            r[k] = g[k]
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol_v1.yaml")
    ap.add_argument("--mode", default="main", choices=["main", "pilot", "robust", "quant"])
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    gens = read_jsonl(Path(cfg["paths"]["raw"]) / f"generations_{a.mode}.jsonl")
    rows = [c for g in gens for c in parse_generation(g)]
    out = Path(cfg["paths"]["parsed"]) / f"citations_{a.mode}.jsonl"
    write_jsonl(out, rows)
    st = {}
    for r in rows:
        st[r["parse_status"]] = st.get(r["parse_status"], 0) + 1
    print(f"{len(gens)} responses -> {len(rows)} citations {st} -> {out}")


if __name__ == "__main__":
    main()

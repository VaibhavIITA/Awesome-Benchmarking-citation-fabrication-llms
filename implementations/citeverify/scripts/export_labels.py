"""Write a BLIND labelling sheet (no B2 label, no model name) for hand labelling.

pilot: every pilot citation (used only to fix the B2 threshold)
main:  n citations sampled by model x B2 label (seed fixed), for RQ3 gold set

The sheet has an opaque key instead of citation_id (citation_id holds the model
name). key_map_<mode>.csv links them back; do not open it while labelling.
Labels to type in column gold_label: exists / metadata_error / fabricated / unresolvable
Run: python scripts/export_labels.py --mode pilot
     python scripts/export_labels.py --mode main --n 200
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import load_cfg, read_jsonl  # noqa: E402

COLS = ["key", "raw", "authors", "year", "title", "doi"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol_v1.yaml")
    ap.add_argument("--mode", choices=["pilot", "main"], default="pilot")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--per_topic", type=int, default=None, help="pilot: keep this many per topic (D16a: 10)")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    P = cfg["paths"]
    c = pd.DataFrame(read_jsonl(Path(P["parsed"]) / f"citations_{a.mode}.jsonl"))
    c = c[c["parse_status"] != "fail"]
    if a.mode == "main":
        e = pd.DataFrame(read_jsonl(Path(P["labels"]) / "auto_main.jsonl"))[["citation_id", "label"]]
        c = c.merge(e, on="citation_id")
        n = a.n or cfg["gold"]["size"]
        k = c.groupby(["model", "label"]).ngroups
        if n >= len(c):      # D25 fast run: label every main citation
            pass
        else:
            c = pd.concat([d.sample(min(len(d), max(1, n // k)), random_state=cfg["seeds"]["main"])
                           for _, d in c.groupby(["model", "label"])])
        c = c.sample(frac=1, random_state=cfg["seeds"]["main"]).head(n)
    # citation_id contains the model name, so the sheet shows only an opaque key
    c["key"] = ["k" + hashlib.sha1(x.encode()).hexdigest()[:9] for x in c["citation_id"]]   # "k" stops Excel turning it into a number
    if a.mode == "pilot" and a.per_topic:
        c = pd.concat([d.sample(min(len(d), a.per_topic), random_state=cfg["seeds"]["pilot"])
                       for _, d in c.groupby("topic_id")])
    s = c[COLS].copy()
    s = s.sample(frac=1, random_state=cfg["seeds"]["pilot"])   # shuffle so order hides the model
    s["gold_label"], s["round"], s["note"] = "", 1, ""
    out = Path(P["labels"]) / f"to_label_{a.mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(out, index=False)
    c[["key", "citation_id"]].to_csv(out.with_name(f"key_map_{a.mode}.csv"), index=False)
    print(f"{len(s)} rows -> {out}. Fill gold_label, save as gold_{a.mode}.csv")


if __name__ == "__main__":
    main()

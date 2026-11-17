"""Build the frozen topic list.

Input : data/topics_input.csv with one column `topic` (at least 3 x (n_per_band + n_pilot) rows).
Output: data/topics_frozen.csv with topic_id, topic, works_count, band, is_pilot, is_robust, fetched_utc.

Prominence = OpenAlex works count for a title/abstract search of the topic name.
Bands are terciles of that count (high / medium / low).

Run: python -m src.topics --config configs/protocol_v1.yaml
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import requests

from src.common import load_cfg

OA = "https://api.openalex.org/works"
BANDS = ["high", "medium", "low"]


def works_count(name, key="", mail="", get=requests.get):
    """Return OpenAlex meta.count for a search on `name`."""
    p = {"search": name, "per_page": 1}
    if key:
        p["api_key"] = key
    if mail:
        p["mailto"] = mail
    r = get(OA, params=p, timeout=30)
    r.raise_for_status()
    return int(r.json()["meta"]["count"])


def assign_bands(df, n_band, n_pilot, n_robust, seed):
    """Tercile bands by works_count, then sample n_band + n_pilot topics per band.

    The first n_pilot sampled topics of each band are pilot topics (excluded from
    the main analysis). n_robust non-pilot topics per band are flagged for the
    multi-seed robustness check.
    """
    if df["topic"].duplicated().any():
        raise ValueError("duplicate topic names in input")
    d = df.copy()
    rk = d["works_count"].rank(method="first", ascending=False)
    d["band"] = pd.qcut(rk, 3, labels=BANDS).astype(str)
    rng = np.random.default_rng(seed)
    out = []
    for b in BANDS:
        sub = d[d["band"] == b]
        need = n_band + n_pilot
        if len(sub) < need:
            raise ValueError(f"band {b} has {len(sub)} topics, need {need}")
        idx = rng.permutation(sub.index.to_numpy())[:need]
        pick = sub.loc[idx].copy()
        pick["is_pilot"] = [True] * n_pilot + [False] * n_band
        flags = [False] * need
        for i in range(n_pilot, n_pilot + n_robust):
            flags[i] = True
        pick["is_robust"] = flags
        out.append(pick)
    res = pd.concat(out).sort_values(["band", "works_count"], ascending=[True, False])
    res = res.reset_index(drop=True)
    res.insert(0, "topic_id", [f"T{i + 1:03d}" for i in range(len(res))])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol_v1.yaml")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    tc = cfg["topics"]
    key = os.environ.get("OPENALEX_API_KEY", "")
    mail = os.environ.get("CONTACT_EMAIL", "")
    df = pd.read_csv(tc["input_csv"])
    cnt = []
    for t in df["topic"]:
        cnt.append(works_count(t, key, mail))
        time.sleep(0.2)
    df["works_count"] = cnt
    df["fetched_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res = assign_bands(df, tc["n_per_band"], tc["n_pilot_per_band"],
                       tc["robustness_per_band"], cfg["seeds"]["pilot"])
    res.to_csv(tc["output_csv"], index=False)
    print(res.groupby("band")["works_count"].describe())
    print(f"wrote {len(res)} topics to {tc['output_csv']}")


if __name__ == "__main__":
    main()

"""End-to-end check of src.analyse_a on SYNTHETIC data written by this test.
The numbers here mean nothing about real models."""
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from src import analyse_a as A
from src.common import write_jsonl

MODELS = ["org/ModelA-7B-Instruct", "org/ModelB-7B-Instruct"]


def fake(n_top=10, per=6, seed=0, pilot=False):
    rng = np.random.default_rng(seed)
    c, e = [], []
    for m in MODELS:
        for t in range(n_top):
            for i in range(per):
                cid = f"{m}|T{t}|c{i}"
                fab = rng.random() < 0.4
                c.append({"citation_id": cid, "model": m, "topic_id": f"T{t}", "is_pilot": pilot,
                          "parse_status": "ok", "title": "A long title about things" if not fab else "Short",
                          "authors": "Smith, J." if not fab else None,
                          "year": 2020 if not fab else None, "doi": "10.1145/1" if not fab else None,
                          "first_surname": "smith", "raw": "Smith (2020) Journal 1-2" if not fab else "x"})
                e.append({"citation_id": cid, "best_sim": 0.5 if fab else 0.97, "surname_ok": True,
                          "year_ok": None, "doi_ok": True if not fab else None,
                          "doi_resolves": True if not fab else None, "doi_sim": None,
                          "seconds": 0.4, "api_calls": 2})
    return c, e


def setup(tmp_path, mode, thr):
    c, e = fake(pilot=(mode == "pilot"))
    p = {k: str(tmp_path / k) for k in ("raw", "parsed", "labels", "cache", "results")}
    p["tables"] = str(tmp_path / "paper" / "tables")
    cfg = {"matcher": {"threshold": thr, "year_tolerance": 1}, "b0": {"min_flags": 2},
           "b1": {"C": 1.0, "cv_folds": 5}, "seeds": {"bootstrap": 2026, "cv": 42},
           "bootstrap_resamples": 50, "paths": p}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    write_jsonl(f"{p['parsed']}/citations_{mode}.jsonl", c)
    write_jsonl(f"{p['labels']}/auto_{mode}.jsonl", e)
    g = pd.DataFrame({"citation_id": [x["citation_id"] for x in c[::3]], "round": 1})
    g["gold_label"] = ["fabricated" if x["title"] == "Short" else "exists" for x in c[::3]]
    if mode == "pilot":   # blind sheet form: opaque key plus a separate key map
        g["key"] = [f"k{i:04d}" for i in range(len(g))]
        g[["key", "citation_id"]].to_csv(f"{p['labels']}/key_map_{mode}.csv", index=False)
        g = g.drop(columns="citation_id")
    g.to_csv(f"{p['labels']}/gold_{mode}.csv", index=False)
    return p


def run(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(tmp_path / "cfg.yaml"), *extra])
    A.main()


def test_prf_and_kappa():
    r = A.prf([1, 1, 0, 0], [1, 0, 1, 0])
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1) and r["f1"] == 0.5
    assert A.kappa("aabb", "aabb") == 1.0


def test_check_catches_problems():
    c, e = fake()
    c = pd.DataFrame(c)
    e = pd.DataFrame(e)
    assert A.check(c, e, "main") == []
    bad = pd.concat([c, c.iloc[:1]])
    assert any("duplicate" in x for x in A.check(bad, e, "main"))
    c2 = c.copy()
    c2.loc[0, "is_pilot"] = True
    assert any("pilot" in x for x in A.check(c2, e, "main"))


def test_main_refuses_without_threshold(tmp_path, monkeypatch):
    setup(tmp_path, "main", None)
    with pytest.raises(SystemExit):
        run(monkeypatch, tmp_path)


def test_pilot_then_main(tmp_path, monkeypatch, capsys):
    setup(tmp_path, "pilot", None)
    run(monkeypatch, tmp_path, "--pilot")
    assert "Best B2 threshold" in capsys.readouterr().out
    p = setup(tmp_path, "main", 0.85)
    run(monkeypatch, tmp_path)
    r = pd.read_csv(f"{p['results']}/results.csv")
    assert set(r["analysis"]) >= {"RQ2", "RQ3", "cost", "FR", "quality"}
    f1 = r[(r["analysis"] == "RQ3") & (r["metric"] == "f1")].set_index("cell")["value"]
    assert f1["b2"] == 1.0          # synthetic data is separable by construction
    for t in ("t1_transfer", "t2_verifiers", "t3_cost", "t4_fr"):
        s = open(f"{p['tables']}/{t}.tex").read()
        assert "\\begin{tabular}" in s and "--" not in s


def test_blank_gold_label_is_caught(tmp_path, monkeypatch):
    p = setup(tmp_path, "pilot", None)
    f = f"{p['labels']}/gold_pilot.csv"
    g = pd.read_csv(f)
    g.loc[0, "gold_label"] = None
    g.to_csv(f, index=False)
    with pytest.raises(SystemExit, match="unknown gold labels"):
        run(monkeypatch, tmp_path, "--pilot")

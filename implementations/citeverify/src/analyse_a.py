"""Evaluation for Version A (after change notice D08).

RQ2  transfer of the string-only detector B1 between two open-weight models
RQ3  B0 rules vs B1 detector vs B2 database matching, against human gold labels
Sec. fabrication rate under two definitions (G2)

Every reported number goes to results/results.csv; tables to paper/tables;
figures to results/figures and paper/figures. Stops on any data problem.

Run (fix B2 threshold on pilot): python -m src.analyse_a --pilot
Run (main analysis):             python -m src.analyse_a
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import b0_rules, b1_detector
from src.common import load_cfg, read_jsonl
from src.verify import label_at

GOLD = {"exists", "metadata_error", "fabricated", "unresolvable"}
GRID = [round(x, 2) for x in np.arange(0.70, 0.951, 0.05)]
METRICS = {
    "auc": "higher is better (ranking quality of a score)",
    "auc_drop": "within-model AUC minus transfer AUC; larger = worse transfer",
    "precision": "higher is better", "recall": "higher is better", "f1": "higher is better (primary)",
    "kappa": "higher is better (label reliability)",
    "sec_per_cit": "lower is better (cost)", "api_per_cit": "lower is better (cost)",
    "fr_exist": "outcome, not a score", "fr_broad": "outcome, not a score",
}


# ---------------- loading and checks ----------------
def load(cfg, mode):
    p = cfg["paths"]
    c = pd.DataFrame(read_jsonl(Path(p["parsed"]) / f"citations_{mode}.jsonl"))
    e = pd.DataFrame(read_jsonl(Path(p["labels"]) / f"auto_{mode}.jsonl"))
    return c, e


def check(c, e, mode, gold=None):
    crit = []
    if c.empty or e.empty:
        return ["no citations or no evidence rows"]
    for n, d in (("citations", c), ("evidence", e)):
        if d["citation_id"].duplicated().any():
            crit.append(f"duplicate citation_id in {n}")
    a, b = set(c["citation_id"]), set(e["citation_id"])
    if a != b:
        crit.append(f"id mismatch: {len(a - b)} without evidence, {len(b - a)} without citation")
    if mode == "main" and c["is_pilot"].any():
        crit.append("pilot citations in main corpus")
    for col in ("model", "topic_id", "parse_status"):
        if c[col].isna().any():
            crit.append(f"missing values in {col}")
    if gold is not None:
        bad = set(gold["gold_label"]) - GOLD
        if bad:
            crit.append(f"unknown gold labels {bad}")
        if gold["citation_id"].isna().any() or set(gold["citation_id"]) - a:
            crit.append("gold ids not in corpus")
    return crit


def recs(d):
    """Rows as dicts with NaN turned into None (pandas fills gaps with NaN)."""
    return d.astype(object).where(d.notna(), None).to_dict("records")


def merged(c, e, t):
    m = c.merge(e, on="citation_id", how="inner", validate="one_to_one")
    m["label"] = [label_at(r, r, t) for r in recs(m)]
    return m


# ---------------- metrics ----------------
def prf(g, p):
    g, p = np.asarray(g, bool), np.asarray(p, bool)
    tp, fp, fn = int((g & p).sum()), int((~g & p).sum()), int((g & ~p).sum())
    pr = tp / (tp + fp) if tp + fp else np.nan
    rc = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return {"precision": pr, "recall": rc, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "n": len(g)}


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    if not n:
        return np.nan
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n."""
    if not n:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def cboot(df, stat, seed, n):
    """Bootstrap over topics (clusters). Returns 2.5 and 97.5 percentiles."""
    rng = np.random.default_rng(seed)
    tops = df["topic_id"].unique()
    grp = {k: df[df["topic_id"] == k] for k in tops}
    v = []
    for _ in range(n):
        s = stat(pd.concat([grp[k] for k in rng.choice(tops, len(tops), replace=True)]))
        if not np.isnan(s):
            v.append(s)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)


# ---------------- B1 training ----------------
def oof(rows, y, groups, folds, C, seed, names=None):
    """Out-of-fold B1 probabilities and decisions, folds grouped by topic."""
    from sklearn.model_selection import GroupKFold
    X = b1_detector.matrix(rows, names)
    p, d = np.zeros(len(rows)), np.zeros(len(rows), bool)
    k = min(folds, len(set(groups)))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        m = b1_detector.make(C).fit(X[tr], y[tr])
        thr = b1_detector.best_threshold(y[tr], m.predict_proba(X[tr])[:, 1])
        p[te] = m.predict_proba(X[te])[:, 1]
        d[te] = p[te] >= thr
    return p, d


def transfer(src, dst, C):
    """Train on all of src, score dst. Returns probabilities for dst."""
    X, y = b1_detector.matrix(recs(src)), src["y"].to_numpy()
    m = b1_detector.make(C).fit(X, y)
    return m.predict_proba(b1_detector.matrix(recs(dst)))[:, 1]


# ---------------- output helpers ----------------
def tex(df, path, cap, lab, wide=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    env = "table*" if wide else "table"
    L = [f"\\begin{{{env}}}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3pt}",
         f"\\caption{{{cap}}}", f"\\label{{{lab}}}",
         "\\begin{tabular}{" + "l" * len(df.columns) + "}", "\\toprule",
         " & ".join(str(x).replace("_", "\\_").replace("%", "\\%") for x in df.columns) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells.append("n/a" if np.isnan(v) else f"{v:.3f}")
            else:
                cells.append(str(v).replace("_", "\\_").replace("%", "\\%"))
        L.append(" & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    Path(path).write_text("\n".join(L) + "\n")


def short(m):
    return m.split("/")[-1].replace("-Instruct", "").replace("-v0.3", "")


def read_gold(path):
    """Gold sheet keyed by the blind key; map back to citation_id."""
    g = pd.read_csv(path, dtype={"key": str})
    if "citation_id" not in g.columns:
        k = pd.read_csv(Path(path).with_name(Path(path).name.replace("gold_", "key_map_")), dtype={"key": str})
        g = g.merge(k, on="key", how="left", validate="one_to_one")
    g["gold_label"] = g["gold_label"].astype(str).str.strip().str.lower()
    return g


def pick_threshold(c, e, gold):
    g = gold[(gold["round"] == 1) & (gold["gold_label"] != "unresolvable")].set_index("citation_id")["gold_label"]
    rows = []
    for t in GRID:
        m = merged(c, e, t).set_index("citation_id").loc[g.index]
        rows.append({"t": t, **prf(g.values == "fabricated", m["label"].values == "fabricated")})
    tab = pd.DataFrame(rows)
    return float(tab.sort_values(["f1", "t"], ascending=[False, True]).iloc[0]["t"]), tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol_v1.yaml")
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    P = cfg["paths"]
    R, T = Path(P["results"]), Path(P["tables"])
    (R / "figures").mkdir(parents=True, exist_ok=True)
    pf = T.parent / "figures"
    pf.mkdir(parents=True, exist_ok=True)
    seed, nb = cfg["seeds"]["bootstrap"], cfg["bootstrap_resamples"]

    if a.pilot:
        c, e = load(cfg, "pilot")
        gold = read_gold(Path(P["labels"]) / "gold_pilot.csv")
        crit = check(c, e, "pilot", gold)
        if crit:
            raise SystemExit("CRITICAL: " + "; ".join(crit))
        t, tab = pick_threshold(c, e, gold)
        tab.to_csv(R / "pilot_threshold_sweep.csv", index=False)
        print(tab.to_string(index=False))
        print(f"\nBest B2 threshold on pilot: {t}. Write it to matcher.threshold, log it, then freeze.")
        return

    t = cfg["matcher"]["threshold"]
    if t is None:
        raise SystemExit("matcher.threshold is null: run --pilot first")
    c, e = load(cfg, "main")
    gp = Path(P["labels"]) / "gold_main.csv"
    gold = read_gold(gp) if gp.exists() else None
    crit = check(c, e, "main", gold)
    if crit:
        raise SystemExit("CRITICAL: " + "; ".join(crit))
    df = merged(c, e, t)
    ok = df[df["parse_status"] != "fail"].copy().reset_index(drop=True)
    ok["y"] = (ok["label"] == "fabricated").astype(int)
    out = []

    def add(an, cell, metric, val, lo=np.nan, hi=np.nan, n=np.nan):
        out.append({"analysis": an, "cell": cell, "metric": metric, "value": val,
                    "ci_lo": lo, "ci_hi": hi, "n": n, "threshold": t})

    add("quality", "all", "parse_fail_rate", float((df["parse_status"] == "fail").mean()), n=len(df))
    gens = read_jsonl(Path(P["raw"]) / "generations_main.jsonl")
    lim = cfg.get("decoding", {}).get("max_new_tokens", 10 ** 9)
    for mm in sorted({x["model"] for x in gens}):
        gm = [x for x in gens if x["model"] == mm]
        add("quality", short(mm), "answers_at_token_limit", sum(x["n_new_tokens"] >= lim for x in gm), n=len(gm))
    if "gen_id" in df:
        add("quality", "all", "duplicate_citations_in_answer", int(df.duplicated(["gen_id", "title"]).sum()), n=len(df))

    # ---- RQ2: transfer of B1 between models (scored against B2 labels, large n) ----
    C, k = cfg["b1"]["C"], cfg["b1"]["cv_folds"]
    models = sorted(ok["model"].unique())
    t1, rocs = [], {}
    for m in models:
        dst, src = ok[ok["model"] == m].copy(), ok[ok["model"] != m]
        if src.empty or dst["y"].nunique() < 2:
            continue
        dst["p_in"], _ = oof(recs(dst), dst["y"].to_numpy(),
                             dst["topic_id"].to_numpy(), k, C, cfg["seeds"]["cv"])
        dst["p_tr"] = transfer(src, dst, C)
        a_in, a_tr = auc(dst["y"], dst["p_in"]), auc(dst["y"], dst["p_tr"])
        drop = lambda x: auc(x["y"], x["p_in"]) - auc(x["y"], x["p_tr"])
        lo, hi = cboot(dst, drop, seed, nb)
        add("RQ2", f"{short(m)}|within", "auc", a_in, n=len(dst))
        add("RQ2", f"{short(m)}|transfer", "auc", a_tr, n=len(dst))
        add("RQ2", f"{short(m)}", "auc_drop", a_in - a_tr, lo, hi, len(dst))
        t1.append({"test_model": short(m), "train_model": "+".join(short(x) for x in src["model"].unique()),
                   "auc_within": a_in, "auc_transfer": a_tr, "drop": a_in - a_tr, "ci_lo": lo, "ci_hi": hi,
                   "n": len(dst)})
        rocs[m] = dst
    if t1:
        tex(pd.DataFrame(t1), T / "t1_transfer.tex",
            "Transfer of the string-only detector B1 (AUC against B2 labels; 95\\% topic bootstrap CI of the drop).",
            "tab:transfer", wide=True)

    # ---- RQ3: B0 vs B1 vs B2 against gold labels ----
    t0 = time.time()
    ok["b0"] = [b0_rules.predict(r, cfg["b0"]["min_flags"]) == "fabricated" for r in recs(ok)]
    ok["b0_s"] = [b0_rules.score(r) for r in recs(ok)]
    s_b0 = (time.time() - t0) / max(len(ok), 1)
    t0 = time.time()
    ok["b1_p"], ok["b1"] = oof(recs(ok), ok["y"].to_numpy(), ok["topic_id"].to_numpy(),
                               k, C, cfg["seeds"]["cv"])
    s_b1 = (time.time() - t0) / max(len(ok), 1)
    ok["b2"] = ok["label"] == "fabricated"
    if gold is not None:
        g1 = gold[(gold["round"] == 1) & (gold["gold_label"] != "unresolvable")]
        t2 = []
        # ablation: B1 without its DOI features (Prompt 12 plan)
        nod = [k for k in b1_detector.NAMES if k not in b1_detector.DOI_FEATS]
        _, ok["b1_nodoi"] = oof(recs(ok), ok["y"].to_numpy(), ok["topic_id"].to_numpy(), k, C, cfg["seeds"]["cv"], nod)
        gg = ok.merge(g1[["citation_id", "gold_label"]], on="citation_id", how="inner")
        gg["g"] = gg["gold_label"] == "fabricated"
        gg["all_fab"] = True     # trivial reference: call every citation fabricated
        names = {"b0": "B0 rules", "b1": "B1 string detector", "b2": "B2 database",
                 "b1_nodoi": "B1 without DOI features (ablation)", "all_fab": "Trivial: all fabricated"}
        for v in ("b0", "b1", "b2", "b1_nodoi", "all_fab"):
            r = prf(gg["g"], gg[v])
            plo, phi = wilson(r["tp"], r["tp"] + r["fp"])
            rlo, rhi = wilson(r["tp"], r["tp"] + r["fn"])
            for kk in ("precision", "recall", "f1"):
                add("RQ3", v, kk, r[kk], *(dict(precision=(plo, phi), recall=(rlo, rhi)).get(kk, (np.nan, np.nan))), n=r["n"])
            neg = int((~gg["g"]).sum())
            spec = (neg - r["fp"]) / neg if neg else np.nan
            add("RQ3", v, "specificity", spec, n=neg)
            t2.append({"verifier": names[v], "precision": r["precision"], "P 95% CI": f"{plo:.2f} to {phi:.2f}",
                       "recall": r["recall"], "R 95% CI": f"{rlo:.2f} to {rhi:.2f}", "f1": r["f1"],
                       "specificity": spec, "tp": r["tp"], "fp": r["fp"], "fn": r["fn"]})
        # B2 errors for the failure analysis
        cols = [x for x in ["citation_id", "model", "title", "best_sim", "best_title", "gold_label", "label"] if x in gg]
        gg[(gg["g"] != gg["b2"])][cols].to_csv(R / "b2_disagreements.csv", index=False)
        g_auc = auc(gg["g"].astype(int), 1 - gg["best_sim"])
        add("RQ3", "b2", "auc", g_auc, n=len(gg))
        add("RQ3", "b1", "auc", auc(gg["g"].astype(int), gg["b1_p"]), n=len(gg))
        add("RQ3", "b0", "auc", auc(gg["g"].astype(int), gg["b0_s"]), n=len(gg))
        for x, y_ in (("b2", "b1"), ("b2", "b0"), ("b1", "b0")):
            fd = lambda d: prf(d["g"], d[x])["f1"] - prf(d["g"], d[y_])["f1"]
            lo, hi = cboot(gg, fd, seed, nb)
            add("RQ3", f"{x}-{y_}", "f1_diff", fd(gg), lo, hi, len(gg))
        r2 = gold[gold["round"] == 2]
        if len(r2):
            j = r2.merge(gold[gold["round"] == 1], on="citation_id", suffixes=("_2", "_1"))
            add("RQ3", "relabel", "kappa", kappa(j["gold_label_1"], j["gold_label_2"]), n=len(j))
        tex(pd.DataFrame(t2), T / "t2_verifiers.tex",
            "Verifier comparison against the reference labels (class: fabricated; Wilson 95\\% CIs).", "tab:verifiers", wide=True)
    # cost
    s_b2 = float(e["seconds"].mean()) if "seconds" in e else np.nan
    a_b2 = float(e["api_calls"].mean()) if "api_calls" in e else np.nan
    cost = pd.DataFrame([{"verifier": "B0 rules", "sec_per_cit": s_b0, "api_per_cit": 0.0},
                         {"verifier": "B1 string detector", "sec_per_cit": s_b1, "api_per_cit": 0.0},
                         {"verifier": "B2 database", "sec_per_cit": s_b2, "api_per_cit": a_b2}])
    for _, r in cost.iterrows():
        add("cost", r["verifier"], "sec_per_cit", r["sec_per_cit"])
        add("cost", r["verifier"], "api_per_cit", r["api_per_cit"])
    tex(cost, T / "t3_cost.tex", "Cost per citation (B1 time includes training within cross-validation).", "tab:cost")

    # ---- Secondary: FR under two definitions, per model ----
    t4 = []
    for m in models:
        d = df[(df["model"] == m) & (df["label"] != "unparseable")]
        fe, fb = float((d["label"] == "fabricated").mean()), float(d["label"].isin(["fabricated", "metadata_error"]).mean())
        lo, hi = wilson(int((d["label"] == "fabricated").sum()), len(d))
        add("FR", short(m), "fr_exist", fe, lo, hi, n=len(d))
        lo2, hi2 = wilson(int(d["label"].isin(["fabricated", "metadata_error"]).sum()), len(d))
        add("FR", short(m), "fr_broad", fb, lo2, hi2, n=len(d))
        t4.append({"model": short(m), "fr_exist": fe, "CI": f"{lo:.2f} to {hi:.2f}", "fr_broad": fb,
                   "CI broad": f"{lo2:.2f} to {hi2:.2f}", "n": len(d)})
        if gold is not None:
            gm = d.merge(g1[["citation_id", "gold_label"]], on="citation_id")
            k_ = int((gm["gold_label"] == "fabricated").sum())
            lo3, hi3 = wilson(k_, len(gm))
            add("FR_reference", short(m), "fr_exist", k_ / max(len(gm), 1), lo3, hi3, n=len(gm))
    tex(pd.DataFrame(t4), T / "t4_fr.tex", "Fabrication rate under two definitions (B2 labels).", "tab:fr")

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    if rocs:
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        for m, d in rocs.items():
            for col, ls, nm in (("p_in", "-", "within"), ("p_tr", "--", "transfer")):
                fp, tp, _ = roc_curve(d["y"], d[col])
                ax.plot(fp, tp, ls, label=f"{short(m)} {nm}")
        ax.plot([0, 1], [0, 1], ":", color="grey", lw=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(R / "figures" / "f1_roc_transfer.png", dpi=300)
        fig.savefig(pf / "f1_roc_transfer.pdf")
    if gold is not None:
        sens = []
        for tt in GRID:
            dd = merged(c, e, tt).merge(g1[["citation_id", "gold_label"]], on="citation_id")
            sens.append({"t": tt, **prf(dd["gold_label"] == "fabricated", dd["label"] == "fabricated")})
        sens = pd.DataFrame(sens)
        sens.to_csv(R / "b2_threshold_sensitivity.csv", index=False)
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for kk in ("precision", "recall", "f1"):
            ax.plot(sens["t"], sens[kk], marker=".", label=kk)
        ax.axvline(t, ls="--", lw=0.8, color="grey")
        ax.set_xlabel("B2 title-match threshold t")
        ax.set_ylabel("Score vs reference labels")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(R / "figures" / "f2_b2_sensitivity.png", dpi=300)
        fig.savefig(pf / "f2_b2_sensitivity.pdf")

    pd.DataFrame(out).to_csv(R / "results.csv", index=False)
    print(f"wrote {len(out)} rows to {R / 'results.csv'}")


if __name__ == "__main__":
    main()

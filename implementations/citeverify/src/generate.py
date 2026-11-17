"""Generate reference lists for every model x topic x prompt-type cell (GPU).

Output: data/raw/generations.jsonl, one row per response. Re-running skips
cells already present, so a Colab disconnect does not lose work.

Run (main corpus):   python -m src.generate --config configs/protocol_v1.yaml
Run (pilot topics):  python -m src.generate --pilot
Run (robustness):    python -m src.generate --robust
Run (quant check):   python -m src.generate --quant   (1.5B/3B in 4-bit, robustness topics)
Run (one model):     python -m src.generate --models Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from src.common import append_jsonl, load_cfg, read_jsonl


def build_msgs(tpl, topic):
    """Chat messages for one cell. Only a user turn, so every model's own template applies."""
    return [{"role": "user", "content": tpl.format(topic=topic)}]


def gen_id(mid, tid, ptype, seed, quant=None):
    q = f"|{quant}" if quant else ""
    return f"{mid.split('/')[-1]}|{tid}|{ptype}|s{seed}{q}"


def subset(topics, per_band, seed):
    """Reduced main run (D25): per_band non-pilot topics from each band, fixed seed."""
    tp = topics[~topics["is_pilot"]]
    if not per_band:
        return tp
    parts = [d.sample(min(len(d), per_band), random_state=seed) for _, d in tp.groupby("band")]
    return pd.concat(parts).sort_values("topic_id")


def plan_cells(topics, models, prompts, mode, seeds, per_band=None):
    """List of (model, topic row, prompt type, seed) for the chosen mode."""
    if mode == "quant":
        tp, ss = topics[topics["is_robust"]], [seeds["main"]]
    elif mode == "pilot":
        tp, ss = topics[topics["is_pilot"]], [seeds["main"]]
    elif mode == "robust":
        tp, ss = topics[topics["is_robust"]], seeds["robustness"]
    else:
        tp, ss = subset(topics, per_band, seeds["main"]), [seeds["main"]]
    cells = []
    for m in models:
        for _, t in tp.iterrows():
            for pt in prompts:
                for s in ss:
                    cells.append((m, t, pt, s))
    return cells


def load_model(mid, quant, rev=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    kw = {"device_map": "auto", "revision": rev}
    if quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)
    else:
        kw["dtype"] = torch.float16          # transformers 5.x name (torch_dtype is deprecated)
    mdl = AutoModelForCausalLM.from_pretrained(mid, **kw)
    mdl.eval()
    return tok, mdl


def run_one(tok, mdl, msgs, dec, seed):
    import torch
    from transformers import set_seed
    set_seed(seed)
    # transformers 5.x returns a dict by default (return_dict=True), so ask for it explicitly
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(mdl.device) for k, v in enc.items()}
    ids = enc["input_ids"]
    t0 = time.time()
    with torch.no_grad():
        out = mdl.generate(**enc, do_sample=True, temperature=dec["temperature"],
                           top_p=dec["top_p"], max_new_tokens=dec["max_new_tokens"],
                           pad_token_id=tok.eos_token_id)
    new = out[0, ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True), int(new.shape[0]), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol_v1.yaml")
    ap.add_argument("--models", nargs="*", default=None)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--robust", action="store_true")
    g.add_argument("--quant", action="store_true")
    ap.add_argument("--out", default=None, help="output file (one per GPU process; merge after)")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    mode = "pilot" if a.pilot else "robust" if a.robust else "quant" if a.quant else "main"
    topics = pd.read_csv(cfg["topics"]["output_csv"])
    models = [m for m in cfg["models"] if a.models is None or m["id"] in a.models]
    if mode == "quant":
        models = [dict(m, quant="nf4") for m in models if m["id"] in cfg.get("quant_check", [])]
    qtag = "nf4" if mode == "quant" else None
    out = Path(a.out) if a.out else Path(cfg["paths"]["raw"]) / f"generations_{mode}.jsonl"
    done = {r["gen_id"] for r in read_jsonl(out)}
    cells = plan_cells(topics, models, list(cfg["prompts"]), mode, cfg["seeds"],
                       cfg["topics"].get("main_per_band"))
    print(f"mode={mode} cells={len(cells)} already_done={len(done)}")
    import torch
    for m in models:
        todo = [c for c in cells if c[0]["id"] == m["id"]
                and gen_id(m["id"], c[1]["topic_id"], c[2], c[3], qtag) not in done]
        if not todo:
            continue
        rev = cfg.get("revisions", {}).get(m["id"])
        tok, mdl = load_model(m["id"], m["quant"], rev)
        for _, t, pt, s in todo:
            msgs = build_msgs(cfg["prompts"][pt], t["topic"])
            txt, n, sec = run_one(tok, mdl, msgs, cfg["decoding"], s)
            append_jsonl(out, {
                "gen_id": gen_id(m["id"], t["topic_id"], pt, s, qtag), "model": m["id"],
                "family": m["family"], "size_b": m["size_b"], "quant": m["quant"],
                "revision": rev, "topic_id": t["topic_id"], "topic": t["topic"],
                "band": t["band"], "is_pilot": bool(t["is_pilot"]), "prompt_type": pt,
                "seed": s, "text": txt, "n_new_tokens": n, "seconds": round(sec, 2),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            print(f"{m['id']} {t['topic_id']} {pt} s{s}: {n} tok, {sec:.1f}s")
        del mdl
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

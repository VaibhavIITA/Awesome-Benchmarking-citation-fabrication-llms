"""Environment test for the T21 Version A study (detector reproduction + verifier comparison).
Run:  python scripts/env_test.py
Writes results/env_report.json. Never prints API keys."""
import importlib, json, os, platform, random, sys, time
from pathlib import Path

PKGS = ["torch", "transformers", "accelerate", "bitsandbytes", "sklearn",
        "pandas", "numpy", "rapidfuzz", "requests", "huggingface_hub",
        "datasets", "matplotlib", "yaml", "tqdm", "statsmodels"]
rep = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "python": sys.version.split()[0], "platform": platform.platform(),
       "packages": {}, "gpu": {}, "seed": {}, "network": {}, "checks": {}}

def mark(name, ok, note=""):
    rep["checks"][name] = {"status": "PASS" if ok else "FAIL", "note": note}
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {note}")

# 1. Packages
print("== Packages ==")
for p in PKGS:
    try:
        m = importlib.import_module(p)
        v = getattr(m, "__version__", "unknown")
        rep["packages"][p] = v
        print(f"  {p:16s} {v}")
    except Exception as e:
        rep["packages"][p] = f"MISSING ({type(e).__name__})"
        print(f"  {p:16s} MISSING")
miss = [p for p, v in rep["packages"].items() if str(v).startswith("MISSING")]
mark("packages_importable", not miss, f"missing={miss}" if miss else "")

# 2. GPU
print("== GPU ==")
try:
    import torch
    cu = torch.cuda.is_available()
    rep["gpu"]["cuda_available"] = cu
    rep["gpu"]["torch_cuda_build"] = torch.version.cuda
    if cu:
        pr = torch.cuda.get_device_properties(0)
        rep["gpu"]["name"] = pr.name
        rep["gpu"]["vram_gb"] = round(pr.total_memory / 1e9, 2)
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        t0 = time.time(); b = a @ a; torch.cuda.synchronize()
        rep["gpu"]["matmul_ms"] = round((time.time() - t0) * 1000, 2)
        rep["gpu"]["bf16_supported"] = torch.cuda.is_bf16_supported()
        print(f"  {pr.name}, {rep['gpu']['vram_gb']} GB, matmul {rep['gpu']['matmul_ms']} ms")
    mark("cuda_gpu", cu, "" if cu else "no CUDA device: Runtime > Change runtime type > T4 GPU")
    if cu:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
            mark("bnb_4bit_config", True)
        except Exception as e:
            mark("bnb_4bit_config", False, f"{type(e).__name__}: {e}")
except Exception as e:
    mark("cuda_gpu", False, f"torch unusable: {type(e).__name__}")

# 3. Seed determinism
print("== Seeds ==")
def draw(s):
    random.seed(s); out = [random.random()]
    try:
        import numpy as np
        np.random.seed(s); out.append(float(np.random.rand()))
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(s); out.append(float(torch.rand(1)))
    except Exception:
        pass
    return out
d1, d2 = draw(42), draw(42)
rep["seed"] = {"seed": 42, "draw": d1}
mark("seed_repeatable", d1 == d2, f"{len(d1)} generators checked")

# 4. Network and data access (status code + latency only)
print("== Network ==")
mail = os.environ.get("CONTACT_EMAIL", "")
okey = os.environ.get("OPENALEX_API_KEY", "")
hft = os.environ.get("HF_TOKEN", "")
urls = {
    "crossref": f"https://api.crossref.org/works?rows=1&query=citation&mailto={mail}",
    "openalex": f"https://api.openalex.org/works?per_page=1&search=citation&api_key={okey}",
    "semantic_scholar": "https://api.semanticscholar.org/graph/v1/paper/search?query=citation&limit=1",
    "hf_naser_dataset": "https://huggingface.co/api/datasets/mznaser/llm-citation-hallucination-audit",
    "github_naser_repo": "https://api.github.com/repos/mznaser-clemson/llm-citation-hallucination-audit",
    "hf_qwen7b": "https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct",
    "hf_mistral7b": "https://huggingface.co/api/models/mistralai/Mistral-7B-Instruct-v0.3",
}
try:
    import requests
    for k, u in urls.items():
        try:
            hdr = {"Authorization": f"Bearer {hft}"} if (hft and "huggingface" in u) else {}
            t0 = time.time(); r = requests.get(u, timeout=20, headers=hdr)
            rep["network"][k] = {"status": r.status_code, "ms": round((time.time() - t0) * 1000)}
            if k == "crossref" and r.status_code == 200:
                rep["network"][k]["rate_limit"] = r.headers.get("x-rate-limit-limit")
                rep["network"][k]["rate_interval"] = r.headers.get("x-rate-limit-interval")
        except Exception as e:
            rep["network"][k] = {"status": None, "error": type(e).__name__}
        print(f"  {k:18s} {rep['network'][k]}")
        mark(f"net_{k}", rep["network"][k].get("status") == 200)
except Exception as e:
    mark("requests_available", False, type(e).__name__)
if not mail:
    print("  note: CONTACT_EMAIL not set (CrossRef polite pool)")
if not hft:
    print("  note: HF_TOKEN not set (only needed if a repo is gated)")
if not okey:
    print("  note: OPENALEX_API_KEY not set (key required since Feb 2026)")

# 5. Report
Path("results").mkdir(exist_ok=True)
Path("results/env_report.json").write_text(json.dumps(rep, indent=2))
n_fail = sum(v["status"] == "FAIL" for v in rep["checks"].values())
print(f"== Summary: {len(rep['checks']) - n_fail} PASS, {n_fail} FAIL ==")
print("Report written to results/env_report.json")

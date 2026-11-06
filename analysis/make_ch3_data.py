"""Summarize the CLS-Bench v2 unlearning grid (results/ch3_clsbench/v2_runs.json,
180 runs) into the CSVs that Chapter 3's tables and figures read. Values are written with
six decimals so that tables rounded from them are not rounded twice.
Run from the repository root:  python analysis/make_ch3_data.py"""
import json, os, re, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "ch3_clsbench")
D = os.path.join(ROOT, "output", "ch3"); os.makedirs(D, exist_ok=True)
runs = {}
for n, rec in json.load(open(os.path.join(RES, "v2_runs.json"))).items():
    m = re.match(r"(\w+?)_(\w+?)(_lam[\d.]+)?_s(\d)$", n)
    runs.setdefault((m[1], m[2] + (m[3] or "")), []).append(rec)
mean = lambda v: st.mean(v); sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0
MODELS = ["mistral", "llama", "qwen"]
METHODS = ["ga", "graddiff", "npo", "simnpo", "taskvector", "rmu", "canpo"]
NAME = {"ga": "GA", "graddiff": "GradDiff", "npo": "NPO", "simnpo": "SimNPO",
        "taskvector": "TaskVec", "rmu": "RMU", "canpo": "CA-NPO"}

# main summary: one row per (model, method)
with open(os.path.join(D, "v2_summary.csv"), "w") as out:
    out.write("model,method,n,forget_probe_before,forget_probe_after,d_forget,d_linked,d_linked_sd,"
              "d_distant,d_distant_sd,leak,leak_sd,catastrophic\n")
    for mo in MODELS:
        for me in METHODS:
            R = runs[(mo, me)]
            g = lambda k: [r["deltas_nll"][k] for r in R]
            L = [r["targeted_leakage"] for r in R]
            out.write(f"{mo},{NAME[me]},{len(R)},{R[0]['before']['forget_probe']:.6f},"
                      f"{mean([r['after']['forget_probe'] for r in R]):.6f},{mean(g('forget')):.6f},"
                      f"{mean(g('linked')):.6f},{sd(g('linked')):.6f},{mean(g('distant')):.6f},{sd(g('distant')):.6f},"
                      f"{mean(L):.6f},{sd(L):.6f},{sum(r['catastrophic_collateral'] for r in R)}\n")

# one plotting file per model: linked vs distant damage per method
for mo in MODELS:
    with open(os.path.join(D, f"v2_damage_{mo}.csv"), "w") as out:
        out.write("method,linked,distant\n")
        for me in METHODS:
            R = runs[(mo, me)]
            out.write(f"{NAME[me]},{mean([r['deltas_nll']['linked'] for r in R]):.6f},"
                      f"{mean([r['deltas_nll']['distant'] for r in R]):.6f}\n")

# CA-NPO lambda sweep (lambda=1.0 is the default 'canpo' run)
with open(os.path.join(D, "v2_lambda_sweep.csv"), "w") as out:
    out.write("lam," + ",".join(f"{m}_linked,{m}_forgetprobe" for m in MODELS) + "\n")
    for lam, key in [(0.0, "canpo_lam0.0"), (0.05, "canpo_lam0.05"), (0.1, "canpo_lam0.1"),
                     (0.5, "canpo_lam0.5"), (1.0, "canpo"), (2.0, "canpo_lam2.0")]:
        cells = []
        for mo in MODELS:
            R = runs[(mo, key)]
            cells += [f"{mean([r['deltas_nll']['linked'] for r in R]):.6f}",
                      f"{mean([r['after']['forget_probe'] for r in R]):.6f}"]
        out.write(f"{lam}," + ",".join(cells) + "\n")
print("wrote v2_summary.csv, v2_damage_*.csv, v2_lambda_sweep.csv")

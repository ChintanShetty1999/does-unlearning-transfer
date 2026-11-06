"""Build Chapter 6 tables/figure data from the full transfer run.
runs.json          : evaluation of every unlearned model (and base_<model>)
training_logs.json : training log of every run (final kappa = log[-1]['cos'])
Run from the repository root:  python analysis/make_ch6_data.py"""
import json, os, re, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "ch6_transfer_study")
D = os.path.join(ROOT, "output", "ch6"); os.makedirs(D, exist_ok=True)
RUNS = json.load(open(os.path.join(R, "runs.json")))
LOGS = json.load(open(os.path.join(R, "training_logs.json")))
TAGS = ["zephyr", "llama8b", "qwen7b", "gemma9b"]
NAME = {"zephyr": "Zephyr", "llama8b": "Llama", "qwen7b": "Qwen", "gemma9b": "Gemma"}
base = {t: RUNS[f"base_{t}"] for t in TAGS}

with open(os.path.join(D, "base_models.csv"), "w") as f:
    f.write("model,wmdp,mmlu,compsec,collegecs,ppl_forget,ppl_linked,ppl_general\n")
    for t in TAGS:
        a, p = base[t]["acc"], base[t]["ppl"]
        f.write(f"{NAME[t]},{a['wmdp_cyber']:.2f},{a['mmlu']:.2f},{a['mmlu_computer_security']:.1f},"
                f"{a['mmlu_college_computer_science']:.1f},{p['forget']:.2f},{p['linked']:.2f},{p['general']:.2f}\n")

# one row per unlearning run: config, kappa, accuracies, ratios
rows = []
for k in RUNS:
    if "_sd" not in k or k.startswith("relearn_"):
        continue
    r = RUNS[k]; c = r["cfg"]; b = base[c["tag"]]
    log = LOGS[k]["log"]
    rows.append(dict(model=NAME[c["tag"]], tag=c["tag"], method=c["method"], value=c["value"],
                     layer=c["layer"], seed=c["seed"], coeff=round(c["coeff_effective"], 3),
                     kappa=round(log[-1]["cos"], 4),
                     wmdp=round(r["acc"]["wmdp_cyber"], 2),
                     d_wmdp=round(r["acc"]["wmdp_cyber"] - b["acc"]["wmdp_cyber"], 2),
                     d_mmlu=round(r["acc"]["mmlu"] - b["acc"]["mmlu"], 2),
                     compsec=r["acc"]["mmlu_computer_security"],
                     forget_ratio=round(r["ppl"]["forget"] / b["ppl"]["forget"], 3),
                     linked_ratio=round(r["ppl"]["linked"] / b["ppl"]["linked"], 3),
                     general_ratio=round(r["ppl"]["general"] / b["ppl"]["general"], 4)))
rows.sort(key=lambda d: (TAGS.index(d["tag"]), d["method"], d["value"], d["layer"], d["seed"]))
with open(os.path.join(D, "all_runs.csv"), "w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# per-model sweep files for the beta-sweep figure (NormRMU, relative layer, seed 42)
REL = {"zephyr": 7, "llama8b": 7, "qwen7b": 6, "gemma9b": 9}
for t in TAGS:
    with open(os.path.join(D, f"sweep_{t}.csv"), "w") as f:
        f.write("beta,wmdp,kappa,d_mmlu,feasible\n")
        for d in rows:
            if d["tag"] == t and d["method"] == "normrmu" and d["layer"] == REL[t] and d["seed"] == 42:
                feas = int(d["d_mmlu"] >= -2.0 and d["general_ratio"] <= 1.5)
                f.write(f"{d['value']},{d['wmdp']},{d['kappa']},{d['d_mmlu']},{feas}\n")

# kappa vs forgetting scatter, one file per model (all runs)
for t in TAGS:
    with open(os.path.join(D, f"kappa_{t}.csv"), "w") as f:
        f.write("kappa,d_wmdp\n")
        for d in rows:
            if d["tag"] == t:
                f.write(f"{d['kappa']},{d['d_wmdp']}\n")
print(f"{len(rows)} runs")

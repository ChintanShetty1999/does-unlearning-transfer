"""Build Chapter 4 plot/table CSVs from results/ch4_activation_norm_study.
Run from the repository root:  python analysis/make_ch4_data.py"""
import json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "ch4_activation_norm_study")
D = os.path.join(ROOT, "output", "ch4"); os.makedirs(D, exist_ok=True)
load = lambda name: json.load(open(os.path.join(R, name)))
PROBES, RUNS, PPL, LOGS = (load(n) for n in ("norm_probes.json", "rmu_runs.json", "perplexity_qwen7b.json", "training_logs.json"))
ORDER = ["mistral7b", "zephyr", "llama8b", "olmo7b", "qwen0.5b", "qwen3b",
         "qwen1.5b", "qwen7b", "qwen14b", "gemma9b"]

# 1. per-layer forget-text activation norm against relative depth, one file per model
rows = []
for t in ORDER:
    p = PROBES[f"probe_{t}"]
    n, L = p["forget_norms"], p["n_layers"]
    with open(os.path.join(D, f"norm_profile_{t}.csv"), "w") as f:
        f.write("depth,norm\n")
        for i, v in enumerate(n):
            f.write(f"{i / (len(n) - 1):.4f},{v:.4f}\n")
    rows.append((t, L, p["hidden_size"], n[7]))
with open(os.path.join(D, "norm_L7_all.csv"), "w") as f:
    f.write("model,layers,hidden,norm_L7,beta_of_6p5\n")
    for t, L, h, v in rows:
        f.write(f"{t},{L},{h},{v:.2f},{6.5 / v:.3f}\n")

# 2. final representation cosine (updated vs frozen) from each training log
def final_cos(log):
    recs = re.findall(r"\{'step'.*?\}", LOGS[log])
    last = eval(recs[-1])           # trusted: our own log format
    return last["step"] + 1, last["cos"]
with open(os.path.join(D, "train_cos.csv"), "w") as f:
    f.write("run,steps,final_cos\n")
    for log in sorted(LOGS):
        s, c = final_cos(log)
        f.write(f"{log},{s},{c:.3f}\n")

# 3. accuracy table: base, published c=6.5, norm-scaled c, Qwen budget sweep
def acc(run):
    r = RUNS[run]["results"]
    g = lambda k: 100 * r[k]["acc,none"]
    return g("wmdp_cyber"), g("mmlu_computer_security"), g("mmlu_college_computer_science")
with open(os.path.join(D, "grid_acc.csv"), "w") as f:
    f.write("run,wmdp_cyber,mmlu_compsec,mmlu_collegecs\n")
    for run in sorted(k for k in RUNS if k.startswith(("grid_", "base_"))):
        a = acc(run); f.write(f"{run},{a[0]:.1f},{a[1]:.1f},{a[2]:.1f}\n")

# 4. perplexity contrast (Qwen2.5-7B): base, surgical, collapsed
with open(os.path.join(D, "ppl_contrast.csv"), "w") as f:
    f.write("run,wmdp_cyber,ppl_forget,ppl_linked,ppl_general\n")
    for tag in ["base", "good", "bad"]:
        d = PPL[f"ppl_qwen_{tag}"]
        p = d["perplexity"]
        f.write(f"{tag},{100 * d['results']['wmdp_cyber']['acc,none']:.1f},"
                f"{p['forget']:.2f},{p['retain']:.2f},{p['wikitext']:.2f}\n")
print(open(os.path.join(D, "norm_L7_all.csv")).read())
print(open(os.path.join(D, "train_cos.csv")).read())
print(open(os.path.join(D, "ppl_contrast.csv")).read())

# 5. training traces (every 10 steps): representation cosine and retain loss
TRACES = {"zephyr_c6.5": "rmu_zephyr_cyber", "qwen_75_lr5e-5": "sw_qwen7b_c156",
          "qwen_250_lr5e-5": "bg_qwen7b_c156_lr5e-5_s500", "qwen_75_lr5e-4": "bg_qwen7b_c156_lr5e-4_s75",
          "qwen_75_lr5e-3": "bg_qwen7b_c156_lr5e-3_s75"}
for tag, log in TRACES.items():
    recs = [eval(r) for r in re.findall(r"\{'step'.*?\}", LOGS[log])]
    with open(os.path.join(D, f"trace_{tag}.csv"), "w") as f:
        f.write("step,cos,retain\n")
        for r in recs:
            f.write(f"{r['step']},{r['cos']:.4f},{max(r['retain'], 1e-3):.4f}\n")
print("wrote traces")

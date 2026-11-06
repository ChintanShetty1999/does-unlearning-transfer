"""Turn the transfer-study results in results/ch6_transfer_study/ into small CSVs
that the figures read.
Run from the repository root:  python analysis/make_plot_data.py"""
import json, os, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "ch6_transfer_study")
D = os.path.join(ROOT, "output"); os.makedirs(os.path.join(D, "ch6"), exist_ok=True)
NAME = {"zephyr": "Zephyr", "llama8b": "Llama", "qwen7b": "Qwen", "gemma9b": "Gemma"}
ORDER = list(NAME)

prof = json.load(open(os.path.join(R, "norm_profile.json")))["profile"]   # per-layer list; index 7 = original RMU layer
pd.DataFrame({"model": [NAME[t] for t in ORDER],
              "norm": [round(prof[t][7], 2) for t in ORDER]}).to_csv(os.path.join(D, "norms_L7.csv"), index=False)

s = pd.read_csv(os.path.join(R, "E4_transfer_summary.csv"))
rl = pd.read_csv(os.path.join(R, "E6_relearning.csv"))
base = rl.groupby("tag")["wmdp_base"].first()
pick = lambda t, m: s[(s.tag == t) & (s.Method == m)]["wmdp_mean"].item()
acc = pd.DataFrame({"model": [NAME[t] for t in ORDER],
                    "base":   [round(base[t], 2) for t in ORDER],
                    "rmu":    [round(pick(t, "RMU (c=6.5, L7)"), 2) for t in ORDER],
                    "oracle": [round(pick(t, "Oracle (per-model tuned)"), 2) for t in ORDER]})
acc.to_csv(os.path.join(D, "wmdp_transfer.csv"), index=False)

g = pd.read_csv(os.path.join(R, "transfer_gap.csv"))
gp = g.pivot(index="tag", columns="Method", values="transfer_gap_wmdp")
tg = ORDER[1:]
pd.DataFrame({"model": [NAME[t] for t in tg],
              "rmu":     [round(gp.loc[t, "RMU (c=6.5, L7)"], 2) for t in tg],
              "armu":    [round(gp.loc[t, "Adaptive RMU"], 2) for t in tg],
              "normrmu": [round(gp.loc[t, "NormRMU (ours)"], 2) for t in tg]}).to_csv(os.path.join(D, "transfer_gap.csv"), index=False)

rl["recovered_pct"] = 100 * (rl.wmdp_relearned - rl.wmdp_unlearned) / (rl.wmdp_base - rl.wmdp_unlearned)
rl["forgot_pts"] = rl.wmdp_base - rl.wmdp_unlearned
# recovery is only defined when the run actually forgot something (>= 5 points)
rl.loc[rl.forgot_pts < 5, "recovered_pct"] = float("nan")
rl.round(2).to_csv(os.path.join(D, "relearning.csv"), index=False)
# Figure 6.4 data: four decimals, so that the one-decimal bar labels are rounded
# once, from the raw value (two-decimal input turned 53.548 into 53.6).
LBL = {("zephyr", "RMU (c=6.5, L7)"): "Zephyr RMU", ("zephyr", "Adaptive RMU"): "Zephyr A-RMU",
       ("zephyr", "NormRMU (ours)"): "Zephyr NormRMU", ("qwen7b", "Adaptive RMU"): "Qwen A-RMU",
       ("qwen7b", "NormRMU (ours)"): "Qwen NormRMU"}
fig = rl[rl.forgot_pts >= 5].copy()
fig.insert(0, "run", [LBL[(t, m)] for t, m in zip(fig.tag, fig.Method)])
fig = fig.rename(columns={"wmdp_base": "base", "wmdp_unlearned": "unlearned", "wmdp_relearned": "relearned"})
fig[["run", "base", "unlearned", "relearned"]].round(4).to_csv(os.path.join(D, "ch6", "relearn_plot.csv"), index=False)
print(acc.to_string(index=False)); print(rl[["tag", "Method", "forgot_pts", "recovered_pct"]].round(1).to_string(index=False))

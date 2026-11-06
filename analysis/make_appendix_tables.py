"""Generate the LaTeX tables of Appendix A directly from the result files,
so that no number is copied by hand. Run from the repository root, after
make_ch3_data.py:
    python analysis/make_appendix_tables.py
Writes output/appendix_tables/*.tex"""
import csv, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R6 = os.path.join(ROOT, "results", "ch6_transfer_study")
CH3 = os.path.join(ROOT, "output", "ch3")   # written by make_ch3_data.py
OUT = os.path.join(ROOT, "output", "appendix_tables"); os.makedirs(OUT, exist_ok=True)
NAME = {"zephyr": "Zephyr-7B", "llama8b": "Llama-3.1-8B", "qwen7b": "Qwen2.5-7B", "gemma9b": "Gemma-2-9B"}
ORDER = ["zephyr", "llama8b", "qwen7b", "gemma9b"]
MNAME = {"RMU (c=6.5, L7)": "RMU", "Adaptive RMU": "Adaptive RMU",
         "NormRMU (ours)": "NormRMU", "Oracle (per-model tuned)": "Oracle"}
def fx(x, nd):
    """Round half up from the decimal value, so that values read from a CSV are
    not rounded twice (Python's float formatting can round 0.585 down)."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))
def ratio(x):
    x = float(x)
    return f"{x:.2f}" if x < 10 else (f"{x:.1f}" if x < 1000 else f"{x:,.0f}".replace(",", "{,}"))

# A.1 per-seed transfer results
rows = list(csv.DictReader(open(os.path.join(R6, "E4_transfer_all_runs.csv"))))
rows.sort(key=lambda r: (ORDER.index(r["tag"]), list(MNAME).index(r["Method"]), int(r["seed"])))
with open(os.path.join(OUT, "per_seed.tex"), "w") as f:
    f.write("\\begin{longtable}{@{}llrrrrrrr@{}}\n"
            "\\titledcaption{Every transfer run.}{$c$: coefficient. "
            "For Adaptive RMU the multiplier $\\beta_A$ is shown instead. WMDP-Cyber, MMLU, and Comp.\\ sec.\\ in "
            "percent. Forget, linked: perplexity ratio to the base model.}\\label{tab:perseed}\\\\\n\\toprule\n"
            "Model & Method & Seed & $c$ & WMDP-Cyber & MMLU & Comp.\\ sec. & Forget & Linked \\\\\n\\midrule\n\\endfirsthead\n"
            "\\multicolumn{9}{@{}l}{\\small\\emph{\\cref{tab:perseed} continued}}\\\\\n\\toprule\n"
            "Model & Method & Seed & $c$ & WMDP-Cyber & MMLU & Comp.\\ sec. & Forget & Linked \\\\\n\\midrule\n\\endhead\n"
            "\\bottomrule\n\\endfoot\n")
    prev = None
    for r in rows:
        if prev and prev != r["tag"]:
            f.write("\\midrule\n")
        prev = r["tag"]
        cval = (f"$\\beta_A{{=}}{float(r['value']):g}$" if r["method"] == "armu"
                else f"{float(r['coeff_effective']):.2f}")
        f.write(f"{NAME[r['tag']]} & {MNAME[r['Method']]} & {r['seed']} & {cval} & "
                f"{float(r['wmdp']):.1f} & {float(r['mmlu']):.1f} & {float(r['mmlu_sec']):.0f} & "
                f"{ratio(r['forget_ppl_ratio'])} & {ratio(r['linked_ppl_ratio'])} \\\\\n")
    f.write("\\end{longtable}\n")

# A.2 full NormRMU sweep with perplexity ratios
sw = list(csv.DictReader(open(os.path.join(R6, "E2_beta_sweep.csv"))))
with open(os.path.join(OUT, "sweep_full.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\titledcaption{Complete NormRMU coefficient sweep (seed 42, relative layer).}{"
            "Accuracy in percent; perplexity as a ratio to the base model.}\\label{tab:sweepfull}\n"
            "\\centering\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{@{}llrrrrrrr@{}}\n\\toprule\n"
            "Model & $\\beta$ & $c$ & WMDP-Cyber & $\\Delta$MMLU & Comp.\\ sec. & Forget & Linked & Distant \\\\\n\\midrule\n")
    prev = None
    for r in sorted(sw, key=lambda r: (ORDER.index(r["tag"]), float(r["value"]))):
        if prev and prev != r["tag"]:
            f.write("\\midrule\n")
        f.write(f"{NAME[r['tag']] if prev != r['tag'] else ''} & {float(r['value']):g} & {float(r['coeff_effective']):.1f} & "
                f"{float(r['wmdp']):.1f} & {float(r['d_mmlu']):+.1f} & {float(r['mmlu_sec']):.0f} & "
                f"{ratio(r['forget_ppl_ratio'])} & {ratio(r['linked_ppl_ratio'])} & {float(r['general_ppl_ratio']):.3f} \\\\\n")
        prev = r["tag"]
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# A.3 Adaptive RMU source sweep
ar = list(csv.DictReader(open(os.path.join(R6, "E3_armu_source_sweep.csv"))))
with open(os.path.join(OUT, "armu_sweep.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\caption{Adaptive RMU coefficient sweep on the source model, Zephyr-7B "
            "(seed 42, layer 7).}\\label{tab:armusweep}\n\\centering\\small\n"
            "\\begin{tabular}{@{}rrrrr@{}}\n\\toprule\n"
            "$\\beta_A$ & WMDP-Cyber & MMLU & $\\Delta$MMLU & Distant perplexity ratio \\\\\n\\midrule\n")
    for r in ar:
        f.write(f"{float(r['value']):g} & {float(r['wmdp']):.1f} & {float(r['mmlu']):.1f} & "
                f"{float(r['d_mmlu']):+.1f} & {float(r['general_ppl_ratio']):.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# A.4 activation-norm profile, every layer (transfer-study measurement)
prof = json.load(open(os.path.join(R6, "norm_profile.json")))
P, L = prof["profile"], prof["num_layers"]
n = max(L.values())
with open(os.path.join(OUT, "norm_profile.tex"), "w") as f:
    f.write("\\begin{longtable}{@{}rrrrr@{}}\n"
            "\\titledcaption{Mean activation norm after every decoder layer.}{Measured on the forget corpus, "
            "for the four models of the transfer study.}\\label{tab:normfull}\\\\\n\\toprule\n"
            "Layer & Zephyr-7B & Llama-3.1-8B & Qwen2.5-7B & Gemma-2-9B \\\\\n\\midrule\n\\endfirsthead\n"
            "\\multicolumn{5}{@{}l}{\\small\\emph{\\cref{tab:normfull} continued}}\\\\\n\\toprule\n"
            "Layer & Zephyr-7B & Llama-3.1-8B & Qwen2.5-7B & Gemma-2-9B \\\\\n\\midrule\n\\endhead\n"
            "\\bottomrule\n\\endfoot\n")
    for i in range(n):
        cells = [f"{P[t][i]:.2f}" if i < L[t] else "" for t in ORDER]
        f.write(f"{i} & " + " & ".join(cells) + " \\\\\n")
    f.write("\\end{longtable}\n")

# A.5 CLS-Bench v2 full summary with distant damage
v2 = list(csv.DictReader(open(os.path.join(CH3, "v2_summary.csv"))))
MN = {"mistral": "Mistral-7B", "llama": "Llama-3.1-8B", "qwen": "Qwen2.5-7B"}
V2METHOD = {"TaskVec": "Task vector"}
with open(os.path.join(OUT, "v2_full.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\titledcaption{Complete CLS-Bench v2 results, mean $\\pm$ standard deviation over five "
            "seeds.}{Probe: forget-probe accuracy after unlearning. $\\Delta$: increase in NLL. Cat.: number of "
            "seeds flagged as catastrophic collateral damage.}\\label{tab:v2full}\n\\centering\\footnotesize\n"
            "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n"
            " & & & & & & Targeted & \\\\\n"
            "Model & Method & Probe & $\\Delta_{\\text{forget}}$ & $\\Delta_{\\text{linked}}$ & "
            "$\\Delta_{\\text{distant}}$ & leakage & Cat. \\\\\n\\midrule\n")
    prev = None
    for r in v2:
        if prev and prev != r["model"]:
            f.write("\\midrule\n")
        f.write(f"{MN[r['model']] if prev != r['model'] else ''} & {V2METHOD.get(r['method'], r['method'])} & {fx(r['forget_probe_after'], 2)} & "
                f"{fx(r['d_forget'], 2)} & ${fx(r['d_linked'], 2)}\\pm{fx(r['d_linked_sd'], 2)}$ & "
                f"${fx(r['d_distant'], 2)}\\pm{fx(r['d_distant_sd'], 2)}$ & "
                f"${fx(r['leak'], 2)}\\pm{fx(r['leak_sd'], 2)}$ & {r['catastrophic']} \\\\\n")
        prev = r["model"]
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# A.6 CA-NPO lambda sweep
lam = list(csv.DictReader(open(os.path.join(CH3, "v2_lambda_sweep.csv"))))
with open(os.path.join(OUT, "v2_lambda.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\titledcaption{CA-NPO anchor-weight sweep on CLS-Bench v2.}{Mean over five seeds: "
            "damage to the linked principle ($\\Delta$NLL) and forget-probe accuracy after unlearning.}"
            "\\label{tab:v2lambda}\n\\centering\\small\n\\begin{tabular}{@{}rcccccc@{}}\n\\toprule\n"
            " & \\multicolumn{2}{c}{Mistral-7B} & \\multicolumn{2}{c}{Llama-3.1-8B} & \\multicolumn{2}{c}{Qwen2.5-7B} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "$\\lambda$ & $\\Delta_{\\text{linked}}$ & Probe & $\\Delta_{\\text{linked}}$ & Probe & $\\Delta_{\\text{linked}}$ & Probe \\\\\n\\midrule\n")
    for r in lam:
        f.write(f"{float(r['lam']):g} & {fx(r['mistral_linked'], 3)} & {fx(r['mistral_forgetprobe'], 2)} & "
                f"{fx(r['llama_linked'], 3)} & {fx(r['llama_forgetprobe'], 2)} & "
                f"{fx(r['qwen_linked'], 3)} & {fx(r['qwen_forgetprobe'], 2)} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
# A.7 held-out validation of the kappa rule (Section 6.7), from validate_kappa.py
KV = os.path.join(ROOT, "results", "ch6_heldout_validation")
KVNAME = {"mistral7b": "Mistral-7B", "qwen3b": "Qwen2.5-3B", "qwen14b": "Qwen2.5-14B"}
with open(os.path.join(OUT, "holdout_full.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\titledcaption{Complete NormRMU sweep on the held-out models (seed 42, "
            "relative layer).}{Accuracy in percent; perplexity as a ratio to the base model; "
            "$\\kappa$: alignment.}\\label{tab:holdoutfull}\n"
            "\\centering\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{@{}llrrrrrrrr@{}}\n\\toprule\n"
            "Model & $\\beta$ & $c$ & WMDP-Cyber & $\\Delta$MMLU & Comp.\\ sec. & Forget & Linked & Distant & $\\kappa$ \\\\\n\\midrule\n")
    for i, tag in enumerate(KVNAME):
        if i:
            f.write("\\midrule\n")
        kv = sorted(csv.DictReader(open(os.path.join(KV, f"sweep_with_kappa_{tag}.csv"))), key=lambda r: float(r["value"]))
        for j, r in enumerate(kv):
            f.write(f"{KVNAME[tag] if j == 0 else ''} & {float(r['value']):g} & {float(r['coeff_effective']):.1f} & "
                    f"{float(r['wmdp']):.1f} & {float(r['d_mmlu']):+.1f} & {float(r['mmlu_sec']):.0f} & "
                    f"{ratio(r['forget_ppl_ratio'])} & {ratio(r['linked_ppl_ratio'])} & "
                    f"{float(r['general_ppl_ratio']):.3f} & {float(r['kappa']):.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
print("wrote", sorted(os.listdir(OUT)))

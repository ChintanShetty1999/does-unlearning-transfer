"""Build the figure data for the held-out validation of the kappa rule
(Section 6.7) from the files written by validate_kappa.py:
    results/ch6_heldout_validation/sweep_with_kappa_<model>.csv
Writes output/ch6/kappa_holdout_<model>.csv (kappa, change in WMDP-Cyber).
Run from the repository root:  python analysis/make_holdout_data.py"""
import csv, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "ch6_heldout_validation")
D = os.path.join(ROOT, "output", "ch6"); os.makedirs(D, exist_ok=True)
for tag in ["mistral7b", "qwen3b", "qwen14b"]:
    rows = list(csv.DictReader(open(os.path.join(RAW, f"sweep_with_kappa_{tag}.csv"))))
    with open(os.path.join(D, f"kappa_holdout_{tag}.csv"), "w") as f:
        f.write("beta,kappa,d_wmdp\n")
        for r in rows:
            f.write(f"{float(r['value']):g},{float(r['kappa']):.4f},{float(r['d_wmdp']):.2f}\n")
    print(tag, len(rows), "runs")

"""Merge many small result files into one JSON file, keyed by run name.

The bundles in results/ were made with this script from the per-run files the
experiments write. JSON files are stored as parsed objects and text logs as
strings, so nothing is lost. Examples (run from the repository root):

  python analysis/bundle_results.py results/ch6_transfer_study/runs.json \\
      $RMU_HOME/RMU-Thesis/results_transfer/runs/*.json
  python analysis/bundle_results.py --key parent results/ch6_transfer_study/training_logs.json \\
      $RMU_HOME/RMU-Thesis/scratch/transfer_runs/*/meta.json
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument("out", help="bundle to write")
ap.add_argument("files", nargs="+", help="result files to merge")
ap.add_argument("--key", choices=["name", "parent"], default="name",
                help="key each entry by its file name (default) or by its folder name")
a = ap.parse_args()

bundle = {}
for f in sorted(a.files):
    key = os.path.basename(os.path.dirname(f)) if a.key == "parent" else os.path.splitext(os.path.basename(f))[0]
    if key in bundle:
        raise SystemExit(f"duplicate key {key}")
    with open(f) as fh:
        bundle[key] = json.load(fh) if f.endswith(".json") else fh.read()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "w") as fh:
    json.dump(bundle, fh, indent=1)
    fh.write("\n")
print(f"wrote {a.out} ({len(bundle)} entries)")

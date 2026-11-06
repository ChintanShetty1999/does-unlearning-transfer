# Does Unlearning Transfer?

Code and results for the M.S. thesis **"Does Unlearning Transfer? Model-Relative Calibration of
Representation Misdirection Across LLM Families"** by Chintan Shetty, The University of Texas at
San Antonio, 2026. Advisor: Dr. Sushil K. Prasad.

## Summary

RMU (Representation Misdirection for Unlearning) removes hazardous knowledge from a language model
by pushing the model's activations at one layer toward a fixed random vector of length *c*. Its
published configuration (*c* = 6.5 at layer 7) was tuned on Zephyr-7B. Copied unchanged to
Llama-3.1-8B, Qwen2.5-7B, and Gemma-2-9B, it moves hazardous accuracy by at most 1.5 points. This
thesis contributes:

- **NormRMU.** It sets the coefficient relative to each model's mean activation norm,
  *c* = β·*N*<sub>ℓ</sub>, and sets the intervention layer relative to depth, ℓ = round(ρ·*L*) with
  ρ = 7/32. It also selects the updated weights by name.
- **A source-only transfer protocol.** All tuning happens on one source model. β and ρ transfer
  unchanged, and each target is graded against a per-model oracle.
- **The alignment κ.** κ is the mean cosine between the forget tokens' activations and RMU's random
  vector *u* at the last training step. The κ rule uses it to choose β without test answers.

Main results:

- On Qwen2.5-7B, the transfer gap falls from 19.2 to 1.0 points.
- On three held-out models, the κ rule cuts the mean gap from 8.2 to 2.0 points.
- A 50-step relearning attack recovers 90–95% of the removed accuracy.

## What is in this repository

```
code/       the experiment scripts (need a GPU)
results/    the raw results of every experiment, one folder per chapter
analysis/   the scripts that turn the results into the thesis tables and figure data (no GPU)
figures/    the figure sources and the script that renders them
```

Files that the scripts can regenerate are not stored here. This includes processed CSVs, LaTeX
tables and rendered figures, which are written to `output/`. Many experiments write one small file
per run. To keep the repository small, those files are merged into one JSON file per study, keyed
by the original file name. `analysis/bundle_results.py` does the merging.

| Folder | File | Contents |
| --- | --- | --- |
| `code/` | `transfer_study.py` | Chapters 5–6: norm profiles, β sweeps, source-only transfer, depth ablation, relearning attack |
| | `validate_kappa.py` | Section 6.7: held-out validation of the κ rule |
| | `activation_norm_study/` | Chapter 4: `norm_probe.py`, `run_rmu.py`, `eval_rmu.py`, `rmu_core.py` |
| `results/ch3_clsbench/` | `pilot_*.json` | Pilot study |
| | `v2_runs.json` | All 180 runs on CLS-Bench v2 |
| `results/ch4_activation_norm_study/` | `norm_probes.json` | Mean activation norm after every layer of ten models |
| | `rmu_runs.json` | Evaluations of the base models and of every RMU run, including the Zephyr-7B reproduction |
| | `perplexity_qwen7b.json` | Surgical vs. collapsed Qwen2.5-7B run |
| | `training_logs.json` | Training logs (cosine to the frozen copy and retain loss every 10 steps) |
| | `harness_check.json` | Evaluation harness check on the authors' released checkpoint and its base model |
| `results/ch6_transfer_study/` | `E2_*.csv` to `E6_*.csv` | β sweep, Adaptive RMU source sweep, transfer runs, depth ablation, relearning |
| | `runs.json` | Metrics of every transfer run and base model |
| | `training_logs.json` | Training log of every transfer run, including κ at each step |
| | `norm_profile.json`, `selected_hparams.json`, `transfer_gap.csv` | Activation norms, β\* chosen on the source model, transfer gaps |
| `results/ch6_heldout_validation/` | `sweep_with_kappa_*.csv`, `kappa_rule_validation.csv` | Held-out models |

## Setup

```bash
git clone https://github.com/ChintanShetty1999/does-unlearning-transfer.git
cd does-unlearning-transfer
conda create -n rmu python=3.11 -y
conda activate rmu
pip install -r requirements.txt
```

`requirements.txt` lists the package versions used in the thesis (Appendix C). The analysis scripts
need only Python with pandas, and the figures need `pdflatex` with TikZ and pgfplots.

## 1. Regenerate the thesis tables and figures (no GPU)

Run these from the repository root:

```bash
python analysis/make_ch3_data.py         # Chapter 3
python analysis/make_ch4_data.py         # Chapter 4
python analysis/make_ch6_data.py         # Chapter 6
python analysis/make_holdout_data.py     # Section 6.7
python analysis/make_plot_data.py        # Chapter 1 and 6 figure data
python analysis/make_appendix_tables.py  # Appendix A tables
bash figures/build_figures.sh            # all figures, as PDF
```

Everything is written to `output/`. The CSVs and Appendix A tables match the ones used in the
thesis exactly.

## 2. Re-run the experiments (GPU)

All experiments ran on a DGX A100 with eight A100 80 GB GPUs, using one GPU per run. The header
of `transfer_study.py` notes that a single 48 GB GPU is enough.

**Model access.**
- `transfer_study.py` downloads ungated mirrors of Llama-3.1-8B-Instruct and Gemma-2-9B-it
  (`unsloth/...`). To use the official repositories instead, set `USE_OFFICIAL_REPOS = True` and
  log in with `huggingface-cli login`.
- The Chapter 4 scripts load the official repositories, so they need the login too.

**Data.** Everything is downloaded automatically from the Hugging Face Hub:
- the forget corpus and the linked corpus: `cais/wmdp-corpora` (`cyber-forget-corpus`,
  `cyber-retain-corpus`);
- the retain set and distant perplexity: WikiText-2 (`wikitext-2-raw-v1`);
- the benchmarks: `wmdp_cyber` and `mmlu`, through the LM Evaluation Harness.

No WMDP text is stored in this repository.

### Chapter 4: activation-norm study and RMU runs

```bash
export HF_HOME=/path/to/hf_cache
S=code/activation_norm_study
python $S/norm_probe.py --model Qwen/Qwen2.5-7B-Instruct --out out/probe_qwen7b.json
python $S/run_rmu.py --model Qwen/Qwen2.5-7B-Instruct --out_dir runs/qwen7b_c6.5
python $S/eval_rmu.py --model Qwen/Qwen2.5-7B-Instruct \
    --deltas runs/qwen7b_c6.5/deltas.safetensors --out out/grid_qwen7b_c6.5.json
```

- `run_rmu.py` defaults to the published configuration: *c* = 6.5, intervention layer 7,
  `mlp.down_proj` of layers 5–7 updated, learning rate 5e-5, 75 steps.
- The run names in `rmu_runs.json` give each run's settings: `grid_<model>_c<c>[_lr<lr>_s<steps>]`.
  For example, the one surgical run is `--steering_coeff 156 --lr 5e-5 --max_num_batches 500`.
  It stops after 250 steps, when the forget corpus runs out.

### Chapters 5–6: transfer study

```bash
export RMU_HOME=/path/to/workdir
RMU_QUICK=0 python code/transfer_study.py > full_run.log 2>&1
```

- **Set `RMU_QUICK=0`.** The default is a one-seed pilot that is not reported in the thesis.
- The script caches every result, so rerunning it resumes where it stopped.
- The full run took about five hours on one A100.

To compare a new run with the stored results, merge its per-run files the same way:

```bash
python analysis/bundle_results.py new_runs.json $RMU_HOME/RMU-Thesis/results_transfer/runs/*.json
python analysis/bundle_results.py --key parent new_logs.json \
    $RMU_HOME/RMU-Thesis/scratch/transfer_runs/*/meta.json
```

### Section 6.7: held-out validation of the κ rule

```bash
export RMU_HOME=/path/to/workdir
KV_HF_CACHE=/path/to/hf_cache python code/validate_kappa.py > kappa_val.log 2>&1
```

- Results are written to `$RMU_HOME/RMU-Thesis/results_kappa_validation/`.
- The script also lists OLMo-2-7B. That model could not be loaded and was dropped before any run.

## Where each result comes from

Paths are relative to `results/`.

| Thesis result | Source |
| --- | --- |
| Tables 3.1, 3.2 (pilot study) | `ch3_clsbench/pilot_*.json` |
| Table 3.3, Figures 3.2, 3.3 | `ch3_clsbench/v2_runs.json` |
| Table 4.1 | `ch4_activation_norm_study/harness_check.json` |
| Section 4.2 (reproduction on Zephyr-7B) | `ch4_activation_norm_study/rmu_runs.json` (`zephyr_*` entries) |
| Table 4.3, Figure 4.1 | `ch4_activation_norm_study/norm_probes.json` |
| Tables 4.4, 4.5 | `ch4_activation_norm_study/rmu_runs.json` |
| Table 4.6, Figure 4.3 | `ch4_activation_norm_study/perplexity_qwen7b.json` |
| Figure 4.2 | `ch4_activation_norm_study/training_logs.json` |
| Tables 5.2, 6.2, 6.3 | `ch6_transfer_study/E4_transfer_*.csv`, `norm_profile.json` |
| Tables 6.4, 6.6, Figures 6.1, 6.2 | `ch6_transfer_study/E2_beta_sweep.csv`, `runs.json`, `training_logs.json` |
| Table 6.5 | `ch6_transfer_study/E5_depth_ablation.csv` |
| Table 6.9, Figure 6.4 | `ch6_transfer_study/E6_relearning.csv` |
| Tables 6.7, 6.8, A.7, Figure 6.3 | `ch6_heldout_validation/*.csv` |

## Notes

- **Checkpoints.** Each run saves only the change to the updated weights (`deltas.safetensors`,
  several hundred MB per run). The checkpoints are not stored here, but every run can be
  regenerated with the seeds used in the thesis (42, 43, 44).
- **Small differences.** GPU kernels in bfloat16 are not bit-for-bit deterministic, so a re-run can
  differ slightly from the stored numbers.
- **MMLU subset.** The transfer study evaluates MMLU on 40 questions per subject.
- **Names in the code.** The code keeps the names used while the experiments ran:
  - "general" perplexity is the distant perplexity of the thesis;
  - "retain" perplexity in `eval_rmu.py` is the linked perplexity.
- **Script name.** `transfer_study.py` was called `noval_method.py` when the experiments ran. Its
  contents are unchanged.

## Citation

```bibtex
@mastersthesis{shetty2026unlearning,
  author = {Chintan Shetty},
  title  = {Does Unlearning Transfer? Model-Relative Calibration of Representation
            Misdirection Across {LLM} Families},
  school = {The University of Texas at San Antonio},
  year   = {2026}
}
```

## License

The code is released under the MIT License. Models and datasets keep their own licenses.

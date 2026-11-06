# Generated from: noval_method.ipynb
# Converted at: 2026-10-06T05:27:21.082Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ============================================================
# DO RMU UNLEARNING HYPERPARAMETERS TRANSFER ACROSS LLM FAMILIES?
#
# Claim tested:
#   RMU hyperparameters tuned on ONE source model (Zephyr) transfer
#   zero-shot to other families (Llama-3.1, Qwen2.5, Gemma-2) when
#     (1) the steering coefficient is expressed relative to the
#         model's activation norm:   c = beta * N_layer
#     (2) the steered layer is chosen by RELATIVE depth:
#         layer = round(rho * num_layers)
#   whereas the original absolute settings (c = 6.5, layer 7)
#   do not.
#
# Experiments (all cached; rerunning resumes where it stopped):
#   E0  Baselines
#   E1  Norm profile per layer, every model            -> Fig 1
#   E2  beta sweep, every model (seed 42)              -> Fig 2 (scale collapse)
#   E3  Tune on source only (Zephyr): pick beta*, beta_A*
#   E4  Zero-shot transfer, 3 methods x 3 seeds        -> Table 1, Fig 3
#         RMU (c=6.5, layer 7)  | Adaptive RMU (beta_A*, layer 7)
#         NormRMU (beta*, relative layer)  | Oracle (best beta per model)
#   E5  Depth ablation: NormRMU at absolute layer 7     -> Table 2
#   E6  Relearning attack (LoRA fine-tune on forget data) -> Fig 4
#
# Requirements:
#   pip install torch transformers datasets safetensors "lm_eval[hf]" \
#               peft pandas matplotlib
#
# GPU: one 48 GB card (A6000) is enough. Full run ~ 60-70 unlearning
# runs + evals; set QUICK = True for a 1-seed first pass.
# ============================================================

import os

# ------------------------------------------------------------
# 0. USER SETTINGS
# ------------------------------------------------------------

HF_TOKEN_VALUE = ""          # only needed for official gated repos
USE_OFFICIAL_REPOS = False   # True -> meta-llama / google repos (needs token)

QUICK = os.environ.get("RMU_QUICK", "1") == "1"   # RMU_QUICK=0 for the full run

SEEDS = [42, 43, 44]
BETAS = [0.5, 1.0, 2.0, 4.0, 8.0]      # relative coefficients swept
SOURCE = "zephyr"                      # the ONLY model used for tuning

FIXED_C = 6.5                # original RMU coefficient (WMDP, Zephyr)
FIXED_LAYER = 7              # original RMU layer
RHO = 7 / 32                 # relative depth of layer 7 in Zephyr (32 layers)

LR = 5e-5
STEPS = 150
ALPHA = 1200.0               # retain weight (WMDP setting)
TRAIN_MAX_LEN = 512
EVAL_MAX_LEN = 512
PPL_BATCHES = 25

# Selection rule for "best" beta (applied identically everywhere):
#   minimise WMDP-Cyber accuracy subject to
#   MMLU drop <= MAX_MMLU_DROP points and general PPL ratio <= MAX_PPL_RATIO
MAX_MMLU_DROP = 2.0
MAX_PPL_RATIO = 1.5

MMLU_LIMIT = 40              # questions per MMLU subject (None = full MMLU)

RELEARN = True
RELEARN_STEPS = 50
RELEARN_LR = 1e-4
RELEARN_RANK = 16

if QUICK:
    SEEDS = [42]
    BETAS = [1.0, 2.0, 4.0]
    RELEARN = False
    MMLU_LIMIT = 20

if HF_TOKEN_VALUE:
    os.environ["HF_TOKEN"] = HF_TOKEN_VALUE

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # override: CUDA_VISIBLE_DEVICES=3 python ...
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import importlib.util

_required = {
    "torch": "torch", "transformers": "transformers", "datasets": "datasets",
    "safetensors": "safetensors", "lm_eval": '"lm_eval[hf]"', "peft": "peft",
    "pandas": "pandas", "matplotlib": "matplotlib",
}
def _installed(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ValueError, ImportError):
        return mod in __import__("sys").modules


_missing = [p for m, p in _required.items() if not _installed(m)]
if _missing:
    raise ImportError(
        "Missing packages. Run this, then restart the kernel:\n"
        "    %pip install " + " ".join(_missing)
    )

import json
import math
import time
import gc
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

HOME = os.environ.get("RMU_HOME", "/home/chintan/rmu_transfer")
PROJECT = os.path.join(HOME, "RMU-Thesis")


def pick_scratch():
    for cand in [os.path.join(PROJECT, "scratch"), os.path.join(HOME, "scratch"),
                 "/tmp/" + os.path.basename(HOME)]:
        try:
            os.makedirs(cand, exist_ok=True)
            test = os.path.join(cand, ".write_test")
            with open(test, "w") as f:
                f.write("ok")
            os.remove(test)
            return cand
        except OSError:
            continue
    raise RuntimeError("No writable scratch directory found.")


SCRATCH =  pick_scratch()
HF_CACHE = os.path.join(SCRATCH, "hf_cache")
RUNS = os.path.join(SCRATCH, "transfer_runs")        # deltas + training logs
OUT = os.path.join(PROJECT, "results_transfer")      # results, tables, figures
RES_DIR = os.path.join(OUT, "runs")
FIG_DIR = os.path.join(OUT, "figures")

for d in [PROJECT, SCRATCH, HF_CACHE, RUNS, OUT, RES_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# Keep the token from `huggingface-cli login` visible after moving HF_HOME
if not os.environ.get("HF_TOKEN"):
    for p in [os.path.join(os.environ.get("HF_HOME", ""), "token"),
              os.path.join(HOME, ".cache", "huggingface", "token")]:
        if os.path.isfile(p):
            with open(p) as f:
                tok = f.read().strip()
            if tok:
                os.environ["HF_TOKEN"] = tok
                break

os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE, "datasets")
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available; this pipeline needs a GPU.")

DEV = "cuda:0"
DTYPE = torch.bfloat16

print("=" * 70)
print("RMU TRANSFER STUDY")
print("=" * 70)
print("OUT     :", OUT)
print("RUNS    :", RUNS)
print("GPU     :", torch.cuda.get_device_name(0))
print("QUICK   :", QUICK, "| seeds:", SEEDS, "| betas:", BETAS)


# ------------------------------------------------------------
# 2. MODELS
# ------------------------------------------------------------

MODELS = {
    "zephyr": "HuggingFaceH4/zephyr-7b-beta",
    "llama8b": "unsloth/Llama-3.1-8B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma9b": "unsloth/gemma-2-9b-it",
}
if USE_OFFICIAL_REPOS:
    MODELS["llama8b"] = "meta-llama/Llama-3.1-8B-Instruct"
    MODELS["gemma9b"] = "google/gemma-2-9b-it"


def check_model_access(models):
    from huggingface_hub import hf_hub_download
    blocked = []
    for tag, repo in models.items():
        try:
            hf_hub_download(repo, "config.json")
        except Exception as e:
            blocked.append(f"{tag} ({repo}): {type(e).__name__}")
    if blocked:
        raise RuntimeError(
            "Cannot download:\n  " + "\n  ".join(blocked) +
            "\nSee HF_TOKEN_VALUE / USE_OFFICIAL_REPOS at the top."
        )
    print("Model access: all OK")


check_model_access(MODELS)

if SOURCE not in MODELS:
    raise ValueError(f"SOURCE '{SOURCE}' is not in MODELS")


# ------------------------------------------------------------
# 3. DATA
#    Forget corpus is split 80/20: unlearning + relearning use the
#    train part; forget-PPL is measured on the held-out part.
# ------------------------------------------------------------

def load_texts(name, split=None, min_len=0):
    if name == "wikitext":
        raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                           split=split or "test",
                           cache_dir=os.environ["HF_DATASETS_CACHE"])
    else:
        raw = load_dataset("cais/wmdp-corpora", name, split=split or "train",
                           cache_dir=os.environ["HF_DATASETS_CACHE"])
    out = []
    for ex in raw:
        t = ex.get("text", "")
        if t is None:
            continue
        t = str(t).strip()
        if len(t) > min_len:
            out.append(t)
    return out


def batchify(texts, bs=4):
    return [texts[i:i + bs] for i in range(0, len(texts), bs)]


print("\nLoading corpora...")
_forget = load_texts("cyber-forget-corpus")
_perm = np.random.default_rng(0).permutation(len(_forget))
_forget = [_forget[i] for i in _perm]
_cut = int(0.8 * len(_forget))

FORGET_TRAIN = batchify(_forget[:_cut])
FORGET_EVAL = batchify(_forget[_cut:])
LINKED = batchify(load_texts("cyber-retain-corpus"))
GENERAL = batchify(load_texts("wikitext", split="test"))
RETAIN_TRAIN = batchify(load_texts("wikitext", split="train", min_len=50))

for name, b in [("FORGET_TRAIN", FORGET_TRAIN), ("FORGET_EVAL", FORGET_EVAL),
                ("LINKED", LINKED), ("GENERAL", GENERAL),
                ("RETAIN_TRAIN", RETAIN_TRAIN)]:
    print(f"  {name:13s}: {len(b)} batches")
    if not b:
        raise RuntimeError(f"{name} is empty")


# ------------------------------------------------------------
# 4. MODEL HELPERS
# ------------------------------------------------------------

def free():
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def load_model(repo):
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    kw = dict(low_cpu_mem_usage=True)
    if "gemma-2" in repo.lower():
        kw["attn_implementation"] = "eager"
    try:
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=DTYPE, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=DTYPE, **kw)
    model = model.to(DEV).eval()
    model.config.use_cache = False
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    return model, tok


def get_layers(model):
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers
    if inner is not None and hasattr(inner, "language_model"):
        return inner.language_model.layers
    raise AttributeError(f"Cannot find decoder layers in {type(model).__name__}")


def tokenize(tok, texts, max_len):
    tok.padding_side = "right"
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len)
    return {k: v.to(DEV) for k, v in enc.items()}


class _Stop(Exception):
    pass


def layer_acts(model, enc, layer_id):
    """Output of decoder layer `layer_id`; forward stops there."""
    store = {}

    def hook(mod, inp, out):
        store["h"] = out[0] if isinstance(out, (tuple, list)) else out
        raise _Stop

    h = get_layers(model)[layer_id].register_forward_hook(hook)
    try:
        model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
              use_cache=False)
    except _Stop:
        pass
    finally:
        h.remove()
    if "h" not in store:
        raise RuntimeError("layer hook did not fire")
    return store["h"]


def apply_deltas(model, path):
    layers = get_layers(model)
    for key, delta in load_file(path).items():
        parts = key.split(".")
        mod = layers[int(parts[1])]
        for p in parts[2:-1]:
            mod = getattr(mod, p)
        param = getattr(mod, parts[-1])
        if param.shape != delta.shape:
            raise ValueError(f"shape mismatch for {key}")
        with torch.no_grad():
            param.copy_((param.float() + delta.to(param.device, torch.float32))
                        .to(param.dtype))
    return model


# ------------------------------------------------------------
# 5. E1 - NORM PROFILE (every layer, every model)
# ------------------------------------------------------------

NUM_LAYERS = {}
PROFILE = {}

PROFILE_PATH = os.path.join(OUT, "norm_profile.json")
if os.path.exists(PROFILE_PATH):
    with open(PROFILE_PATH) as f:
        _saved = json.load(f)
    PROFILE = _saved.get("profile", {})
    NUM_LAYERS = _saved.get("num_layers", {})


@torch.no_grad()
def norm_profile(repo, n=10):
    """Mean per-token L2 norm of every decoder layer's output on forget data."""
    model, tok = load_model(repo)
    L = len(get_layers(model))
    sums, count = np.zeros(L), 0
    for batch in FORGET_TRAIN[:n]:
        enc = tokenize(tok, batch, TRAIN_MAX_LEN)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        mask = enc["attention_mask"].bool()
        for i in range(L):
            sums[i] += hs[i + 1].float().norm(dim=-1)[mask].sum().item()
        count += mask.sum().item()
        del hs
    del model
    free()
    return (sums / max(count, 1)).tolist(), L


for tag, repo in MODELS.items():
    if tag not in PROFILE:
        print(f"\nNorm profile: {tag}")
        PROFILE[tag], NUM_LAYERS[tag] = norm_profile(repo)
        with open(PROFILE_PATH, "w") as f:
            json.dump({"profile": PROFILE, "num_layers": NUM_LAYERS}, f, indent=2)


def rel_layer(tag):
    """Layer at relative depth RHO (at least 2 so layers l-2..l exist)."""
    return max(2, min(NUM_LAYERS[tag] - 2, round(RHO * NUM_LAYERS[tag])))


print("\nLayer-norms:")
for tag in MODELS:
    lr_ = rel_layer(tag)
    print(f"  {tag:8s} L={NUM_LAYERS[tag]:2d}  N(layer 7)={PROFILE[tag][7]:8.2f}  "
          f"relative layer={lr_:2d}  N(rel)={PROFILE[tag][lr_]:8.2f}")


# ------------------------------------------------------------
# 6. UNLEARNING: RMU / ADAPTIVE RMU / NormRMU
# ------------------------------------------------------------

@contextmanager
def original_weights(params, originals):
    cur = [p.data for p in params]
    for p, o in zip(params, originals):
        p.data = o
    try:
        yield
    finally:
        for p, c in zip(params, cur):
            p.data = c


def masked_mse(a, b, mask):
    return ((a - b) ** 2 * mask).sum() / (mask.sum() * a.size(-1)).clamp_min(1.0)


def train_unlearn(repo, out_dir, mode, coeff, layer, seed):
    """
    mode = "fixed":    target = coeff * u
    mode = "adaptive": target = coeff * ||h_frozen(x)|| * u   (per sample)
    Trains mlp.down_proj of layers layer-2..layer with fp32 master weights.
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    f_order = rng.permutation(len(FORGET_TRAIN))
    r_order = rng.permutation(len(RETAIN_TRAIN))

    model, tok = load_model(repo)
    for p in model.parameters():
        p.requires_grad_(False)
    layers = get_layers(model)
    upd = [layer - 2, layer - 1, layer]
    params = [layers[i].mlp.down_proj.weight for i in upd]
    for p in params:
        p.requires_grad_(True)

    orig_bf16 = [p.detach().clone() for p in params]
    masters = [torch.nn.Parameter(p.detach().float().clone()) for p in params]
    orig_fp32 = [m.detach().clone() for m in masters]
    opt = torch.optim.AdamW(masters, lr=LR, weight_decay=0.0)

    g = torch.Generator().manual_seed(seed)
    u = torch.rand(model.config.hidden_size, generator=g)
    u = (u / u.norm()).to(DEV).view(1, 1, -1)

    log, t0 = [], time.time()
    for step in range(STEPS):
        f_enc = tokenize(tok, FORGET_TRAIN[f_order[step % len(f_order)]], TRAIN_MAX_LEN)
        r_enc = tokenize(tok, RETAIN_TRAIN[r_order[step % len(r_order)]], TRAIN_MAX_LEN)
        fm = f_enc["attention_mask"].unsqueeze(-1).float()
        rm = r_enc["attention_mask"].unsqueeze(-1).float()

        with torch.no_grad(), original_weights(params, orig_bf16):
            r_frozen = layer_acts(model, r_enc, layer).float()
            if mode == "adaptive":
                f_frozen = layer_acts(model, f_enc, layer).float()

        h_f = layer_acts(model, f_enc, layer).float()
        h_r = layer_acts(model, r_enc, layer).float()

        if mode == "adaptive":
            n = f_frozen.norm(dim=-1, keepdim=True)
            scale = coeff * (n * fm).sum(1, keepdim=True) / fm.sum(1, keepdim=True).clamp_min(1)
        else:
            scale = torch.tensor(float(coeff), device=DEV)

        f_loss = masked_mse(h_f, scale * u, fm)
        r_loss = masked_mse(h_r, r_frozen, rm)
        loss = f_loss + ALPHA * r_loss

        for p in params:
            p.grad = None
        loss.backward()
        for p, m in zip(params, masters):
            m.grad = p.grad.detach().float() if p.grad is not None else torch.zeros_like(m)
            p.grad = None
        opt.step()
        opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            for p, m in zip(params, masters):
                p.copy_(m.to(p.dtype))
            cos = F.cosine_similarity(h_f.detach(), u, dim=-1)
            cos = (cos * fm.squeeze(-1)).sum() / fm.sum().clamp_min(1)

        e = {"step": step, "loss": loss.item(), "forget_loss": f_loss.item(),
             "retain_loss": r_loss.item(), "cos": cos.item()}
        log.append(e)
        if not math.isfinite(e["loss"]):
            print(f"  non-finite loss at step {step}; stopping")
            break
        if step % 50 == 0 or step == STEPS - 1:
            print(f"  step {step:4d} | forget {e['forget_loss']:.4f} | "
                  f"retain {e['retain_loss']:.6f} | cos {e['cos']:.3f}")

    save_file({f"layers.{i}.mlp.down_proj.weight": (m.detach() - o).cpu().contiguous()
               for i, m, o in zip(upd, masters, orig_fp32)},
              os.path.join(out_dir, "deltas.safetensors"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"mode": mode, "coeff": coeff, "layer": layer, "seed": seed,
                   "seconds": round(time.time() - t0, 1), "log": log}, f, indent=2)

    del model, masters, orig_bf16, orig_fp32, opt
    free()


# ------------------------------------------------------------
# 7. EVALUATION
# ------------------------------------------------------------

TASKS = ["wmdp_cyber", "mmlu_computer_security", "mmlu_college_computer_science"]


def eval_bs(tag):
    return 4 if tag == "gemma9b" else 16


@torch.no_grad()
def ppl(model, tok, batches, n=PPL_BATCHES, max_len=EVAL_MAX_LEN):
    nll, ntok = 0.0, 0
    for batch in batches[:n]:
        enc = tokenize(tok, batch, max_len)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        logits = model(**enc, use_cache=False).logits[:, :-1].float()
        lab = labels[:, 1:]
        nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), lab.reshape(-1),
                               ignore_index=-100, reduction="sum").item()
        ntok += (lab != -100).sum().item()
        del logits
    return math.exp(min(nll / max(ntok, 1), 20.0))


def _acc(res, task):
    for src in ("results", "groups"):
        r = res.get(src, {}).get(task)
        if r:
            for k in ("acc,none", "acc", "acc_norm,none"):
                if k in r:
                    return float(r[k]) * 100
    if task == "mmlu":   # very old lm-eval: average the subjects
        vals = [float(v["acc,none"]) for k, v in res["results"].items()
                if k.startswith("mmlu_") and "acc,none" in v]
        if vals:
            return 100 * float(np.mean(vals))
    raise KeyError(f"no accuracy for {task}: {list(res.get('results', {}))[:10]}")


def lm_eval_acc(model, tok, bs, tasks, limit=None):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs, max_batch_size=bs,
              backend="causal")
    res = simple_evaluate(model=lm, tasks=tasks, batch_size=bs, limit=limit)
    tok.padding_side = "right"
    return {t: _acc(res, t) for t in tasks}


def evaluate_model(model, tok, bs, full=True):
    acc = lm_eval_acc(model, tok, bs, TASKS)
    if not full:
        return {"acc": acc}
    acc.update(lm_eval_acc(model, tok, bs, ["mmlu"], limit=MMLU_LIMIT))
    return {
        "acc": acc,
        "ppl": {
            "forget": ppl(model, tok, FORGET_EVAL),
            "linked": ppl(model, tok, LINKED),
            "general": ppl(model, tok, GENERAL),
        },
    }


def _save(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _load(path):
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------
# 8. E0 - BASELINES
# ------------------------------------------------------------

BASE = {}
for tag, repo in MODELS.items():
    path = os.path.join(RES_DIR, f"base_{tag}.json")
    if os.path.exists(path):
        BASE[tag] = _load(path)
        continue
    print(f"\nBaseline: {tag}")
    model, tok = load_model(repo)
    BASE[tag] = evaluate_model(model, tok, eval_bs(tag))
    del model
    free()
    _save(path, BASE[tag])

print("\nBaselines:")
for tag in MODELS:
    a = BASE[tag]["acc"]
    print(f"  {tag:8s} WMDP {a['wmdp_cyber']:5.1f}  MMLU {a['mmlu']:5.1f}  "
          f"forgetPPL {BASE[tag]['ppl']['forget']:7.2f}")


# ------------------------------------------------------------
# 9. RUN ONE CONFIG (cached)
#    method: "rmu"     -> fixed absolute coefficient
#            "armu"    -> Adaptive RMU (per-sample norm)
#            "normrmu" -> coefficient = beta * N(layer)  (ours)
# ------------------------------------------------------------

def run(tag, method, value, layer, seed):
    key = f"{tag}_{method}_v{value:g}_L{layer}_lr{LR:g}_s{STEPS}_sd{seed}"
    path = os.path.join(RES_DIR, key + ".json")
    if os.path.exists(path):
        return _load(path)

    if method == "rmu":
        mode, coeff = "fixed", value
    elif method == "armu":
        mode, coeff = "adaptive", value
    elif method == "normrmu":
        mode, coeff = "fixed", value * PROFILE[tag][layer]
    else:
        raise ValueError(method)

    run_dir = os.path.join(RUNS, key)
    delta = os.path.join(run_dir, "deltas.safetensors")
    if not os.path.exists(delta):
        print(f"\n[train] {key}  (effective c = {coeff:.3f})")
        train_unlearn(MODELS[tag], run_dir, mode, coeff, layer, seed)

    print(f"[eval ] {key}")
    model, tok = load_model(MODELS[tag])
    apply_deltas(model, delta)
    r = evaluate_model(model, tok, eval_bs(tag))
    del model
    free()

    r["cfg"] = {"tag": tag, "method": method, "value": value, "layer": layer,
                "seed": seed, "coeff_effective": coeff,
                "beta_effective": coeff / PROFILE[tag][layer] if mode == "fixed" else value,
                "lr": LR, "steps": STEPS, "alpha": ALPHA, "key": key}
    _save(path, r)
    return r


def flat(r):
    """One table row with deltas vs. the model's baseline."""
    c, b = r["cfg"], BASE[r["cfg"]["tag"]]
    return {
        **{k: c[k] for k in ["tag", "method", "value", "layer", "seed",
                             "coeff_effective", "beta_effective"]},
        "wmdp": r["acc"]["wmdp_cyber"],
        "mmlu": r["acc"]["mmlu"],
        "mmlu_sec": r["acc"]["mmlu_computer_security"],
        "mmlu_cs": r["acc"]["mmlu_college_computer_science"],
        "d_wmdp": r["acc"]["wmdp_cyber"] - b["acc"]["wmdp_cyber"],
        "d_mmlu": r["acc"]["mmlu"] - b["acc"]["mmlu"],
        "forget_ppl_ratio": r["ppl"]["forget"] / b["ppl"]["forget"],
        "linked_ppl_ratio": r["ppl"]["linked"] / b["ppl"]["linked"],
        "general_ppl_ratio": r["ppl"]["general"] / b["ppl"]["general"],
        "key": c["key"],
    }


def select_best(df):
    """Lowest WMDP among configs that keep utility; else the mildest config."""
    ok = df[(df["d_mmlu"] >= -MAX_MMLU_DROP) & (df["general_ppl_ratio"] <= MAX_PPL_RATIO)]
    if len(ok):
        return ok.sort_values("wmdp").iloc[0]
    return df.sort_values("value").iloc[0]


# ------------------------------------------------------------
# 10. E2 - BETA SWEEP ON EVERY MODEL (scale-collapse figure)
# ------------------------------------------------------------

print("\n" + "=" * 70 + "\nE2: beta sweep\n" + "=" * 70)
sweep = []
for tag in [SOURCE] + [t for t in MODELS if t != SOURCE]:
    for beta in BETAS:
        sweep.append(flat(run(tag, "normrmu", beta, rel_layer(tag), 42)))
df_sweep = pd.DataFrame(sweep)
df_sweep.to_csv(os.path.join(OUT, "E2_beta_sweep.csv"), index=False)


# ------------------------------------------------------------
# 11. E3 - TUNE ON THE SOURCE MODEL ONLY
# ------------------------------------------------------------

print("\n" + "=" * 70 + f"\nE3: tuning on {SOURCE} only\n" + "=" * 70)
BETA_STAR = float(select_best(df_sweep[df_sweep["tag"] == SOURCE])["value"])

armu_src = pd.DataFrame([flat(run(SOURCE, "armu", b, FIXED_LAYER, 42)) for b in BETAS])
armu_src.to_csv(os.path.join(OUT, "E3_armu_source_sweep.csv"), index=False)
BETA_A_STAR = float(select_best(armu_src)["value"])

ORACLE = {t: float(select_best(df_sweep[df_sweep["tag"] == t])["value"]) for t in MODELS}

_save(os.path.join(OUT, "selected_hparams.json"),
      {"source": SOURCE, "beta_star_normrmu": BETA_STAR,
       "beta_star_armu": BETA_A_STAR, "oracle_beta_per_model": ORACLE,
       "rule": f"min WMDP s.t. dMMLU >= -{MAX_MMLU_DROP} and general PPL ratio "
               f"<= {MAX_PPL_RATIO}"})
print(f"  NormRMU beta* = {BETA_STAR}   ARMU beta* = {BETA_A_STAR}")
print(f"  Oracle per model: {ORACLE}")


# ------------------------------------------------------------
# 12. E4 - ZERO-SHOT TRANSFER, 3 SEEDS
# ------------------------------------------------------------

print("\n" + "=" * 70 + "\nE4: zero-shot transfer\n" + "=" * 70)
METHODS = {
    "RMU (c=6.5, L7)": lambda t: ("rmu", FIXED_C, FIXED_LAYER),
    "Adaptive RMU": lambda t: ("armu", BETA_A_STAR, FIXED_LAYER),
    "NormRMU (ours)": lambda t: ("normrmu", BETA_STAR, rel_layer(t)),
}

main = []
for tag in MODELS:
    for name, spec in METHODS.items():
        m, v, L = spec(tag)
        for sd in SEEDS:
            main.append({**flat(run(tag, m, v, L, sd)), "Method": name})
    main.append({**flat(run(tag, "normrmu", ORACLE[tag], rel_layer(tag), 42)),
                 "Method": "Oracle (per-model tuned)"})
df_main = pd.DataFrame(main)
df_main.to_csv(os.path.join(OUT, "E4_transfer_all_runs.csv"), index=False)

agg_cols = ["wmdp", "mmlu", "d_wmdp", "d_mmlu", "forget_ppl_ratio",
            "linked_ppl_ratio", "general_ppl_ratio"]
df_agg = (df_main.groupby(["tag", "Method"])[agg_cols]
          .agg(["mean", "std"]).reset_index())
df_agg.columns = ["_".join(c).strip("_") for c in df_agg.columns]
df_agg.to_csv(os.path.join(OUT, "E4_transfer_summary.csv"), index=False)


# ------------------------------------------------------------
# 13. E5 - DEPTH ABLATION (NormRMU with absolute layer 7)
# ------------------------------------------------------------

print("\n" + "=" * 70 + "\nE5: depth ablation\n" + "=" * 70)
abl = []
for tag in MODELS:
    for name, L in [("relative depth", rel_layer(tag)), ("absolute layer 7", FIXED_LAYER)]:
        abl.append({**flat(run(tag, "normrmu", BETA_STAR, L, 42)), "Layer choice": name})
df_abl = pd.DataFrame(abl)
df_abl.to_csv(os.path.join(OUT, "E5_depth_ablation.csv"), index=False)


# ------------------------------------------------------------
# 14. E6 - RELEARNING ATTACK
#     LoRA fine-tune the unlearned model on forget-train text, then
#     re-measure WMDP-Cyber. Recovered accuracy = shallow unlearning.
# ------------------------------------------------------------

def relearn(tag, key, seed=42):
    path = os.path.join(RES_DIR, f"relearn_{key}.json")
    if os.path.exists(path):
        return _load(path)
    from peft import LoraConfig, get_peft_model

    print(f"\n[relearn] {key}")
    torch.manual_seed(seed)
    model, tok = load_model(MODELS[tag])
    apply_deltas(model, os.path.join(RUNS, key, "deltas.safetensors"))

    cfg = LoraConfig(r=RELEARN_RANK, lora_alpha=2 * RELEARN_RANK, lora_dropout=0.0,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                     task_type="CAUSAL_LM")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, cfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=RELEARN_LR)
    order = np.random.default_rng(seed + 1000).permutation(len(FORGET_TRAIN))
    for step in range(RELEARN_STEPS):
        enc = tokenize(tok, FORGET_TRAIN[order[step % len(order)]], TRAIN_MAX_LEN)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        loss = model(**enc, labels=labels, use_cache=False).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            print(f"  relearn step {step:3d} | loss {loss.item():.3f}")

    model = model.merge_and_unload()
    model.eval()
    model.gradient_checkpointing_disable()
    acc = lm_eval_acc(model, tok, eval_bs(tag), ["wmdp_cyber"])["wmdp_cyber"]
    del model, opt
    free()
    out = {"key": key, "wmdp_after_relearn": acc,
           "steps": RELEARN_STEPS, "lr": RELEARN_LR, "rank": RELEARN_RANK}
    _save(path, out)
    return out


df_rel = pd.DataFrame()
if RELEARN:
    print("\n" + "=" * 70 + "\nE6: relearning attack\n" + "=" * 70)
    rel = []
    for _, r in df_main[(df_main["seed"] == 42) &
                        (df_main["Method"] != "Oracle (per-model tuned)")].iterrows():
        rr = relearn(r["tag"], r["key"])
        rel.append({"tag": r["tag"], "Method": r["Method"],
                    "wmdp_base": BASE[r["tag"]]["acc"]["wmdp_cyber"],
                    "wmdp_unlearned": r["wmdp"],
                    "wmdp_relearned": rr["wmdp_after_relearn"]})
    df_rel = pd.DataFrame(rel)
    df_rel.to_csv(os.path.join(OUT, "E6_relearning.csv"), index=False)


# ------------------------------------------------------------
# 15. LATEX TABLES
# ------------------------------------------------------------

def pm(mean, std):
    return f"{mean:.1f}" if (std != std or std == 0) else f"{mean:.1f}$\\pm${std:.1f}"


lines = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"Model & Method & WMDP-Cyber $\downarrow$ & MMLU $\uparrow$ & "
         r"$\Delta$WMDP & $\Delta$MMLU \\", r"\midrule"]
for tag in MODELS:
    b = BASE[tag]["acc"]
    lines.append(f"{tag} & Base & {b['wmdp_cyber']:.1f} & {b['mmlu']:.1f} & -- & -- \\\\")
    for meth in list(METHODS) + ["Oracle (per-model tuned)"]:
        r = df_agg[(df_agg["tag"] == tag) & (df_agg["Method"] == meth)].iloc[0]
        lines.append(f" & {meth} & {pm(r.wmdp_mean, r.wmdp_std)} & "
                     f"{pm(r.mmlu_mean, r.mmlu_std)} & {pm(r.d_wmdp_mean, r.d_wmdp_std)} & "
                     f"{pm(r.d_mmlu_mean, r.d_mmlu_std)} \\\\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines.append(r"\end{tabular}")
with open(os.path.join(OUT, "table1_transfer.tex"), "w") as f:
    f.write("\n".join(lines))

lines = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"Model & Layer & Index & WMDP $\downarrow$ & MMLU $\uparrow$ & Gen.\ PPL ratio \\",
         r"\midrule"]
for _, r in df_abl.iterrows():
    lines.append(f"{r.tag} & {r['Layer choice']} & {int(r.layer)} & {r.wmdp:.1f} & "
                 f"{r.mmlu:.1f} & {r.general_ppl_ratio:.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
with open(os.path.join(OUT, "table2_depth_ablation.tex"), "w") as f:
    f.write("\n".join(lines))


# ------------------------------------------------------------
# 16. FIGURES
# ------------------------------------------------------------

import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
COL = {"zephyr": "#4C72B0", "llama8b": "#DD8452", "qwen7b": "#55A868",
       "gemma9b": "#C44E52"}
MRK = {"zephyr": "o", "llama8b": "s", "qwen7b": "^", "gemma9b": "D"}


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))
    fig.savefig(os.path.join(FIG_DIR, name + ".png"))
    plt.show()
    plt.close(fig)


# Fig 1: norm profile vs relative depth
fig, ax = plt.subplots(figsize=(3.4, 2.4))
for tag in MODELS:
    L = NUM_LAYERS[tag]
    # last entry is after the final norm layer in HF, so it is excluded
    ax.plot(np.arange(L - 1) / L, PROFILE[tag][:-1], color=COL[tag], label=tag, lw=1.5)
    ax.scatter([7 / L], [PROFILE[tag][7]], color=COL[tag], marker=MRK[tag], s=22, zorder=3)
ax.axhline(FIXED_C, ls="--", c="gray", lw=0.8)
ax.text(0.01, FIXED_C * 1.15, f"RMU c = {FIXED_C}", ha="left", fontsize=7, color="gray")
ax.set_yscale("log")
ax.set_xlabel("Relative depth")
ax.set_ylabel("Mean activation norm")
ax.legend(frameon=False, fontsize=7)
save(fig, "fig1_norm_profile")

# Fig 2: scale collapse - WMDP vs absolute c (left) and vs beta (right)
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
for tag in MODELS:
    d = df_sweep[df_sweep["tag"] == tag].sort_values("value")
    kw = dict(color=COL[tag], marker=MRK[tag], label=tag, ms=4, lw=1.2)
    axes[0].plot(d["coeff_effective"], d["wmdp"], **kw)
    axes[1].plot(d["value"], d["wmdp"], **kw)
    for ax in axes:
        ax.axhline(BASE[tag]["acc"]["wmdp_cyber"], color=COL[tag], ls=":", lw=0.7)
axes[0].axvline(FIXED_C, ls="--", c="gray", lw=0.8)
axes[0].set_xscale("log")
axes[1].set_xscale("log", base=2)
axes[0].set_xlabel("Absolute coefficient c")
axes[1].set_xlabel(r"Relative coefficient $\beta = c / N_\ell$")
axes[0].set_ylabel("WMDP-Cyber acc. (%)")
axes[1].legend(frameon=False, fontsize=7)
save(fig, "fig2_scale_collapse")

# Fig 3: forgetting vs utility, mean +- std over seeds
fig, ax = plt.subplots(figsize=(3.4, 2.6))
mstyle = {"RMU (c=6.5, L7)": "x", "Adaptive RMU": "v", "NormRMU (ours)": "*",
          "Oracle (per-model tuned)": "P"}
for _, r in df_agg.iterrows():
    ax.errorbar(r["d_mmlu_mean"], r["d_wmdp_mean"],
                xerr=0 if r["d_mmlu_std"] != r["d_mmlu_std"] else r["d_mmlu_std"],
                yerr=0 if r["d_wmdp_std"] != r["d_wmdp_std"] else r["d_wmdp_std"],
                fmt=mstyle.get(r["Method"], "o"), color=COL[r["tag"]], ms=6,
                elinewidth=0.8, capsize=2)
ax.axvline(-MAX_MMLU_DROP, ls=":", c="gray", lw=0.8)
ax.set_xlabel(r"$\Delta$MMLU (pts)")
ax.set_ylabel(r"$\Delta$WMDP-Cyber (pts)")
from matplotlib.lines import Line2D
h = [Line2D([], [], color="k", marker=m, ls="", label=k) for k, m in mstyle.items()]
h += [Line2D([], [], color=c, lw=3, label=t) for t, c in COL.items()]
ax.legend(handles=h, frameon=False, fontsize=6, ncol=2)
save(fig, "fig3_transfer_tradeoff")

# Fig 4: relearning
if len(df_rel):
    fig, ax = plt.subplots(figsize=(6.8, 2.3))
    meths = list(METHODS)
    w = 0.8 / len(meths)
    x = np.arange(len(MODELS))
    for i, m in enumerate(meths):
        d = df_rel[df_rel["Method"] == m].set_index("tag").reindex(list(MODELS))
        ax.bar(x + i * w, d["wmdp_unlearned"], w, color=f"C{i}", alpha=0.45)
        ax.bar(x + i * w, d["wmdp_relearned"] - d["wmdp_unlearned"], w,
               bottom=d["wmdp_unlearned"], color=f"C{i}", label=m)
    for j, t in enumerate(MODELS):
        ax.hlines(BASE[t]["acc"]["wmdp_cyber"], j - 0.1, j + 0.8, colors="k", lw=0.8)
    ax.axhline(25, ls=":", c="gray", lw=0.8)
    ax.set_xticks(x + w * (len(meths) - 1) / 2)
    ax.set_xticklabels(list(MODELS))
    ax.set_ylabel("WMDP-Cyber acc. (%)")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    save(fig, "fig4_relearning")


# ------------------------------------------------------------
# 17. SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70 + "\nHEADLINE NUMBERS\n" + "=" * 70)
print(f"Tuned on {SOURCE} only: NormRMU beta*={BETA_STAR}, ARMU beta*={BETA_A_STAR}")
show = df_agg[["tag", "Method", "wmdp_mean", "wmdp_std", "d_mmlu_mean",
               "general_ppl_ratio_mean"]].round(2)
print(show.to_string(index=False))

gaps = []
for tag in MODELS:
    o = df_agg[(df_agg.tag == tag) & (df_agg.Method == "Oracle (per-model tuned)")].iloc[0]
    for meth in METHODS:
        r = df_agg[(df_agg.tag == tag) & (df_agg.Method == meth)].iloc[0]
        gaps.append({"tag": tag, "Method": meth,
                     "transfer_gap_wmdp": r.wmdp_mean - o.wmdp_mean,
                     "transfer_gap_mmlu": r.mmlu_mean - o.mmlu_mean})
df_gap = pd.DataFrame(gaps)
df_gap.to_csv(os.path.join(OUT, "transfer_gap.csv"), index=False)
print("\nTransfer gap vs. per-model oracle (WMDP pts; closer to 0 = better transfer):")
print(df_gap.pivot(index="tag", columns="Method", values="transfer_gap_wmdp")
      .round(2).to_string())

print("\nAll outputs in:", OUT)
for root, _, files in os.walk(OUT):
    if root.endswith("runs"):
        continue
    for fn in sorted(files):
        print("  ", os.path.relpath(os.path.join(root, fn), OUT))
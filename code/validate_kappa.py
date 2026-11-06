# validate_kappa.py -- held-out test of the kappa selection rule (thesis Ch. 5.7)
# Built from noval_method.py; the unlearning, evaluation and selection code is
# unchanged. Differences: held-out models only, separate output folder,
# beta sweep (E2) only, and a final comparison of the kappa rule with the oracle.
# The kappa rule is fixed in advance exactly as written in the thesis:
#   choose the smallest beta whose final kappa >= 0.9; if none, the largest kappa.
#
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

QUICK = False                                      # validation always uses the full sweep

SEEDS = [42]
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

RELEARN = False
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
HF_CACHE = os.environ.get("KV_HF_CACHE", os.path.join(SCRATCH, "hf_cache"))  # reuse an existing cache if given
RUNS = os.path.join(SCRATCH, "transfer_runs")        # deltas + training logs
OUT = os.path.join(PROJECT, "results_kappa_validation")  # kept apart from the thesis results
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
print("KAPPA RULE VALIDATION (held-out models)")
print("=" * 70)
print("OUT     :", OUT)
print("RUNS    :", RUNS)
print("GPU     :", torch.cuda.get_device_name(0))
print("QUICK   :", QUICK, "| seeds:", SEEDS, "| betas:", BETAS)


# ------------------------------------------------------------
# 2. MODELS
# ------------------------------------------------------------

# Held-out models: none of these was used to design the kappa rule.
HOLDOUT = {
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "olmo7b":    "allenai/OLMo-2-1124-7B-Instruct",
    "qwen3b":    "Qwen/Qwen2.5-3B-Instruct",
    "qwen14b":   "Qwen/Qwen2.5-14B-Instruct",
}
_only = os.environ.get("KV_MODELS")              # e.g. KV_MODELS=mistral7b,qwen3b
MODELS = {t: r for t, r in HOLDOUT.items() if not _only or t in _only.split(",")}


def check_model_access(models):
    """Drop (with a message) any model that cannot be downloaded or that this
    transformers version cannot build, instead of stopping the whole run."""
    from transformers import AutoConfig
    for tag, repo in list(models.items()):
        try:
            AutoConfig.from_pretrained(repo)
        except Exception as e:
            print(f"SKIPPING {tag} ({repo}): {type(e).__name__}: {str(e)[:120]}")
            del models[tag]
    if not models:
        raise RuntimeError("No held-out model is usable.")
    print("Models in this run:", list(models))


check_model_access(MODELS)

SOURCE = next(iter(MODELS))   # only used for ordering in this script


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
    return 4 if tag in ("gemma9b", "qwen14b") else 16


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
# 11. KAPPA RULE vs ORACLE on held-out models
# ------------------------------------------------------------

KAPPA_T = 0.9


def final_kappa(tag, beta):
    key = f"{tag}_normrmu_v{beta:g}_L{rel_layer(tag)}_lr{LR:g}_s{STEPS}_sd42"
    return _load(os.path.join(RUNS, key, "meta.json"))["log"][-1]["cos"]


rows = []
for tag in MODELS:
    d = df_sweep[df_sweep["tag"] == tag].copy()
    d["kappa"] = [final_kappa(tag, b) for b in d["value"]]
    ok = d[d["kappa"] >= KAPPA_T].sort_values("value")
    rule_beta = float(ok.iloc[0]["value"]) if len(ok) else float(d.sort_values("kappa").iloc[-1]["value"])
    oracle = select_best(d)
    oracle_beta = float(oracle["value"])
    wm = lambda b: float(d[d["value"] == b]["wmdp"].iloc[0])
    rows.append({"model": tag, "layers": NUM_LAYERS[tag], "layer": rel_layer(tag),
                 "N_layer": round(PROFILE[tag][rel_layer(tag)], 2),
                 "rule_beta": rule_beta, "oracle_beta": oracle_beta,
                 "match": rule_beta == oracle_beta,
                 "wmdp_base": round(BASE[tag]["acc"]["wmdp_cyber"], 2),
                 "wmdp_rule": round(wm(rule_beta), 2), "wmdp_oracle": round(wm(oracle_beta), 2),
                 "gap_rule": round(wm(rule_beta) - wm(oracle_beta), 2),
                 "kappas": {f"{b:g}": round(k, 3) for b, k in zip(d["value"], d["kappa"])}})
    d.to_csv(os.path.join(OUT, f"sweep_with_kappa_{tag}.csv"), index=False)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "kappa_rule_validation.csv"), index=False)
_save(os.path.join(OUT, "kappa_rule_validation.json"), rows)

print("\n" + "=" * 70 + "\nKAPPA RULE ON HELD-OUT MODELS\n" + "=" * 70)
print(res[["model", "layer", "N_layer", "rule_beta", "oracle_beta", "match",
           "wmdp_base", "wmdp_rule", "wmdp_oracle", "gap_rule"]].to_string(index=False))
print(f"\nRule matches oracle on {int(res['match'].sum())} of {len(res)} held-out models.")
print("Outputs in:", OUT)

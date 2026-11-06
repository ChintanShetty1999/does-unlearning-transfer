"""Core RMU utilities — architecture-agnostic."""
import os, json, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(name, device="cuda:0", dtype=torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, trust_remote_code=True,
        device_map={"": device}, attn_implementation="eager",
    )
    try:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=False)
    except Exception:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return model, tok

def get_layers(model):
    """Decoder layer list. Covers Llama/Mistral/Qwen2/Gemma2/OLMo2 layouts."""
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"Cannot locate decoder layers on {type(model).__name__}")

def get_update_params(model, layer_ids, param_substr="mlp.down_proj"):
    """NAME-based selection. Index-based selection (the reference impl)
    silently picks o_proj on models with attention biases, e.g. Qwen2.5."""
    for p in model.parameters():
        p.requires_grad_(False)
    layers, params, names = get_layers(model), [], []
    for lid in layer_ids:
        for n, p in layers[lid].named_parameters():
            if param_substr in n:
                p.requires_grad_(True)
                params.append(p); names.append(f"layers.{lid}.{n}")
    if not params:
        raise ValueError(f"No params matched '{param_substr}'")
    return params, names

def forward_with_cache(model, inputs, module, no_grad=True):
    cache = []
    def hook(mod, inp, out):
        cache.append(out[0] if isinstance(out, tuple) else out)
    h = module.register_forward_hook(hook)
    if no_grad:
        with torch.no_grad(): model(**inputs)
    else:
        model(**inputs)
    h.remove()
    return cache[0]

def load_corpus(name, min_len=0, batch_size=4):
    if name == "wikitext":
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        raw = load_dataset("cais/wmdp-corpora", name, split="train")
    texts = [str(x["text"]) for x in raw if len(x["text"]) > min_len]
    return [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

def save_deltas(params, names, baseline, out_dir, meta):
    """Store only the weight deltas (~100-300MB) instead of full checkpoints."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    deltas = {n: (p.detach() - baseline[n]).cpu().contiguous()
              for n, p in zip(names, params)}
    save_file(deltas, os.path.join(out_dir, "deltas.safetensors"))
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)

def apply_deltas(model, delta_path):
    from safetensors.torch import load_file
    deltas, layers = load_file(delta_path), get_layers(model)
    for key, d in deltas.items():
        _, lid, rest = key.split(".", 2)
        mod = layers[int(lid)]
        for part in rest.split(".")[:-1]:
            mod = getattr(mod, part)
        tgt = getattr(mod, rest.split(".")[-1])
        tgt.data += d.to(tgt.device, tgt.dtype)
    return model

@torch.no_grad()
def corpus_perplexity(model, tok, batches, max_len=512, n_batches=25):
    """Token-level perplexity over a corpus. No floor, unlike MCQ accuracy."""
    import math
    total_nll, total_tok = 0.0, 0
    for b in batches[:n_batches]:
        enc = tok(b, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(model.device)
        ids, am = enc["input_ids"], enc["attention_mask"]
        labels = ids.clone()
        labels[am == 0] = -100
        out = model(input_ids=ids, attention_mask=am)
        logits = out.logits[:, :-1].float()
        tgt = labels[:, 1:]
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
            ignore_index=-100, reduction="sum")
        n = (tgt != -100).sum().item()
        if n:
            total_nll += nll.item(); total_tok += n
    if total_tok == 0:
        return float("nan")
    return float(math.exp(min(total_nll / total_tok, 20)))  # cap at e^20

import argparse, json, os, torch, numpy as np
os.environ.setdefault("HF_HOME", "/raid/chintan/hf_cache")
from rmu_core import load_model, get_layers, load_corpus

@torch.no_grad()
def profile(model, tok, batches, max_len, n_batches):
    layers = get_layers(model)
    acc = {i: [] for i in range(len(layers))}
    kurt = {i: [] for i in range(len(layers))}
    cache = {}
    hooks = [l.register_forward_hook(
        lambda m, i, o, idx=k: cache.__setitem__(idx, o[0] if isinstance(o, tuple) else o))
        for k, l in enumerate(layers)]
    for b in batches[:n_batches]:
        enc = tok(b, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(model.device)
        model(**enc)
        mask = enc["attention_mask"].bool()
        for k in acc:
            h = cache[k].float()
            valid = h[mask]                       # drop pad positions
            acc[k].append(valid.norm(dim=-1).mean().item())
            m = valid.mean(0)
            c = valid - m
            kurt[k].append(((c**4).mean() / (c.var(0).mean()**2 + 1e-9)).item())
    for h in hooks: h.remove()
    return ([float(np.mean(v)) for v in acc.values()],
            [float(np.mean(v)) for v in kurt.values()])

def main(a):
    model, tok = load_model(a.model, "cuda:0")
    model.eval()
    fb = load_corpus(a.forget_corpus, batch_size=4)
    rb = load_corpus(a.retain_corpus, batch_size=4)
    fn, fk = profile(model, tok, fb, 768, a.n_batches)
    rn, rk = profile(model, tok, rb, 512, a.n_batches)
    out = dict(model=a.model, n_layers=len(fn),
               hidden_size=model.config.hidden_size,
               forget_norms=fn, retain_norms=rn,
               forget_kurtosis=fk, retain_kurtosis=rk)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"{a.model}  layers={len(fn)}  h={model.config.hidden_size}")
    for l in [3, 5, 7, len(fn)//2, len(fn)-3]:
        if l < len(fn):
            print(f"  L{l:2d}  forget|h|={fn[l]:8.2f}  retain|h|={rn[l]:8.2f}  "
                  f"c=6.5 ratio={6.5/fn[l]:6.3f}")
    print(f"[saved] {a.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--forget_corpus", default="cyber-forget-corpus")
    p.add_argument("--retain_corpus", default="wikitext")
    p.add_argument("--n_batches", type=int, default=25)
    p.add_argument("--out", required=True)
    main(p.parse_args())

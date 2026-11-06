import argparse, json, os, time, numpy as np, torch, tqdm
from rmu_core import (load_model, get_layers, get_update_params,
                      forward_with_cache, load_corpus, save_deltas)

def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    frozen, tok = load_model(a.model, a.device)
    updated, _  = load_model(a.model, a.device)
    frozen.eval(); updated.train()

    params, names = get_update_params(updated, a.layer_ids, a.param_substr)
    baseline = {n: p.detach().clone() for n, p in zip(names, params)}
    print(f"[params] updating {len(params)}: {names}")

    f_mod = get_layers(updated)[a.layer_id]
    z_mod = get_layers(frozen)[a.layer_id]
    opt = torch.optim.AdamW(params, lr=a.lr)

    forget = load_corpus(a.forget_corpus, a.min_len, a.batch_size)
    retain = load_corpus(a.retain_corpus, a.min_len, a.batch_size)
    n = min(a.max_num_batches, len(forget), len(retain))

    hid = updated.config.hidden_size
    rv = torch.rand(1, 1, hid, dtype=updated.dtype, device=updated.device)
    control = rv / rv.norm() * a.steering_coeff

    tok.truncation_side = "right"
    log, t0 = [], time.time()
    for i in tqdm.tqdm(range(n)):
        fi = tok(forget[i], return_tensors="pt", padding=True,
                 truncation=True, max_length=a.forget_max_len).to(updated.device)
        fa = forward_with_cache(updated, fi, f_mod, no_grad=False)
        unlearn_loss = torch.nn.functional.mse_loss(fa, control)

        ri = tok(retain[i], return_tensors="pt", padding=True,
                 truncation=True, max_length=a.retain_max_len).to(updated.device)
        ra = forward_with_cache(updated, ri, f_mod, no_grad=False)
        rz = forward_with_cache(frozen,  ri, z_mod, no_grad=True)
        retain_loss = torch.nn.functional.mse_loss(ra, rz) * a.alpha

        loss = unlearn_loss + retain_loss
        opt.zero_grad(); loss.backward(); opt.step()

        if i % 10 == 0 or i == n - 1:
            with torch.no_grad():
                fz = forward_with_cache(frozen, fi, z_mod, no_grad=True)
                rec = dict(step=i, loss=loss.item(),
                           unlearn=unlearn_loss.item(), retain=retain_loss.item(),
                           forget_norm_after=fa.norm(dim=-1).mean().item(),
                           forget_norm_before=fz.norm(dim=-1).mean().item(),
                           cos=torch.nn.functional.cosine_similarity(
                               fa.float(), fz.float(), dim=-1).mean().item())
            log.append(rec); print(rec, flush=True)

    meta = vars(a) | {"names": names, "minutes": (time.time()-t0)/60, "log": log}
    save_deltas(params, names, baseline, a.out_dir, meta)
    print(f"[done] deltas -> {a.out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceH4/zephyr-7b-beta")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--forget_corpus", default="cyber-forget-corpus")
    p.add_argument("--retain_corpus", default="wikitext")
    p.add_argument("--forget_max_len", type=int, default=768)
    p.add_argument("--retain_max_len", type=int, default=512)
    p.add_argument("--layer_id", type=int, default=7)
    p.add_argument("--layer_ids", type=int, nargs="+", default=[5,6,7])
    p.add_argument("--param_substr", default="mlp.down_proj")
    p.add_argument("--steering_coeff", type=float, default=6.5)
    p.add_argument("--alpha", type=float, default=1200.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min_len", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_num_batches", type=int, default=75)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())

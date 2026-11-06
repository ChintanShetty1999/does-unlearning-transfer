import argparse, json, os, torch
os.environ.setdefault("HF_HOME", "/raid/chintan/hf_cache")
from rmu_core import load_model, apply_deltas, load_corpus, corpus_perplexity
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate

def main(a):
    model, tok = load_model(a.model, "cuda:0")
    tag = "base"
    if a.deltas:
        model = apply_deltas(model, a.deltas)
        tag = os.path.basename(os.path.dirname(a.deltas))
    model.eval()
    tok.padding_side = "left"

    lm = HFLM(pretrained=model, tokenizer=tok,
              batch_size=a.batch_size, max_batch_size=a.batch_size,
              device="cuda:0", backend="causal")
    res = simple_evaluate(model=lm, tasks=a.tasks.split(","),
                          batch_size=a.batch_size)

    out = {t: {k: v for k, v in d.items() if "acc" in k}
           for t, d in res["results"].items()}

    ppl = {}
    if a.ppl:
        for tag, corpus, ml in [("forget", "cyber-forget-corpus", 768),
                                ("retain", "cyber-retain-corpus", 512),
                                ("wikitext", "wikitext", 512)]:
            try:
                ppl[tag] = corpus_perplexity(model, tok, load_corpus(corpus, batch_size=4),
                                             max_len=ml, n_batches=a.ppl_batches)
            except Exception as e:
                ppl[tag] = None
                print(f"[ppl {tag}] failed: {type(e).__name__}: {e}")
        print("perplexity:", json.dumps(ppl, indent=2))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"model": a.model, "deltas": a.deltas, "tag": tag,
               "results": out, "perplexity": ppl}, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"[saved] {a.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--deltas", default=None)
    p.add_argument("--tasks", default="wmdp_cyber,mmlu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--ppl", action="store_true", default=True)
    p.add_argument("--no_ppl", dest="ppl", action="store_false")
    p.add_argument("--ppl_batches", type=int, default=25)
    p.add_argument("--out", required=True)
    main(p.parse_args())

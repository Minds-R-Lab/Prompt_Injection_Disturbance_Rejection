"""Scale-up DETECTION: leave-one-family-out + semantic generalization, cached.

Cheap (no generation): uses disk-cached layer updates. For each seed it splits
the benign corpus, fits the content-aware mid-layer subspace detector, and
reports detection AUC under:
  * LOFO: fit on injections from all-but-one family, test on the held-out
    family (true generalization to an unseen attack TYPE);
  * SEM:  fit on all canonical families, test on paraphrased/translated
    injections (semantic, not lexical, generalization).
Aggregates mean +/- std over seeds. One model per invocation (use the ladder).

  python -m dobsteer.run_scale_detect --model Qwen/Qwen2.5-7B-Instruct --device cuda \
      --n-benign 400 --seeds 0,1,2 --k 32
"""
from __future__ import annotations

import argparse, json, statistics as st
import torch

from . import data, cache
from .run_phase2 import make_fmt
from .run_phase3 import template_span, content_slice, roc_auc


def last_content(u, span):              # u: (L,T,d) -> (L,d)
    lo, hi = content_slice(u.shape[1], span[0], span[1])
    return u[:, hi - 1, :].float()


def fit_detector(bxs, ixs, layers, k, span):
    """bxs: list of benign update tensors; ixs: list of injected update tensors.
    Returns V (subspace), coord mean/sd, Mahalanobis mean/std per layer."""
    fb = torch.nan_to_num(torch.stack([last_content(u, span) for u in bxs]))   # (Nb,L,d)
    fi = torch.nan_to_num(torch.stack([last_content(u, span) for u in ixs]))   # (Ni,L,d)
    D = fi - fb.mean(0, keepdim=True)                               # mean-centered
    V = {}
    for l in layers:
        _, _, Vh = torch.linalg.svd(D[:, l, :], full_matrices=False)
        V[l] = Vh[:k].contiguous()
    mu, sd, mbar, msd = {}, {}, {}, {}
    for l in layers:
        cs = []
        for u in bxs:
            lo, hi = content_slice(u.shape[1], span[0], span[1])
            cs.append(u[l, lo:hi].float() @ V[l].T)
        c = torch.cat(cs)
        mu[l] = c.mean(0); sd[l] = c.std(0).clamp_min(1e-6)
        m = ((c - mu[l]) / sd[l]).norm(dim=-1)
        mbar[l] = m.mean(); msd[l] = m.std().clamp_min(1e-6)
    return V, mu, sd, mbar, msd


def score(u, V, mu, sd, mbar, msd, det_layers, span):
    lo, hi = content_slice(u.shape[1], span[0], span[1])
    zs = []
    for l in det_layers:
        c = u[l, lo:hi].float() @ V[l].T
        m = ((c - mu[l]) / sd[l]).norm(dim=-1)
        zs.append((m - mbar[l]) / msd[l])
    return torch.stack(zs).mean(0).max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-benign", type=int, default=400)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--placements", default="append,mid,document")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--dtype", default="fp16", choices=["fp16","bf16","fp32"],
                    help="bf16 recommended for Gemma-2 (fp16 overflows -> NaN)")
    ap.add_argument("--quant", default="none", choices=["none","4bit","8bit"],
                    help="bitsandbytes quantization for big models (needs bitsandbytes)")
    ap.add_argument("--device-map", default="none",
                    help="\"auto\" to shard across visible GPUs (else single device)")
    ap.add_argument("--out", default="scale_detect.json")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    placements = args.placements.split(",")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    load_kw = {}
    if args.quant in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=(args.quant == "4bit"), load_in_8bit=(args.quant == "8bit"),
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        load_kw["device_map"] = "auto" if args.device_map == "none" else args.device_map
    else:
        _dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
        load_kw["dtype"] = _dt if args.device == "cuda" else torch.float32
        if args.device_map != "none":
            load_kw["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    if "device_map" not in load_kw:
        model = model.to(args.device)
    model = model.eval()
    fmt = make_fmt(tok)
    span = template_span(tok, fmt)
    L = model.config.num_hidden_layers
    det_layers = list(range(L // 4, 3 * L // 4))
    uc = lambda text: cache.updates(model, tok, args.model, text, args.device,
                                    use_cache=not args.no_cache)

    benign = data.load_benign(n=args.n_benign, seed=12345)
    fams = data.injection_families()
    sem = data.held_out_semantic()
    fam_names = list(fams)
    print(f"Model={args.model} benign={len(benign)} families={fam_names} "
          f"placements={placements} span={span}")

    # precompute benign caches once
    print("  caching benign traces...")
    for p in benign:
        uc(fmt(p))

    per_seed = {"lofo": {f: [] for f in fam_names}, "lofo_pooled": [], "sem": []}
    for seed in seeds:
        rng = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(benign), generator=rng).tolist()
        n_tr = len(benign) // 2
        tr_b = [benign[i] for i in idx[:n_tr]]
        te_b = [benign[i] for i in idx[n_tr:]]
        tr_bx = [uc(fmt(p)) for p in tr_b]
        te_bx = [uc(fmt(p)) for p in te_b]
        # ---- LOFO ----
        pooled_pos, pooled_neg = [], []
        for held in fam_names:
            tr_inj_texts = [d["text"] for f in fam_names if f != held for d in fams[f]]
            te_inj_texts = [d["text"] for d in fams[held]]
            # train injected updates: sample (benign x injection) pairs
            tr_ix = []
            for i, p in enumerate(tr_b):
                t = tr_inj_texts[i % len(tr_inj_texts)]
                tr_ix.append(uc(fmt(data.place(p, t, "append"))))
            V, mu, sd, mbar, msd = fit_detector(tr_bx, tr_ix, det_layers, args.k, span)
            neg = [score(u, V, mu, sd, mbar, msd, det_layers, span) for u in te_bx]
            pos = []
            for p in te_b:
                for t in te_inj_texts:
                    for pl in placements:
                        pos.append(score(uc(fmt(data.place(p, t, pl))),
                                         V, mu, sd, mbar, msd, det_layers, span))
            auc = roc_auc(pos, neg)
            per_seed["lofo"][held].append(auc)
            pooled_pos += pos; pooled_neg += neg
        per_seed["lofo_pooled"].append(roc_auc(pooled_pos, pooled_neg))

        # ---- semantic generalization: fit on ALL canonical families ----
        all_inj = [d["text"] for f in fam_names for d in fams[f]]
        tr_ix = [uc(fmt(data.place(p, all_inj[i % len(all_inj)], "append")))
                 for i, p in enumerate(tr_b)]
        V, mu, sd, mbar, msd = fit_detector(tr_bx, tr_ix, det_layers, args.k, span)
        neg = [score(u, V, mu, sd, mbar, msd, det_layers, span) for u in te_bx]
        sem_pos = []
        for p in te_b:
            for f in sem:
                for d in sem[f]:
                    for pl in placements:
                        sem_pos.append(score(uc(fmt(data.place(p, d["text"], pl))),
                                             V, mu, sd, mbar, msd, det_layers, span))
        per_seed["sem"].append(roc_auc(sem_pos, neg))
        print(f"  seed {seed}: LOFO pooled AUC={per_seed['lofo_pooled'][-1]:.3f}  "
              f"SEM AUC={per_seed['sem'][-1]:.3f}")

    def ms(xs):
        return st.mean(xs), (st.pstdev(xs) if len(xs) > 1 else 0.0)

    print("\n  === Detection AUC (mean +/- std over seeds) ===")
    for f in fam_names:
        m, s = ms(per_seed["lofo"][f])
        print(f"  LOFO  {f:>18}: {m:.3f} +/- {s:.3f}")
    m, s = ms(per_seed["lofo_pooled"]); print(f"  LOFO  {'POOLED':>18}: {m:.3f} +/- {s:.3f}")
    m, s = ms(per_seed["sem"]); print(f"  SEM   {'paraphrase+trans':>18}: {m:.3f} +/- {s:.3f}")

    with open(args.out, "w") as f:
        json.dump({"model": args.model, "n_benign": len(benign),
                   "placements": placements, "per_seed": per_seed}, f, indent=1)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

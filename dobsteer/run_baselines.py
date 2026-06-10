"""External baselines + hard-negative FPR + detection overhead (referee W5b/W5d/Q5).

Compares our matched-filter detector against:
  * perplexity filter  -- mean token NLL of the prompt (catches gibberish
    suffixes, not clean injections; the standard cheap baseline);
  * linear probe       -- logistic regression on mean-pooled mid-layer
    activations, evaluated leave-one-family-out.
Also reports false-positive rate on the instruction-bearing benign HARD
NEGATIVES (the imperativeness confound, Q5) and the wall-clock overhead of the
detection forward pass + position scan.

  python -m dobsteer.run_baselines --model Qwen/Qwen2.5-7B-Instruct --device cuda --n-benign 200
"""
from __future__ import annotations

import argparse, json, time, statistics as st
import torch

from . import data, cache
from .run_phase2 import make_fmt
from .run_phase3 import template_span, content_slice, roc_auc
from .run_scale_detect import last_content, fit_detector, score


@torch.no_grad()
def perplexity(model, tok, text, device):
    enc = tok(text, return_tensors="pt").to(device)
    out = model(**enc, labels=enc.input_ids)
    return float(out.loss)        # mean NLL; higher = more "surprising"


def linear_probe_features(u, det_layers, span):
    """Mean-pooled content-position mid-layer update, concatenated -> vector."""
    lo, hi = content_slice(u.shape[1], span[0], span[1])
    feats = [u[l, lo:hi].float().mean(0) for l in det_layers]
    return torch.cat(feats)


def fit_logreg(X, y, iters=300, lr=0.05, l2=1e-3):
    Xb = torch.cat([X, torch.ones(X.shape[0], 1)], 1)
    w = torch.zeros(Xb.shape[1], requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(iters):
        opt.zero_grad()
        p = torch.sigmoid(Xb @ w)
        loss = torch.nn.functional.binary_cross_entropy(p, yt) + l2 * (w[:-1] ** 2).sum()
        loss.backward(); opt.step()
    return w.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-benign", type=int, default=200)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out", default="baselines.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if args.device == "cuda" else torch.float32,
    ).to(args.device).eval()
    fmt = make_fmt(tok)
    span = template_span(tok, fmt)
    L = model.config.num_hidden_layers
    det_layers = list(range(L // 4, 3 * L // 4))
    uc = lambda t: cache.updates(model, tok, args.model, t, args.device)

    benign = data.load_benign(n=args.n_benign, seed=7)
    hardneg = data.benign_hard_negatives()
    fams = data.injection_families(); fam_names = list(fams)
    n_tr = len(benign) // 2
    tr_b, te_b = benign[:n_tr], benign[n_tr:]
    print(f"Model={args.model} benign={len(benign)} hardneg={len(hardneg)} "
          f"families={fam_names}")

    tr_bx = [uc(fmt(p)) for p in tr_b]
    te_bx = [uc(fmt(p)) for p in te_b]
    hn_x = [uc(fmt(p)) for p in hardneg]

    # ---- our matched filter, LOFO pooled ----
    pooled_pos, pooled_neg, hn_scores = [], [], []
    for held in fam_names:
        tr_inj = [d["text"] for f in fam_names if f != held for d in fams[f]]
        tr_ix = [uc(fmt(data.place(p, tr_inj[i % len(tr_inj)], "append")))
                 for i, p in enumerate(tr_b)]
        V, mu, sd, mbar, msd = fit_detector(tr_bx, tr_ix, det_layers, args.k, span)
        pooled_neg += [score(u, V, mu, sd, mbar, msd, det_layers, span) for u in te_bx]
        for p in te_b:
            for d in fams[held]:
                pooled_pos.append(score(uc(fmt(data.place(p, d["text"], "append"))),
                                        V, mu, sd, mbar, msd, det_layers, span))
        hn_scores += [score(u, V, mu, sd, mbar, msd, det_layers, span) for u in hn_x]
    mf_auc = roc_auc(pooled_pos, pooled_neg)
    # hard-negative FPR at benign-q99 threshold
    thr = torch.quantile(torch.tensor(pooled_neg), 0.99).item()
    mf_hn_fpr = sum(s > thr for s in hn_scores) / max(len(hn_scores), 1)

    # ---- perplexity baseline ----
    ppl_neg = [perplexity(model, tok, fmt(p), args.device) for p in te_b]
    ppl_pos = [perplexity(model, tok, fmt(data.place(p, fams[f][0]["text"], "append")),
                          args.device) for f in fam_names for p in te_b]
    ppl_auc = roc_auc(ppl_pos, ppl_neg)

    # ---- linear probe (LOFO) ----
    probe_pos, probe_neg = [], []
    Xb = torch.stack([linear_probe_features(u, det_layers, span) for u in tr_bx])
    for held in fam_names:
        tr_inj = [d["text"] for f in fam_names if f != held for d in fams[f]]
        Xi = torch.stack([linear_probe_features(
            uc(fmt(data.place(p, tr_inj[i % len(tr_inj)], "append"))), det_layers, span)
            for i, p in enumerate(tr_b)])
        X = torch.cat([Xb, Xi]); y = [0]*len(Xb) + [1]*len(Xi)
        mvec = X.mean(0); sdv = X.std(0).clamp_min(1e-6)
        w = fit_logreg((X-mvec)/sdv, y)
        def pscore(u):
            f = (linear_probe_features(u, det_layers, span)-mvec)/sdv
            return float(torch.sigmoid(torch.cat([f, torch.ones(1)]) @ w))
        probe_neg += [pscore(u) for u in te_bx]
        for p in te_b:
            for d in fams[held]:
                probe_pos.append(pscore(uc(fmt(data.place(p, d["text"], "append")))))
    probe_auc = roc_auc(probe_pos, probe_neg)

    # ---- overhead: detection forward pass + scan, per prompt ----
    import time as _t
    V, mu, sd, mbar, msd = fit_detector(tr_bx, [uc(fmt(data.place(p, fams[fam_names[0]][0]["text"], "append"))) for p in tr_b], det_layers, args.k, span)
    t0 = _t.time()
    for p in te_b[:20]:
        _ = score(uc(fmt(p)), V, mu, sd, mbar, msd, det_layers, span)
    scan_ms = 1000 * (_t.time() - t0) / max(len(te_b[:20]), 1)
    t0 = _t.time()
    for p in te_b[:20]:
        cache.updates(model, tok, args.model, fmt(p), args.device, use_cache=False)
    fwd_ms = 1000 * (_t.time() - t0) / max(len(te_b[:20]), 1)

    print(f"\n  Detector            AUC (LOFO pooled)   hard-neg FPR")
    print(f"  matched filter (ours)   {mf_auc:.3f}            {mf_hn_fpr:.2f}")
    print(f"  perplexity filter       {ppl_auc:.3f}              --")
    print(f"  linear probe            {probe_auc:.3f}              --")
    print(f"\n  Overhead: detection forward pass {fwd_ms:.1f} ms/prompt, "
          f"scan {scan_ms:.2f} ms/prompt (cached features)")

    json.dump({"model": args.model, "mf_auc": mf_auc, "mf_hardneg_fpr": mf_hn_fpr,
               "ppl_auc": ppl_auc, "probe_auc": probe_auc,
               "fwd_ms": fwd_ms, "scan_ms": scan_ms}, open(args.out, "w"), indent=1)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

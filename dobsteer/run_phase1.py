"""Phase 1 v2: the disturbance observer as a white-box injection detector.

v1 failed (pooled AUC 0.55-0.67) for three reasons, all fixed here:
1. Surrogate overfit: v1 fit a d x d map from ~30 samples (one position per
   prompt). v2 fits on ALL token positions of the training prompts
   (hundreds-thousands of samples), giving a surrogate that generalizes.
2. Ignored low-rank structure: Phase 0 shows the injection signature is
   low-rank (eff. rank ~7-13) and prompt-consistent (cos ~0.93). v2 projects
   residuals onto an injection subspace learned from TRAINING injections.
3. Last-token only: v2 scores every position and aggregates (max z-score),
   since injected tokens perturb updates at their own positions.

Evaluation is leave-one-injection-out (LOIO): the subspace is fit on 4
injection strings and tested on the 5th, so AUC measures generalization to
unseen attacks. Both detectors are reported:
  norm  : z-scored full residual norm (no injection knowledge)
  proj  : z-scored residual norm inside the injection subspace (LOIO)

Usage:
  python -m dobsteer.run_phase1 --model Qwen/Qwen2.5-1.5B --device cuda
"""
from __future__ import annotations

import argparse
import torch

from .extraction import capture_trace
from .surrogate import fit_ridge
from .prompts import BASE_PROMPTS, INJECTIONS


def roc_auc(scores_pos, scores_neg) -> float:
    pos = torch.tensor(scores_pos).view(-1, 1)
    neg = torch.tensor(scores_neg).view(1, -1)
    greater = (pos > neg).float().sum()
    ties = (pos == neg).float().sum()
    return ((greater + 0.5 * ties) / (pos.numel() * neg.numel())).item()


def fit_all_positions(traces, lam=1e-1, max_samples=8192, seed=0):
    """Per-layer ridge fit using every token position of every trace."""
    g = torch.Generator().manual_seed(seed)
    L = len(traces[0].states) - 1
    models = []
    for l in range(L):
        X = torch.cat([t.states[l] for t in traces])
        Y = torch.cat([t.states[l + 1] for t in traces])
        if X.shape[0] > max_samples:
            idx = torch.randperm(X.shape[0], generator=g)[:max_samples]
            X, Y = X[idx], Y[idx]
        models.append(fit_ridge(X, Y, lam))
    return models


def residual(trace, models, l):
    A, b, _ = models[l]
    return trace.states[l + 1] - trace.states[l] @ A.T - b   # (T, d)


def norm_stats(traces, models, V=None):
    """Per-layer mean/std of (optionally projected) residual norms over all
    positions of the given benign traces."""
    mu, sd = [], []
    for l in range(len(models)):
        ns = []
        for t in traces:
            r = residual(t, models, l)
            if V is not None:
                r = r @ V[l].T
            ns.append(r.norm(dim=-1))
        ns = torch.cat(ns)
        mu.append(ns.mean())
        sd.append(ns.std().clamp_min(1e-6))
    return torch.stack(mu), torch.stack(sd)


def score(trace, models, mu, sd, V=None):
    """Prompt-level score: mean over layers of max-over-positions z-score."""
    zs = []
    for l in range(len(models)):
        r = residual(trace, models, l)
        if V is not None:
            r = r @ V[l].T
        z = (r.norm(dim=-1) - mu[l]) / sd[l]
        zs.append(z.max())
    return torch.stack(zs).mean().item()


def fit_injection_subspace(inj_traces, benign_traces, models, k=8):
    """Top-k right singular vectors of (injected - benign-mean-corrected)
    residuals per layer, from TRAINING injections only."""
    V = {}
    for l in range(len(models)):
        Ri = torch.cat([residual(t, models, l) for t in inj_traces])
        Rb = torch.cat([residual(t, models, l) for t in benign_traces])
        R = Ri - Rb.mean(0)            # remove benign residual bias
        _, _, Vh = torch.linalg.svd(R, full_matrices=False)
        V[l] = Vh[:k]
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="phase1_results.pt")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1e-1)
    ap.add_argument("--k", type=int, default=8, help="injection subspace rank")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    ).to(args.device).eval()

    n = len(BASE_PROMPTS)
    n_train = int(args.train_frac * n)
    train_p, test_p = BASE_PROMPTS[:n_train], BASE_PROMPTS[n_train:]
    print(f"Model={args.model}  train={n_train}  test={len(test_p)}  k={args.k}")

    cap = lambda p: capture_trace(model, tok, p, args.device)
    print("  capturing traces...")
    train_b = [cap(p) for p in train_p]
    test_b = [cap(p) for p in test_p]
    train_i = {j: [cap(p + inj) for p in train_p] for j, inj in enumerate(INJECTIONS)}
    test_i = {j: [cap(p + inj) for p in test_p] for j, inj in enumerate(INJECTIONS)}

    print("  fitting surrogates on all positions...")
    models = fit_all_positions(train_b, lam=args.lam)
    n_pos = sum(t.states[0].shape[0] for t in train_b)
    print(f"  surrogate samples per layer ~ {n_pos}")

    # ---- detector 1: plain z-scored residual norm (no injection knowledge)
    mu, sd = norm_stats(train_b, models)
    neg = [score(t, models, mu, sd) for t in test_b]
    res_norm = {}
    all_pos = []
    for j in test_i:
        pos = [score(t, models, mu, sd) for t in test_i[j]]
        res_norm[j] = roc_auc(pos, neg)
        all_pos += pos
    pooled_norm = roc_auc(all_pos, neg)

    # ---- detector 2: projected onto LOIO injection subspace
    res_proj = {}
    all_pos_p, all_neg_p = [], []
    for j in test_i:
        train_inj = [t for jj in train_i if jj != j for t in train_i[jj]]
        V = fit_injection_subspace(train_inj, train_b, models, k=args.k)
        mu_p, sd_p = norm_stats(train_b, models, V)
        neg_p = [score(t, models, mu_p, sd_p, V) for t in test_b]
        pos_p = [score(t, models, mu_p, sd_p, V) for t in test_i[j]]
        res_proj[j] = roc_auc(pos_p, neg_p)
        all_pos_p += pos_p
        all_neg_p += neg_p

    pooled_proj = roc_auc(all_pos_p, all_neg_p)

    print("\n  AUC per injection (norm / proj-LOIO):")
    for j, inj in enumerate(INJECTIONS):
        print(f"    {j}: {res_norm[j]:.3f} / {res_proj[j]:.3f}  {inj[:45]!r}")
    print(f"\n  Pooled AUC  norm={pooled_norm:.3f}  proj-LOIO={pooled_proj:.3f}")
    best = max(pooled_norm, pooled_proj)
    print(f"  H2 verdict: {'PASS' if best > 0.9 else 'FAIL'} @ 0.90 (best={best:.3f})")

    torch.save({"model": args.model, "auc_norm": res_norm, "auc_proj": res_proj,
                "pooled_norm": pooled_norm, "pooled_proj": pooled_proj},
               args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

"""Phase 1 v5: position-scanning matched filter.

v4 findings: (a) the append confound behind v3's AUC 1.0 was real (1.5B
dropped to 0.838 with benign-suffix decoys) but genuine signal survives at
7B (0.980); (b) mid-context injections were UNDETECTABLE (~0.44) because the
filter only read the LAST token, where no injection is being processed.

v5 scans: project the layer updates at EVERY position onto the injection
direction, z-score against benign all-position statistics, and take the max
over positions. A mid-context injection is then caught at its own tokens.

Detectors (LOIO over injection strings; negatives include benign-suffixed
decoys):
  norm     : z-scored residual norm, no attack knowledge (baseline)
  mf-last  : matched filter at last token only (v4, for comparison)
  mf-scan  : position-scanning matched filter on layer updates
  mfr-scan : position-scanning matched filter on surrogate residuals (DOB)

Usage:
  python -m dobsteer.run_phase1 --model Qwen/Qwen2.5-7B --device cuda
"""
from __future__ import annotations

import argparse
import torch

from .extraction import capture_trace
from .surrogate import fit_ridge
from .prompts import BASE_PROMPTS, INJECTIONS, BENIGN_SUFFIXES, place_mid


def roc_auc(scores_pos, scores_neg) -> float:
    pos = torch.tensor(scores_pos).view(-1, 1)
    neg = torch.tensor(scores_neg).view(1, -1)
    greater = (pos > neg).float().sum()
    ties = (pos == neg).float().sum()
    return ((greater + 0.5 * ties) / (pos.numel() * neg.numel())).item()


# ---------- features ----------

def fit_all_positions(traces, lam=1e-1, max_samples=8192, seed=0):
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


def updates_all(trace):
    """(L, T, d) layer updates at every position."""
    x = torch.stack(trace.states)            # (L+1, T, d)
    return x[1:] - x[:-1]


def residuals_all(trace, models):
    """(L, T, d) surrogate residuals at every position."""
    x = torch.stack(trace.states)
    return torch.stack([x[l + 1] - x[l] @ A.T - b
                        for l, (A, b, _) in enumerate(models)])


# ---------- baseline: residual norm ----------

def norm_stats(traces, models):
    mu, sd = [], []
    for l in range(len(models)):
        ns = torch.cat([(t.states[l + 1] - t.states[l] @ models[l][0].T
                         - models[l][1]).norm(dim=-1) for t in traces])
        mu.append(ns.mean()); sd.append(ns.std().clamp_min(1e-6))
    return torch.stack(mu), torch.stack(sd)


def score_norm(trace, models, mu, sd):
    zs = []
    for l in range(len(models)):
        r = trace.states[l + 1] - trace.states[l] @ models[l][0].T - models[l][1]
        zs.append(((r.norm(dim=-1) - mu[l]) / sd[l]).max())
    return torch.stack(zs).mean().item()


# ---------- matched filter ----------

def mf_directions(fb_train_last, fi_train_last_list):
    """Unit injection direction per layer from last-token paired deltas."""
    deltas = torch.cat([fi - fb_train_last for fi in fi_train_last_list])
    v = deltas.mean(0)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def scan_stats(traces, feats_fn, v):
    """Benign mean/std of the projection per layer, pooled over positions."""
    projs = torch.cat([(feats_fn(t) * v[:, None, :]).sum(-1) for t in traces], dim=1)
    return projs.mean(1), projs.std(1).clamp_min(1e-6)


def scan_score(trace, feats_fn, v, mu, sd, mid):
    """Max over positions of the mid-layer-mean z-scored projection."""
    z = ((feats_fn(trace) * v[:, None, :]).sum(-1) - mu[:, None]) / sd[:, None]
    return z[mid].mean(0).max().item()


def last_score(trace, feats_fn, v, mu, sd, mid):
    """v4-style: z-scored projection at the last position only."""
    z = ((feats_fn(trace)[:, -1, :] * v).sum(-1) - mu) / sd
    return z[mid].mean().item()


def mf_eval(feats_fn, scorer, train_b, train_i, negatives, positives, L):
    """LOIO evaluation of a matched-filter scorer."""
    fb_last = torch.stack([feats_fn(t)[:, -1, :] for t in train_b])
    fi_last = {j: torch.stack([feats_fn(t)[:, -1, :] for t in train_i[j]])
               for j in train_i}
    mid = torch.arange(L // 4, 3 * L // 4)

    per, all_p, all_n = {}, [], []
    for j in positives:
        v = mf_directions(fb_last, [fi_last[jj] for jj in fi_last if jj != j])
        mu, sd = scan_stats(train_b, feats_fn, v)
        neg = [scorer(t, feats_fn, v, mu, sd, mid) for t in negatives]
        pos = [scorer(t, feats_fn, v, mu, sd, mid) for t in positives[j]]
        per[j] = roc_auc(pos, neg)
        all_p += pos; all_n += neg
    return per, roc_auc(all_p, all_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="phase1_results.pt")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1e-1)
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
    print(f"Model={args.model}  train={n_train}  test={len(test_p)}")

    cap = lambda p: capture_trace(model, tok, p, args.device)
    print("  capturing traces...")
    train_b = [cap(p) for p in train_p]
    test_b = [cap(p) for p in test_p]
    test_b_sfx = [cap(p + s) for p in test_p for s in BENIGN_SUFFIXES]
    train_i = {j: [cap(p + inj) for p in train_p] for j, inj in enumerate(INJECTIONS)}
    test_i = {j: [cap(p + inj) for p in test_p] for j, inj in enumerate(INJECTIONS)}
    test_i_mid = {j: [cap(place_mid(p, inj)) for p in test_p]
                  for j, inj in enumerate(INJECTIONS)}
    L = len(train_b[0].states) - 1
    negatives = test_b + test_b_sfx

    print("  fitting surrogates...")
    models = fit_all_positions(train_b, lam=args.lam)
    upd = updates_all
    res = lambda t: residuals_all(t, models)

    # baseline
    mu_n, sd_n = norm_stats(train_b, models)
    neg_n = [score_norm(t, models, mu_n, sd_n) for t in negatives]
    pooled_norm = roc_auc(
        [score_norm(t, models, mu_n, sd_n) for j in test_i for t in test_i[j]], neg_n)

    rows = {}
    for name, positives in (("app", test_i), ("mid", test_i_mid)):
        rows[name] = {
            "mf-last": mf_eval(upd, last_score, train_b, train_i, negatives, positives, L),
            "mf-scan": mf_eval(upd, scan_score, train_b, train_i, negatives, positives, L),
            "mfr-scan": mf_eval(res, scan_score, train_b, train_i, negatives, positives, L),
        }

    print(f"\n  Negatives: {len(test_b)} bare + {len(test_b_sfx)} benign-suffixed decoys")
    print(f"  Pooled norm baseline (appended): {pooled_norm:.3f}")
    for name, label in (("app", "appended"), ("mid", "mid-context")):
        print(f"\n  [{label}] AUC per injection (mf-last / mf-scan / mfr-scan):")
        for j, inj in enumerate(INJECTIONS):
            print(f"    {j}: {rows[name]['mf-last'][0][j]:.3f} / "
                  f"{rows[name]['mf-scan'][0][j]:.3f} / "
                  f"{rows[name]['mfr-scan'][0][j]:.3f}  {inj[:38]!r}")
        pooled = {k: v[1] for k, v in rows[name].items()}
        print(f"    pooled: mf-last={pooled['mf-last']:.3f}  "
              f"mf-scan={pooled['mf-scan']:.3f}  mfr-scan={pooled['mfr-scan']:.3f}")
        best = max(pooled.values())
        print(f"    H2 ({label}): {'PASS' if best > 0.9 else 'FAIL'} @ 0.90 "
              f"(best={best:.3f})")

    torch.save({"model": args.model, "pooled_norm": pooled_norm,
                "rows": {n_: {k: (dict(v[0]), v[1]) for k, v in r.items()}
                         for n_, r in rows.items()}}, args.out)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

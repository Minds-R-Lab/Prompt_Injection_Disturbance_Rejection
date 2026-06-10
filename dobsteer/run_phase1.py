"""Phase 1 v3: matched-filter disturbance detection.

v2 plateaued at AUC ~0.77 with norm == proj, which is diagnostic: the SVD of
raw injected residuals recovers generic benign variance directions, not the
injection. The fix follows directly from Phase 0: the PAIRED delta direction
(update_injected - update_benign) is consistent across prompts (cos ~0.93),
so the optimal detector for a known low-rank signal in noise is a MATCHED
FILTER: project the test layer-update onto the (LOIO) mean delta direction
and z-score against benign statistics of that same projection.

Detectors reported (all leave-one-injection-out where applicable):
  norm    : z-scored full residual norm (no attack knowledge) — baseline
  mf-upd  : matched filter on raw layer updates u_l = x_{l+1} - x_l
  mf-res  : matched filter on surrogate residuals (DOB framing)

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


# ---------- shared pieces ----------

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


def updates_last(trace):
    """(L, d) layer updates at the last token position."""
    x = trace.at_position(-1)
    return x[1:] - x[:-1]


def residuals_last(trace, models):
    """(L, d) surrogate residuals at the last token position."""
    x = trace.at_position(-1)
    return torch.stack([x[l + 1] - x[l] @ A.T - b for l, (A, b, _) in enumerate(models)])


# ---------- detector 1: residual-norm baseline (v2) ----------

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


# ---------- detectors 2/3: matched filter ----------

def mf_directions(train_b_feats, train_i_feats_list):
    """v_l = normalized mean paired delta per layer.

    train_b_feats: (N, L, d) features (updates or residuals) of benign train.
    train_i_feats_list: list over injections of (N, L, d) injected features.
    Returns (L, d) unit directions.
    """
    deltas = torch.cat([fi - train_b_feats for fi in train_i_feats_list])  # (M, L, d)
    v = deltas.mean(0)                                                      # (L, d)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def mf_scores(feats, v, mu, sd, mid):
    """feats: (N, L, d); v: (L, d); z-scored projection, mean over mid layers."""
    proj = (feats * v).sum(-1)               # (N, L)
    z = (proj - mu) / sd
    return z[:, mid].mean(-1).tolist()


def mf_eval(get_feats, train_b, train_i, test_b, test_i, L):
    """LOIO matched-filter evaluation. get_feats: trace -> (L, d)."""
    fb_train = torch.stack([get_feats(t) for t in train_b])
    fb_test = torch.stack([get_feats(t) for t in test_b])
    fi_train = {j: torch.stack([get_feats(t) for t in train_i[j]]) for j in train_i}
    fi_test = {j: torch.stack([get_feats(t) for t in test_i[j]]) for j in test_i}
    mid = list(range(L // 4, 3 * L // 4))

    per, all_p, all_n = {}, [], []
    for j in fi_test:
        v = mf_directions(fb_train, [fi_train[jj] for jj in fi_train if jj != j])
        proj_b = (fb_train * v).sum(-1)                  # (N, L) benign stats
        mu, sd = proj_b.mean(0), proj_b.std(0).clamp_min(1e-6)
        neg = mf_scores(fb_test, v, mu, sd, mid)
        pos = mf_scores(fi_test[j], v, mu, sd, mid)
        per[j] = roc_auc(pos, neg)
        all_p += pos; all_n += neg
    return per, roc_auc(all_p, all_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
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
    train_i = {j: [cap(p + inj) for p in train_p] for j, inj in enumerate(INJECTIONS)}
    test_i = {j: [cap(p + inj) for p in test_p] for j, inj in enumerate(INJECTIONS)}
    L = len(train_b[0].states) - 1

    print("  fitting surrogates...")
    models = fit_all_positions(train_b, lam=args.lam)

    # 1) norm baseline
    mu, sd = norm_stats(train_b, models)
    neg = [score_norm(t, models, mu, sd) for t in test_b]
    auc_norm, all_pos = {}, []
    for j in test_i:
        pos = [score_norm(t, models, mu, sd) for t in test_i[j]]
        auc_norm[j] = roc_auc(pos, neg); all_pos += pos
    pooled_norm = roc_auc(all_pos, neg)

    # 2) matched filter on raw updates
    auc_upd, pooled_upd = mf_eval(updates_last, train_b, train_i, test_b, test_i, L)

    # 3) matched filter on surrogate residuals (DOB)
    auc_res, pooled_res = mf_eval(lambda t: residuals_last(t, models),
                                  train_b, train_i, test_b, test_i, L)

    print("\n  AUC per injection (norm / mf-upd / mf-res, LOIO):")
    for j, inj in enumerate(INJECTIONS):
        print(f"    {j}: {auc_norm[j]:.3f} / {auc_upd[j]:.3f} / {auc_res[j]:.3f}  {inj[:40]!r}")
    print(f"\n  Pooled AUC  norm={pooled_norm:.3f}  mf-upd={pooled_upd:.3f}  "
          f"mf-res={pooled_res:.3f}")
    best = max(pooled_norm, pooled_upd, pooled_res)
    print(f"  H2 verdict: {'PASS' if best > 0.9 else 'FAIL'} @ 0.90 (best={best:.3f})")

    torch.save({"model": args.model, "auc_norm": auc_norm, "auc_upd": auc_upd,
                "auc_res": auc_res, "pooled": {"norm": pooled_norm,
                "upd": pooled_upd, "res": pooled_res}}, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

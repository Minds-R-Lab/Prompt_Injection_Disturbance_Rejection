"""Phase 0 experiment: is the injection effect low-rank, additive, persistent?

Usage:
  python -m dobsteer.run_phase0 --model gpt2-large
  python -m dobsteer.run_phase0 --model Qwen/Qwen2.5-1.5B --device cuda
  python -m dobsteer.run_phase0 --model meta-llama/Llama-3.2-3B   # needs HF login

Verdicts:
  H1a low-rank   : energy captured at k=16 > 0.8 AND the test is meaningful
                   (n_pairs > 32, effective_rank > 16 possible).
  H1b additivity : mean cosine between disjoint-set mean deltas > 0.7 (mid layers).
  H1c persistence: relative ||Delta_l||/||f_l|| stays >~0.3 in mid/late layers.
"""
from __future__ import annotations

import argparse
import torch

from .extraction import capture_trace
from .disturbance import (paired_deltas, layer_updates, low_rank_analysis,
                          additivity_score, persistence_profile,
                          relative_persistence)
from .prompts import BASE_PROMPTS, INJECTIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-large")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="phase0_results.pt")
    ap.add_argument("--k", type=int, default=16, help="rank for the H1a test")
    ap.add_argument("--max-prompts", type=int, default=len(BASE_PROMPTS))
    args = ap.parse_args()

    prompts = BASE_PROMPTS[: args.max_prompts]
    n = len(prompts)
    print(f"Model={args.model}  n_pairs={n}  k={args.k}")
    if n <= 2 * args.k:
        print(f"  WARNING: n_pairs={n} <= 2*k={2*args.k}; the low-rank test is "
              f"rank-limited and may read PASS trivially. Use >=50 prompts.")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    ).to(args.device).eval()

    results = {"model": args.model, "n_pairs": n, "k": args.k, "injections": {}}
    for inj_idx, inj in enumerate(INJECTIONS):
        deltas, benign_updates = [], []
        for p in prompts:
            tb = capture_trace(model, tok, p, args.device)
            ti = capture_trace(model, tok, p + inj, args.device)
            deltas.append(paired_deltas(tb, ti))
            benign_updates.append(layer_updates(tb))
        deltas = torch.stack(deltas)               # (N, L, d)
        benign_updates = torch.stack(benign_updates)

        half = n // 2
        lr = low_rank_analysis(deltas, k_max=max(32, args.k))
        add = additivity_score(deltas[:half], deltas[half:])
        per_raw = persistence_profile(deltas)
        per_rel = relative_persistence(deltas, benign_updates)
        L = deltas.shape[1]
        mid = slice(L // 4, 3 * L // 4)

        ki = min(args.k - 1, lr[0]["energy"].shape[0] - 1)
        e_k = torch.stack([lr[l]["energy"][ki] for l in lr]).mean().item()
        eff_rank = sum(lr[l]["effective_rank"] for l in lr) / L
        add_mid = add[mid].mean().item()
        rel_mid = per_rel[mid].mean().item()

        meaningful = n > 2 * args.k
        h1a = e_k > 0.8 and meaningful
        results["injections"][inj_idx] = {
            "text": inj, "low_rank": lr, "additivity": add,
            "persistence_raw": per_raw, "persistence_rel": per_rel,
            "energy_at_k": e_k, "eff_rank": eff_rank,
            "add_mid": add_mid, "rel_mid": rel_mid,
        }

        print(f"\nInjection {inj_idx}: {inj[:50]!r}")
        print(f"  H1a low-rank : energy@k={args.k} {e_k:.3f}, "
              f"eff_rank {eff_rank:.1f}  "
              f"({'PASS' if h1a else 'INCONCLUSIVE' if not meaningful else 'FAIL'})")
        print(f"  H1b additive : mid-layer cos {add_mid:.3f}  "
              f"({'PASS' if add_mid > 0.7 else 'FAIL'} @ 0.7)")
        print(f"  H1c persist  : relative ||Delta||/||f|| mid {rel_mid:.2f}, "
              f"raw {per_raw[0]:.1f}->{per_raw[-1]:.1f}  "
              f"({'PASS' if rel_mid > 0.3 else 'FAIL'} @ 0.3)")

    torch.save(results, args.out)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

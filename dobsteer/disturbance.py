"""Phase 0: characterize the injection-induced disturbance.

Hypothesis H1: for paired prompts (benign P, injected P+I), the layer-update
difference Delta_l is approximately (a) low-rank across a prompt
distribution, (b) additive, (c) persistent across depth.
"""
from __future__ import annotations

import torch


def layer_updates(trace, pos: int = -1) -> torch.Tensor:
    """f_l(x_l) at one position: (L, d) tensor of x_{l+1} - x_l."""
    x = trace.at_position(pos)
    return x[1:] - x[:-1]


def paired_deltas(trace_benign, trace_injected, pos: int = -1) -> torch.Tensor:
    """Delta_l = update_l(P+I) - update_l(P), aligned at the answer position.

    Uses the last-token position of each prompt (the position from which the
    model generates), so sequence-length mismatch is irrelevant.
    """
    return layer_updates(trace_injected, pos) - layer_updates(trace_benign, pos)


def low_rank_analysis(deltas: torch.Tensor, k_max: int = 32):
    """deltas: (N_pairs, L, d). For each layer, SVD across pairs.

    Returns dict layer -> (singular values, energy captured at each k<=k_max,
    top-k_max right singular vectors). H1(a) holds if energy[16] > 0.8.
    """
    n, L, d = deltas.shape
    out = {}
    for l in range(L):
        M = deltas[:, l, :]                       # (N, d)
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        energy = (S**2).cumsum(0) / (S**2).sum().clamp_min(1e-12)
        out[l] = {"S": S, "energy": energy[:k_max], "V": Vh[:k_max]}
    return out


def additivity_score(deltas_a: torch.Tensor, deltas_b: torch.Tensor) -> torch.Tensor:
    """Does the same injection produce the same Delta on different base
    prompts? Cosine similarity per layer between mean deltas from two
    disjoint base-prompt sets. (L,) tensor; H1(b) holds if high (>0.7).
    """
    ma = deltas_a.mean(0)                          # (L, d)
    mb = deltas_b.mean(0)
    return torch.nn.functional.cosine_similarity(ma, mb, dim=-1)


def persistence_profile(deltas: torch.Tensor) -> torch.Tensor:
    """Mean ||Delta_l|| across pairs, per layer: (L,). H1(c): the disturbance
    does not vanish with depth (or we identify the depth band to control).
    """
    return deltas.norm(dim=-1).mean(0)

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
    """deltas: (N_pairs, L, d). For each layer, SVD of the delta matrix
    across pairs. H1(a) (low-rank) holds if energy[k] > 0.8 for small k.

    IMPORTANT: the SVD rank is min(N_pairs, d), so the energy-at-k test is
    only meaningful when N_pairs is comfortably larger than k. We also report
    `effective_rank` (participation ratio of the singular spectrum) and a
    `meaningful` flag (N_pairs > 2*k_eval) so trivially-rank-limited runs are
    not mistaken for genuine low-rank structure.

    Returns dict layer -> {S, energy[:k_max], V[:k_max], effective_rank}.
    """
    n, L, d = deltas.shape
    out = {}
    for l in range(L):
        M = deltas[:, l, :]                       # (N, d)
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        s2 = S**2
        energy = s2.cumsum(0) / s2.sum().clamp_min(1e-12)
        p = s2 / s2.sum().clamp_min(1e-12)
        eff_rank = (-(p * (p.clamp_min(1e-12)).log()).sum()).exp().item()
        out[l] = {"S": S, "energy": energy[:k_max], "V": Vh[:k_max],
                  "effective_rank": eff_rank, "n_pairs": n}
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
    """Mean ||Delta_l|| across pairs, per layer: (L,). Raw magnitude grows
    with depth simply because residual-stream norms grow, so prefer
    relative_persistence for the H1(c) verdict.
    """
    return deltas.norm(dim=-1).mean(0)


def relative_persistence(deltas: torch.Tensor,
                         benign_updates: torch.Tensor) -> torch.Tensor:
    """||Delta_l|| / ||f_l^benign|| per layer, averaged over pairs: (L,).

    Removes the depth-scaling confound: a value near/above 1 means the
    injection perturbs the layer update on the order of the update itself.

    deltas: (N, L, d); benign_updates: (N, L, d) = layer_updates of benign run.
    """
    num = deltas.norm(dim=-1)                          # (N, L)
    den = benign_updates.norm(dim=-1).clamp_min(1e-6)  # (N, L)
    return (num / den).mean(0)

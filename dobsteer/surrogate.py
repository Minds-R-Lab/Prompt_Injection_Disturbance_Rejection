"""Locally-linear surrogate of layer dynamics: x_{l+1} ~ A_l x_l + b_l.

Two estimators:
- Jacobian at a nominal point (exact local model, one prompt).
- Ridge regression over a prompt distribution (cheaper at use time,
  averaged model with measurable residual bound rho).
"""
from __future__ import annotations

import torch


def fit_ridge(states_in: torch.Tensor, states_out: torch.Tensor, lam: float = 1e-2):
    """states_in/out: (N, d). Returns (A, b, rho) with rho = max residual norm."""
    N, d = states_in.shape
    X = torch.cat([states_in, torch.ones(N, 1)], dim=1)       # (N, d+1)
    G = X.T @ X + lam * torch.eye(d + 1)
    W = torch.linalg.solve(G, X.T @ states_out)               # (d+1, d)
    A, b = W[:d].T, W[d]
    resid = states_out - states_in @ A.T - b
    return A, b, resid.norm(dim=-1).max().item()


def fit_layer_models(traces, pos: int = -1, lam: float = 1e-2):
    """traces: list[Trace] (benign). Returns per-layer (A_l, b_l, rho_l)."""
    X = torch.stack([t.at_position(pos) for t in traces])      # (N, L+1, d)
    models = []
    for l in range(X.shape[1] - 1):
        models.append(fit_ridge(X[:, l], X[:, l + 1], lam))
    return models


def jacobian_at(model_hf, tokenizer, prompt: str, layer: int, device="cuda",
                pos: int = -1) -> torch.Tensor:
    """A_l = d x_{l+1} / d x_l at the nominal trajectory via autograd.

    Note: O(d) backward passes — use on small models / selected layers only.
    """
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model_hf(ids, output_hidden_states=True)
    x_l = out.hidden_states[layer][0, pos].detach()

    blocks_fn = _single_block_fn(model_hf, layer)
    return torch.autograd.functional.jacobian(blocks_fn, x_l.unsqueeze(0)).squeeze()


def _single_block_fn(model_hf, layer):
    """Best-effort callable x_l -> x_{l+1} through block `layer` alone.
    Position/attention context is frozen at the nominal pass; adequate for
    last-token local linearization. Architecture-specific edge cases should
    be handled in experiments, not here.
    """
    from .extraction import get_blocks
    blk = get_blocks(model_hf)[layer]

    def fn(x):
        out = blk(x.unsqueeze(0))
        h = out[0] if isinstance(out, tuple) else out
        return h.squeeze(0)
    return fn

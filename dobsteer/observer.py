"""Disturbance observer and minimal-intervention controller.

Layer-depth DOB:   d_hat_l = Q * (x_{l+1} - A_l x_l - b_l)
Cancellation:      u_l     = -P_E d_hat_l   (projected onto disturbance subspace)
Internal model:    z_{t+1} = z_t + K (d_hat_t - z_t)   (token-axis integrator,
                   feed-forward cancellation of persistent injections)
"""
from __future__ import annotations

import torch


class LayerDOB:
    """Per-layer disturbance estimate from one-step surrogate mismatch."""

    def __init__(self, layer_models, V_subspace=None, q: float = 1.0):
        """layer_models: list of (A_l, b_l, rho_l) from surrogate.fit_layer_models.
        V_subspace: optional dict layer -> (k, d) basis of disturbance subspace
        (from disturbance.low_rank_analysis); restricting to it rejects
        surrogate error outside the injection directions.
        q in (0, 1]: scalar Q-filter gain (bandwidth).
        """
        self.models = layer_models
        self.V = V_subspace or {}
        self.q = q

    def estimate(self, x_traj: torch.Tensor) -> torch.Tensor:
        """x_traj: (L+1, d) depth trajectory. Returns d_hat: (L, d)."""
        d_hats = []
        for l, (A, b, _rho) in enumerate(self.models):
            raw = x_traj[l + 1] - x_traj[l] @ A.T - b
            if l in self.V:
                V = self.V[l]                      # (k, d)
                raw = (raw @ V.T) @ V              # project
            d_hats.append(self.q * raw)
        return torch.stack(d_hats)

    def detection_score(self, x_traj: torch.Tensor, layers=None) -> float:
        """Scalar injection score: mean ||d_hat_l|| over selected layers.
        Phase 1 uses this for the ROC vs. benign prompts.
        """
        d = self.estimate(x_traj)
        idx = layers if layers is not None else range(d.shape[0])
        return torch.stack([d[l].norm() for l in idx]).mean().item()


class DOBController:
    """Callable for extraction.capture_trace_with_intervention.

    Maintains depth-wise state during a forward pass; cancels the disturbance
    estimated at layer l-1 at layer l (one-step-delayed cancellation, the
    causal version). Includes a token-axis internal model for persistent
    disturbances across generation steps.
    """

    def __init__(self, dob: LayerDOB, gain: float = 1.0, imp_k: float = 0.2,
                 deadband: float = 0.0):
        self.dob = dob
        self.gain = gain
        self.imp_k = imp_k
        self.deadband = deadband          # no intervention if ||d_hat|| below
        self._prev_x = None               # x_l from previous hook call
        self._z = {}                      # layer -> internal-model state

    def new_token(self):
        self._prev_x = None               # reset depth recursion each token

    def __call__(self, layer_idx: int, hidden: torch.Tensor):
        x = hidden[0, -1].float().cpu()   # last-position state
        u = None
        if self._prev_x is not None and layer_idx - 1 < len(self.dob.models):
            A, b, _ = self.dob.models[layer_idx - 1]
            d_hat = self.dob.q * (x - self._prev_x @ A.T - b)
            if layer_idx - 1 in self.dob.V:
                V = self.dob.V[layer_idx - 1]
                d_hat = (d_hat @ V.T) @ V
            z = self._z.get(layer_idx, torch.zeros_like(d_hat))
            z = z + self.imp_k * (d_hat - z)
            self._z[layer_idx] = z
            d_total = d_hat + z
            if d_total.norm() > self.deadband:
                u = -self.gain * d_total
        self._prev_x = x
        if u is None:
            return None
        delta = torch.zeros_like(hidden)
        delta[0, -1] = u.to(hidden.dtype)
        return delta

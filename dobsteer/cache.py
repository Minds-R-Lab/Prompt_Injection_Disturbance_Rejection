"""Disk cache for residual-stream traces, keyed by (model, prompt).

Trace extraction (a forward pass per prompt) is the dominant cost of the
model x prompt sweep. Caching the per-layer last-token AND content-pooled
features to disk means a prompt is encoded once per model and reused across
phases, seeds, and threshold settings.

We cache compact float16 tensors, not full traces:
  feats[l] = layer update u_l = x_{l+1}-x_l, at all positions (L, T, d) fp16.
Keyed by sha1(model + chat-formatted text). Use get_or_compute().
"""
from __future__ import annotations

import hashlib
import os
import torch

from .extraction import capture_trace

CACHE_DIR = os.environ.get("DOBSTEER_CACHE", os.path.expanduser("~/.dobsteer_cache"))


def _key(model_id, text):
    h = hashlib.sha1((model_id + "\x00" + text).encode("utf-8")).hexdigest()
    return h


def _path(model_id, text):
    safe = model_id.replace("/", "_")
    d = os.path.join(CACHE_DIR, safe)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, _key(model_id, text) + ".pt")


@torch.no_grad()
def updates(model, tok, model_id, text, device, use_cache=True):
    """Return layer-update tensor (L, T, d) fp16 for `text` (already a final
    string, e.g. chat-formatted). Cached to disk per (model_id, text)."""
    p = _path(model_id, text)
    if use_cache and os.path.exists(p):
        try:
            return torch.load(p, map_location="cpu")
        except Exception:
            pass
    tr = capture_trace(model, tok, text, device)
    x = torch.stack(tr.states)            # (L+1, T, d)
    u = (x[1:] - x[:-1]).half().cpu()     # (L, T, d) fp16
    if use_cache:
        tmp = p + ".tmp"
        torch.save(u, tmp); os.replace(tmp, p)
    return u


def clear(model_id=None):
    import shutil
    d = CACHE_DIR if model_id is None else os.path.join(CACHE_DIR, model_id.replace("/", "_"))
    if os.path.isdir(d):
        shutil.rmtree(d)


def stats():
    if not os.path.isdir(CACHE_DIR):
        return {"dir": CACHE_DIR, "models": {}, "total_files": 0}
    out = {}
    for m in os.listdir(CACHE_DIR):
        md = os.path.join(CACHE_DIR, m)
        if os.path.isdir(md):
            out[m] = len([f for f in os.listdir(md) if f.endswith(".pt")])
    return {"dir": CACHE_DIR, "models": out, "total_files": sum(out.values())}

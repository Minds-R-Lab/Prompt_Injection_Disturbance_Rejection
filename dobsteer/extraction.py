"""Residual-stream extraction utilities (HuggingFace transformers).

State definition: x_l = residual stream at the output of block l, at a chosen
token position (default: last token). Works with GPT-2, Pythia (GPTNeoX),
Llama, Qwen2 — anything exposing model.transformer.h / model.model.layers.
"""
from __future__ import annotations

import torch
from dataclasses import dataclass


def get_blocks(model):
    """Return the list of transformer blocks for common architectures."""
    for path in ("transformer.h", "model.layers", "gpt_neox.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"Unsupported architecture: {type(model).__name__}")


@dataclass
class Trace:
    """Residual stream trajectory: states[l] has shape (n_positions, d_model).

    states[0] is the embedding output (pre-block-0); states[l+1] is the output
    of block l. Length = n_layers + 1.
    """
    states: list  # list[torch.Tensor]
    input_ids: torch.Tensor

    def at_position(self, pos: int = -1) -> torch.Tensor:
        """(L+1, d_model) trajectory across depth at one token position."""
        return torch.stack([s[pos] for s in self.states])


@torch.no_grad()
def capture_trace(model, tokenizer, prompt: str, device="cuda") -> Trace:
    """Run a forward pass and capture the residual stream after every block."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model(ids, output_hidden_states=True)
    # hidden_states: tuple of (1, T, d) tensors, length L+1
    states = [h[0].float().cpu() for h in out.hidden_states]
    return Trace(states=states, input_ids=ids.cpu())


@torch.no_grad()
def capture_trace_with_intervention(model, tokenizer, prompt: str,
                                    controller, device="cuda") -> Trace:
    """Forward pass applying controller(layer_idx, hidden) -> delta to add
    to the residual stream at each block output. controller may return None.
    """
    blocks = get_blocks(model)
    captured = []
    handles = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            delta = controller(layer_idx, h)
            if delta is not None:
                h = h + delta.to(h.dtype).to(h.device)
            captured.append((layer_idx, h[0].float().cpu()))
            if isinstance(output, tuple):
                return (h,) + tuple(output[1:])
            return h
        return hook

    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))
    try:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        out = model(ids, output_hidden_states=True)
        emb = out.hidden_states[0][0].float().cpu()
    finally:
        for hd in handles:
            hd.remove()

    captured.sort(key=lambda t: t[0])
    return Trace(states=[emb] + [h for _, h in captured],
                 input_ids=ids.cpu())

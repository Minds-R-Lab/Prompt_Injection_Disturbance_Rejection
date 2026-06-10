# Prompt Injection as Disturbance — DOB-Steer

Disturbance-observer control of LLM residual streams: model adversarial
context as an unmeasured input disturbance, estimate it online, and cancel
only what is estimated (minimal intervention). See `PROPOSAL.md` for the full
research plan, theory targets, and related-work positioning.

## Setup

```
pip install torch transformers
```

## Run Phase 0 (disturbance characterization)

```
python -m dobsteer.run_phase0 --model gpt2-large
python -m dobsteer.run_phase0 --model Qwen/Qwen2.5-1.5B --device cuda
python -m dobsteer.run_phase0 --model meta-llama/Llama-3.2-3B   # needs HF login
```

60 benign base prompts (`dobsteer/prompts.py`) crossed with 5 injection
strings. Prints verdicts for H1a (low-rank), H1b (additivity), H1c
(persistence).

- H1a reports **effective rank** and flags the test INCONCLUSIVE when
  `n_pairs <= 2*k`, so the rank-limited artifact (energy=1.0 with few prompts)
  is no longer mistaken for genuine low-rank structure.
- H1c uses **relative** persistence `||Delta_l|| / ||f_l^benign||`, removing the
  depth-scaling confound from residual-stream norm growth.

## Run Phase 1 (detection ROC — the go/no-go gate)

```
python -m dobsteer.run_phase1 --model gpt2-large --device cuda
```

Fits benign surrogates on a train split, then reports ROC AUC of the DOB
residual score on held-out benign vs. injected prompts. **AUC > 0.9 supports
H2** and greenlights the closed-loop phase.

## Modules

- `extraction.py` — residual-stream capture, with-intervention forward pass
- `disturbance.py` — paired-prompt delta analysis (H1 tests)
- `surrogate.py` — locally-linear layer models (ridge / Jacobian)
- `observer.py` — `LayerDOB` (estimation + detection), `DOBController`
  (closed-loop cancellation with internal model + deadband)
- `prompts.py` — benign prompt set + injection strings
- `run_phase0.py`, `run_phase1.py` — experiment entry points

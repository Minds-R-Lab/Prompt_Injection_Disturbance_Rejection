# dobsteer

Phase 0/1 scaffold for the DOB-Steer project (see ../PROPOSAL.md).

## Setup

```
pip install torch transformers
```

## Run Phase 0 (disturbance characterization)

```
python -m dobsteer.run_phase0 --model gpt2-large
```

Prints PASS/FAIL verdicts for hypotheses H1a (low-rank), H1b (additivity),
H1c (persistence). Extend `BASE_PROMPTS` to >=50 prompts and add real
injection benchmarks before drawing conclusions.

## Modules

- `extraction.py` — residual-stream capture, with-intervention forward pass
- `disturbance.py` — paired-prompt delta analysis (H1 tests)
- `surrogate.py` — locally-linear layer models (ridge / Jacobian)
- `observer.py` — `LayerDOB` (estimation + detection score), `DOBController`
  (closed-loop cancellation with internal model + deadband)

## Next (Phase 1)

1. Fit surrogates on benign traces: `surrogate.fit_layer_models`
2. Detection ROC: `LayerDOB.detection_score` on injected vs. benign prompts
3. Closed loop: pass `DOBController` to
   `extraction.capture_trace_with_intervention`; call `controller.new_token()`
   between generation steps.

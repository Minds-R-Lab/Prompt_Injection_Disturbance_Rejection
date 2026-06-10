# Experimental log

Setup: 61 benign prompts, 5 injection strings, LOIO where noted, train/test
split 30/31. Negatives from v4 on include 155 benign-suffixed decoys.

## Phase 0 — disturbance characterization (PASS, 3 model families)

| Model | energy@k=16 | eff. rank | additivity (mid cos) | rel. persistence |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0.83-0.88 | 8-13 | 0.92-0.96 | 1.07-1.26 |
| Qwen2.5-7B   | 0.83-0.85 | 10-13 | 0.93-0.95 | 1.19-1.24 |
| Pythia-1.4B  | 0.82-0.88 | 7-13 | 0.91-0.96 | 0.91-1.14 |

The injection signature is low-rank, additive across prompts, persistent
across depth. Foundation for everything downstream.

## Phase 1 — detection (PASS at 7B, both placements)

Detector iterations (pooled AUC, LOIO):

| Version | detector | 1.5B app | 7B app | 1.5B mid | 7B mid | note |
|---|---|---|---|---|---|---|
| v1 | DOB residual norm (last tok, 30-sample surrogate) | 0.67 | 0.55 | - | - | surrogate overfit |
| v2 | + all-position fit, z-score, SVD subspace | 0.77 | 0.76 | - | - | subspace = benign variance |
| v3 | matched filter, last token | 1.00 | 1.00 | - | - | append confound! |
| v4 | + suffix decoy negatives, mid-context eval | 0.84 | 0.98 | 0.40 | 0.44 | last-token blind mid-ctx |
| v5 | position-scanning matched filter | 0.92 | **0.998** | 0.82 | **0.978** | PASS |

Side findings: detection quality scales with model size; raw layer updates
beat surrogate residuals (the averaged linear surrogate adds noise, not
signal); norm-based anomaly detection plateaus ~0.7 regardless.

## Phase 2 — closed-loop cancellation (in progress)

`run_phase2.py`: deadband cancellation (subtract excess projection above
tau sigma in mid layers) on an instruct model. Metrics: ASR with/without
defense, benign next-token KL, intervention energy. Target H3: >=50% ASR
reduction at negligible benign KL.

## Known limitations / next hardening steps

- String-marker ASR heuristics -> replace with LLM judge.
- Paraphrased/translated injections (semantic vs lexical generalization).
- Adaptive attacker (knows v, optimizes around the deadband) -- future work.
- Benchmark suites (GCG suffixes, indirect injection sets) for the paper.

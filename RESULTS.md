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

## Phase 3 — detect-then-cancel (gated), CONFIRMED

Hardened gate (content-aware: chat-template scaffold excluded from the
subspace fit and position scan; fit at last CONTENT token). Qwen2.5-7B-Instruct,
k=32, alpha=3, all-layer cancellation, per-position deadband tau=1.

| metric | value |
|---|---|
| gate AUC (instruct, test) | 0.997 |
| ASR (base -> gated) | 0.91 -> 0.19 (79% reduction) at target-FPR 0 |
| benign FPR | 0.00 |
| benign KL | 0.0000 |

Key fix: v1/v2 gated on an all-layer mean direction fit at the chat-template's
last (scaffold) token -> constant score, AUC 0.55. Excluding the fixed template
span and fitting/scanning only user-content positions restored AUC 0.997.

Residual: at strict target-FPR 0 the threshold sits above the weakest
injections (TPR 0.79), leaving 0.19 ASR. Relaxing to FPR<=0.05 is expected to
push TPR->1, ASR->~0.02 at near-zero benign KL -- traced by run_sweep.

## Foundation battery: run_all.sh
Seed stability (5 seeds) + target-FPR ROC (run_sweep), Phase-2 Pareto, and
Phase-1 across 3 families. All scaffold-correct (content-aware fitting ported
into Phase 2). Run before scaling.

## SCALED detection: LOFO + semantic generalization (Qwen2.5 ladder, 3 seeds)

benign=400 (Alpaca/Dolly), 6 hand-built injection families + wild_deepset
(external benchmark), 3 placements (append/mid/document). LOFO = leave-one-
attack-FAMILY-out. SEM = fit on canonical families, test on paraphrase+translation.

| Model | params | LOFO pooled | SEM | wild_deepset (LOFO) |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.5B | 0.888 +/- 0.010 | 0.887 +/- 0.009 | 0.808 |
| Qwen2.5-1.5B-Instruct | 1.5B | 0.918 +/- 0.004 | 0.951 +/- 0.005 | 0.881 |
| Qwen2.5-3B-Instruct   | 3B   | 0.965 +/- 0.003 | 0.980 +/- 0.001 | 0.947 |
| Qwen2.5-7B-Instruct   | 7B   | 0.979 +/- 0.003 | 0.992 +/- 0.002 | 0.966 |
| Qwen2.5-14B-Instruct  | 14B  | 0.979 +/- 0.003 | 0.997 +/- 0.001 | 0.969 |

Monotone scale trend on all three columns. Per-family LOFO (7B): ignore 1.00,
system_spoof 1.00, exfiltration 0.998, dev_imp 0.996, payload 0.999,
roleplay 0.996, wild_deepset 0.966.

## SCALED control: detect-then-cancel ROC (Qwen2.5-7B-Instruct, 3 seeds, held-out calib)

Gate AUC 0.995 +/- 0.001. Base ASR 0.91. k=32, alpha=3, tau=1.

| target FPR | test FPR | TPR | ASR | benign KL |
|---|---|---|---|---|
| 0.00 | 0.00 | 0.81 | 0.19 | 0.000 |
| 0.01 | 0.02 | 0.87 | 0.15 | 0.18 |
| 0.02 | 0.02 | 0.94 | 0.08 | 0.18 |
| 0.05 | 0.03 | 0.96 | 0.06 | 0.37 |
| 0.10 | 0.09 | 0.97 | 0.05 | 1.28 |

Two operating points: zero-benign-cost (FPR 0: ASR 0.91->0.19) and
stronger-suppression (FPR 0.02: ASR 0.91->0.08, 91% reduction). KL rises with
FPR because falsely-gated benign prompts receive the aggressive alpha=3
cancellation -> motivates gentler cancellation on gated prompts (future work).

NOTE: Llama models gated (401) -> need `huggingface-cli login`. 32B fp16 +
70B/72B 4-bit ready via run_big.sh.

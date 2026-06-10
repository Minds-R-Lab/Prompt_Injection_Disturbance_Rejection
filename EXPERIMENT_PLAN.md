# Scale-up plan for publication

Status: pilot results confirmed on Qwen2.5 (1.5B/7B) + Pythia-1.4B.
Phase 0 (structure), Phase 1 (detection AUC up to 0.998), Phase 2 (full
controllability; ASR 0.91->0.00), Phase 3 (detect-then-cancel: 96% ASR
reduction, TPR 1.0, FPR 0.05 @ thr=1.0). This document scales each axis to
publication strength. Execute only after the pilot operating point is locked.

## 1. Why scale (reviewer-facing weaknesses of the pilot)
- Benign set is 61 hand-written prompts (30 train / 20-31 test): AUC/ASR
  estimates have wide CIs; threshold calibration is noisy (the FPR 0.05 vs
  0.15 gap between runs is sample-size noise).
- 5 injection strings that share surface vocabulary: LOIO shows lexical, not
  semantic, generalization.
- 2 model families, 3 sizes: not enough to claim the phenomenon is general or
  to establish the scale trend predicted by the ISS bound.
- No standard benchmark, no LLM-judge ASR, no human-readable utility metric.

## 2. Datasets

### 2.1 Benign instructions (target: 1,000-2,000 prompts)
Sample, deduplicate, and length-filter from established instruction corpora so
the distribution is realistic and not author-biased:
- Alpaca (52k) / Dolly-15k / No Robots / LMSYS-Chat-1M (first user turns).
- Stratify by task type (QA, summarization, coding, math, creative, multi-turn)
  and by length, so detection/control are reported per-stratum.
- Hold out a calibration split for the gate threshold separate from test.

### 2.2 Injection attacks (target: 30-50+ strings, multiple families)
Pull from established prompt-injection benchmarks rather than hand-writing:
- Direct override / instruction-ignore corpora (e.g. deepset prompt-injections,
  HackAPrompt, Tensor Trust).
- Indirect / tool-use injection suites (e.g. BIPIA, InjecAgent, AgentDojo) for
  the realistic agent threat model.
- Optimized adversarial suffixes (GCG-style) as a distinct, harder family.
Group attacks into families and evaluate leave-one-FAMILY-out (LOFO), a much
stronger generalization claim than leave-one-string-out.

### 2.3 Semantic-generalization probe (required)
For each injection, generate paraphrases and translations (same intent, no
shared wording) via an LLM. Train the filter on originals, test on
paraphrase/translation. If detection holds, the filter captures a semantic
"override" direction, not phrasing -- this is the experiment that converts the
result from "pattern match" to "mechanism".

### 2.4 Placement and embedding
Appended, mid-context, and realistic document-embedded (inside a retrieved
passage / email / web snippet the model is asked to process).

## 3. Models (target: >=8 checkpoints, >=3 families, >=1 clean scaling ladder)
- Scaling ladder in ONE family for the ISS/scale trend: Pythia
  {410M,1.4B,2.8B,6.9B,12B} or Qwen2.5 {0.5,1.5,3,7,14,32B}.
- Cross-family generality: Llama-3.x {1B,3B,8B}, Gemma-2 {2,9,27B},
  Mistral-7B, Phi.
- Base vs instruct pairs (control needs instruct; detection works on both).
- Report detection AUC and Phase-3 ASR/KL as a function of parameter count;
  the predicted monotone improvement with scale is a headline figure.
Compute note: <=14B fits one H100 in fp16; 32-72B needs 2-4 H100 or 4-bit.

## 4. Metrics and rigor
- Detection: ROC-AUC + AUPRC, with 95% CIs via bootstrap over prompts; LOFO.
- Control: attack success by an LLM judge (not string markers) + the original
  task utility (AlpacaEval/MT-Bench-style win rate, IFEval, a small MMLU slice)
  to show benign capability is preserved.
- Benign cost: KL(base||defended) AND task-utility delta on the benign set.
- Operating curves: detection ROC; Phase-2 ASR-vs-KL Pareto (alpha sweep);
  Phase-3 ASR-vs-FPR as the gate threshold varies.
- Seeds: >=3 train/test resamples; report mean +/- std for every headline number.

## 5. Ablations (each as a figure/panel)
- Subspace rank k; intervention layer band (early/mid/late/all); deadband tau;
  gain alpha; detector statistic (norm vs mean-dir vs subspace; last vs scan;
  layer band -- the pilot showed all-layer mean underperforms mid-layer scan).
- Surrogate fit (ridge lambda, #positions) vs Jacobian.
- Internal-model term on/off vs repeated/persistent injections (validates the
  IMP theorem).

## 6. Adaptive attacker (strengthens, or scopes, the safety claim)
- White-box attacker that knows v and optimizes a suffix orthogonal to it or
  inside the deadband; report robustness degradation. Frame as a minimax game;
  even a partial result here greatly raises the paper's ceiling.

## 7. Engineering needed (small)
- prompts.py -> a dataset loader (HF datasets) with caching of traces to disk
  so the 1-2k-prompt x 8-model sweep is tractable; trace extraction is the cost,
  amortize it.
- An LLM-judge harness for ASR + a utility-eval harness.
- A config/sweep runner (hydra or a simple grid) writing tidy CSVs; all figures
  regenerated from CSVs.
- Gate the Phase-3 detector on the mid-layer scan statistic (cheap fix, widens
  the FPR/TPR gap seen in the pilot).

## 8. Suggested order / rough effort
1. Dataset loaders + trace caching + mid-layer gate fix.            (~3-4 days)
2. Re-run Phases 0-3 on the scaling ladder, with CIs.              (~1 wk GPU)
3. LLM-judge + utility harness; Phase-3 ASR-vs-FPR + utility.       (~1 wk)
4. Semantic-generalization (paraphrase/translation) + LOFO.        (~3-4 days)
5. Cross-family models + scale-trend figure.                       (~1 wk GPU)
6. Adaptive-attacker study.                                        (~1-2 wk)
7. Writing: fold all into the paper; tighten theory proofs.        (ongoing)

## 9. Go/no-go gates
- After step 2: detection AUC must hold (>0.95 LOFO) at >=3 sizes, and the
  scale trend must be visible. If not, narrow the model claim.
- After step 4: if semantic generalization fails, reframe as lexical detection
  + the controllability result (still publishable, narrower).

# Referee response: point -> action -> status

Triage of the referee review. "Built" = code committed; "running" = launched
on the cluster; "paper" = addressed in the draft text.

## Weaknesses
- **W1 attack set too small / additivity may be circular.**
  -> 36 injections in 6 families + paraphrase/translation held-outs
  (`data.py`); leave-one-FAMILY-out + semantic generalization
  (`run_scale_detect.py`). Abstract now flags n=5/n=61 pilot scope. [built; paper]
- **W2 non-adaptive threat model.** -> white-box soft-suffix attack that
  minimizes the matched-filter score; reports gate TPR vs evasion weight
  (`run_adaptive.py`). [built]
- **W3 DOB loosely coupled; Eq.6 unevaluated.** -> candid paragraph that the
  deployed detector IS the matched filter and the surrogate underperforms;
  internal-model term evaluated via a persistent-injection experiment
  (queued). [paper; eval queued]
- **W4 zero-FP not significant.** -> rule-of-three caveat added; scaled run
  reports FPR with CIs on a large benign pool. [paper; built]
- **W5 evaluation gaps.**
  (a) LLM-judge ASR + task-retention (`judge.py`). [built]
  (b) external baselines: perplexity filter, linear probe (`run_baselines.py`). [built]
  (c) standard benchmark: deepset prompt-injections wired as a 'wild' family
      (`data.py`); indirect-injection suites queued. [partial]
  (d) compute overhead: forward-pass + scan ms/prompt (`run_baselines.py`). [built]
- **W6 chat-template brittleness.** -> content-aware span handling generalized
  (auto-detected per template); cross-template/multi-turn transfer queued. [partial]

## Minor
- (a) ASR 0.91 vs 0.88 reconciled (seed-mean 0.88->0.18 used). [paper]
- (b) rho: text now notes mfr-scan<mf-scan implies the nominal model adds error,
  not signal, at these scales. [paper]
- (c) Prop 2 framed as standard ISS; proof deferred (kept, flagged). [paper]
- (d) scale trend explicitly hedged as 2 sizes here -> full ladder in extended. [paper]

## Questions
- **Q1 fit on some families, test on held-out family.** = `run_scale_detect`
  LOFO -- the exact experiment. [built; running]
- **Q2 adaptive suffix vs matched-filter score.** = `run_adaptive`. [built]
- **Q3 overhead.** = `run_baselines` timing. [built]
- **Q4 what does the cancelled generation look like.** = `judge.py`
  task_retention + saved generations. [built]
- **Q5 injection vs imperativeness.** = benign hard negatives that legitimately
  contain instructions (`data.benign_hard_negatives`); hard-neg FPR reported in
  `run_baselines`; discussed in the paper. [built; paper]

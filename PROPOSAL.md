# Prompt Injection as Disturbance: Disturbance-Observer Steering of LLM Residual Streams

**Working title:** *DOB-Steer: Rejecting Adversarial Context in Language Models via Disturbance-Observer Control*
**Target venue:** ICLR 2027 (abstract deadline ~Sept 2026), backup NeurIPS 2027 / ICML 2027
**Compute:** single GPU (models up to ~3B params)

---

## 1. One-line pitch

Adversarial or unwanted context (prompt injections, persona drift, jailbreak suffixes) acts on a transformer's residual stream like an **unmeasured input disturbance** on a dynamical system. We estimate that disturbance online with a **disturbance observer (DOB)** and cancel it with minimal intervention — with input-to-state stability (ISS) guarantees — instead of constantly pushing activations toward a fixed setpoint as all existing steering methods do.

## 2. Why this gap is real (June 2026 landscape)

The control-theory-for-steering space got crowded fast, so positioning matters. What is **already taken**:

| Idea | Paper | What it does |
|---|---|---|
| Steering = P-control | classical ActAdd/CAA | open-loop fixed vector |
| PID feedback steering | [arXiv:2510.04309](https://arxiv.org/abs/2510.04309) | PID over layer-depth error dynamics |
| LQR on layer-wise Jacobians | [arXiv:2604.19018](https://arxiv.org/abs/2604.19018) | locally-linear LTV model, LQR setpoint tracking + error bounds |
| CBF safety filters (token level) | [arXiv:2408.15625](https://arxiv.org/abs/2408.15625), [arXiv:2511.03121](https://arxiv.org/abs/2511.03121) | filter predicted tokens for safe-set invariance |
| Learned barrier steering | [arXiv:2602.20102](https://arxiv.org/abs/2602.20102) | barrier on unsafe response trajectories |
| Guardrails as control | [arXiv:2510.13727](https://arxiv.org/pdf/2510.13727) | recovery-oriented guardrail framework |
| Controllability of LM states | [arXiv:2508.01892](https://arxiv.org/abs/2508.01892), [arXiv:2511.17970](https://arxiv.org/pdf/2511.17970) | controllability emergence / SSM Gramians |

What **nobody has done** (verified by search, June 2026):

1. Model adversarial context as an **exogenous disturbance signal** `d` entering the residual-stream dynamics — rather than as a bad initial condition or a bad setpoint.
2. **Estimate** `d` online (disturbance observer / unknown-input observer) and **cancel only what is estimated** — minimal intervention by construction, no intervention when the context is benign.
3. **Internal-model principle** for persistent disturbances: an injection repeated in context (or a long jailbreak suffix) is a *persistent* disturbance; IMP says exact rejection requires a model of the disturbance dynamics inside the controller. No steering paper uses this.
4. ISS guarantees of the closed loop under locally-linear model mismatch (the local-linearity premise is now empirically de-risked by arXiv:2604.19018 — we build on it, not against it).

Setpoint methods (PID/LQR) **always** intervene and need a known target direction. A DOB needs **no target**: nominal behavior is the reference, and the controller only removes the estimated anomalous input. This is a fundamentally different — and for safety, more natural — control objective: *disturbance rejection vs. reference tracking*.

## 3. Problem formulation

Fix a transformer with `L` layers. Let `x_l ∈ R^n` be the residual-stream state at layer `l` (at the last token position, or pooled). Layer dynamics:

```
x_{l+1} = f_l(x_l) + B_l u_l           (nominal, benign context)
x_{l+1} = f_l(x_l) + B_l u_l + E_l d_l (under adversarial context)
```

- `u_l` = our steering input (additive activation edit), `B_l` = I or a low-rank basis.
- `d_l` = the disturbance: the component of the layer update attributable to the adversarial span of the context (via attention from the current position into the injected tokens).
- Locally linearize: `x_{l+1} ≈ A_l x_l + B_l u_l + E_l d_l + r_l`, with residual `‖r_l‖ ≤ ρ` (bound measured empirically; arXiv:2604.19018 shows this is small in practice).

**Key modeling claim (testable):** for a benign prompt `P` and the same prompt with injection `P+I`, the difference of layer updates `Δ_l = f_l^{P+I}(x) − f_l^P(x)` is (a) low-rank, (b) approximately additive, (c) persistent across layers/tokens. This is Experiment 0 and is itself a publishable finding.

### Disturbance observer

Classical DOB adapted to layer-depth (and token-time) dynamics:

```
d̂_l = Q_l ( x_{l+1} − A_l x_l − B_l u_l )      (estimate from one-step model mismatch)
u_l  = −B_l† E_l d̂_l                            (cancellation, projected on steerable subspace)
```

with `Q_l` a filter gain (the "Q-filter" of DOB design) trading estimation speed vs. noise/model-error amplification. For persistent disturbances, augment with an internal model (integrator along token axis): `ẑ_{t+1} = ẑ_t + K (d̂_t − ẑ_t)` and cancel `ẑ_t` feed-forward at every future token — this is the IMP component and is what makes the defense *cheap at generation time*.

### Theory targets (the control contributions)

- **T1 (Identifiability):** conditions on `(A_l, E_l)` (unknown-input observability) under which `d_l` is reconstructible from residual-stream observations; characterize the reconstructible disturbance subspace via Gramian-type quantities.
- **T2 (ISS):** closed-loop semantic error `e_t` (projection of state onto a probe direction, e.g. "obey-the-injection" direction) satisfies `‖e_t‖ ≤ β(‖e_0‖, t) + γ(ρ + d̃)` where `d̃` is the estimation error — i.e., graceful degradation under model mismatch.
- **T3 (Minimal intervention):** DOB-cancellation is the minimum-norm input achieving nominal-equivalence in the locally-linear model; corollary: zero intervention when `d̂ = 0`. Quantify side-effect bound: KL(base ‖ controlled) ≤ c‖d̂‖².
- **T4 (IMP for LLMs):** persistent injections require an internal model; show a memoryless (per-token) defense has nonzero steady-state semantic error, while the IMP-augmented controller drives it to the `ρ`-ball. This gives a *theorem explaining why one-shot defenses fail against repeated injections*.

## 4. Experimental plan (single GPU)

**Models:** GPT-2-large (verification), Pythia-1.4B, Llama-3.2-1B/3B, Qwen2.5-1.5B — at least 3 families to show generality.

**Phase 0 — Validate the disturbance model (2–3 weeks)**
Paired prompts (benign / benign+injection). Measure `Δ_l`: rank, additivity (does `Δ` transfer across base prompts?), persistence across token positions. Success criterion: top-k subspace (k ≤ 16) captures >80% of `‖Δ‖` across a prompt distribution.

**Phase 1 — Offline DOB (3–4 weeks)**
Fit `A_l` (Jacobian or ridge regression on rollouts), build the observer, verify `d̂` detects injections: ROC of `‖d̂‖` as an *injection detector* vs. perplexity-based and classifier baselines. (Detection alone is a result: "DOB as a white-box prompt-injection detector".)

**Phase 2 — Closed-loop rejection (4–6 weeks)**
Benchmarks: prompt-injection suites (e.g., agent-style indirect injection sets, jailbreak suffix sets like GCG-style), persona-drift over long generation.
Metrics: attack success rate ↓, task utility on benign inputs (MMLU subset, MT-Bench-style scoring) →, fluency/perplexity →, KL to base model ↓, intervention energy `Σ‖u‖²` ↓.
Baselines: no defense, system-prompt defense, ActAdd/CAA static steering, PID steering [2510.04309], LQR steering [2604.19018], token-level CBF [2408.15625].
**Headline plot:** utility-vs-robustness Pareto; DOB should dominate setpoint methods on benign-input utility (because it does nothing when `d̂≈0`) at comparable attack-success reduction.

**Phase 3 — Ablations & theory validation**
Q-filter bandwidth sweep (estimation vs. noise tradeoff — a *control-theoretic ablation no ML paper has*), with/without internal model vs. repeated injections (validates T4), rank of `E`, layer selection, transfer of observer across prompts/tasks.

**Falsifiable hypotheses:** H1 injection effect is approx. additive & low-rank; H2 `‖d̂‖` separates injected vs. benign (AUC > 0.9); H3 closed loop cuts attack success ≥50% at <2% benign utility loss; H4 memoryless defense degrades vs. repeated injections, IMP does not.

## 5. Paper outline

1. Intro: steering = tracking; safety = disturbance rejection; these are different control problems.
2. Related work (table above).
3. LLM-as-LTV-system with unmeasured input; identifiability (T1).
4. DOB-Steer + IMP augmentation; ISS and minimal-intervention results (T2–T4).
5. Experiments (Phases 0–3).
6. Discussion: extension to agents (tool-call disturbances), limitations (adaptive attackers who model the observer — note this as future game-theoretic work).

## 6. Risks and mitigations

- **R1: injection effect not additive/low-rank.** → Pivot: nonlinear unknown-input observer on a learned latent; or publish the *negative result + detector* (Phase 1 stands alone as a workshop/short paper).
- **R2: scooped on DOB framing.** → The IMP/persistent-disturbance theory and the Q-filter robustness analysis are second and third lines of defense; also the detector application.
- **R3: utility damage.** → DOB's minimal-intervention property is precisely the mitigation; report KL-to-base.
- **R4: adaptive attacks break it.** → Scope claims to non-adaptive threat model (standard for first paper in a line); discuss honestly.

## 7. Two-paper strategy

- **Paper A (ML conference):** DOB-Steer empirical + ISS bounds (this proposal).
- **Paper B (Automatica/TAC/L-CSS):** unknown-input observability of LTV systems with empirically-identified structure, Gramian characterization of reconstructible disturbance subspaces, with LLM as the motivating application.

## 8. Immediate next steps

1. Run Phase 0 on GPT-2-large with 50 paired prompts (code scaffold in `dobsteer/`).
2. If H1 holds, fit observers and produce the detection ROC (Phase 1) — that's the go/no-go gate, ~1 month in.

# Response to Reviewer (second round)

We thank the reviewer for the careful second-round review and the upgraded
recommendation (Accept / minor-to-moderate revision). We were glad the
disturbance-rejection framing, the leave-one-family-out and semantic
generalization, the adaptive-attacker analysis, and the candor in §7 landed as
intended. Below we respond point-by-point to the five remaining items; the two
flagged as gating an unqualified accept (LLM-judged ASR and the internal-model
term) are addressed with new experiments.

## W1 — ASR is string-marker based; does cancellation preserve utility? (also Q4)

Addressed with a new LLM-judged, task-execution evaluation (new §, Table:
`run_agentic.py`). For indirect-injection inputs (a benign task plus a document
with a mid-passage injection) an independent judge model scores each output on
two axes: *hijacked* (did it obey the injection?) and *task\_done* (did it still
complete the user's task?). We report three conditions:

- no defense (attack present),
- detect-then-cancel,
- clean input, defended (no injection).

This directly answers Q4: it shows whether the cancelled output completes the
benign task or merely degrades into marker-free incoherence, and the clean-input
row confirms the defense leaves benign utility untouched. The headline
suppression numbers are re-reported under the judge alongside task-retention, so
they are no longer upper bounds resting on heuristic scoring.

## W2 — The internal-model term (Eq. 6) is unevaluated

Addressed with a new persistent-injection experiment (`run_persist.py`). We use
a *persistent* injection ("append HACKED to the end of every sentence"), which
re-arrives at every sentence of the generation, and compare three controllers
during autoregressive decoding:

- no defense,
- static (memoryless) cancellation $u_t=-g\,\hat d_t$,
- the token-axis internal model $z_{t+1}=z_t+K(\hat d_t-z_t),\ u_t=-g(\hat d_t+z_t)$.

We report marker occurrences in growing token windows (24/48/72/96) plus a
benign-drift check. The intended result — static suppression decaying as the
generation lengthens while the internal model keeps the marker suppressed — makes
the dynamical formulation load-bearing rather than motivational, exactly the bar
the reviewer set in the closing sentence. (Result pending the run; the harness is
in the released code.)

## W3 — No standard agentic benchmark / discrete optimized attack

Addressed on both fronts. (i) A discrete GCG-style optimized attack
(`run_gcg.py`) complements the soft-embedding upper bound: greedy coordinate
gradient crafts an adversarial injection of real tokens that jointly elicits the
attack marker and minimizes the gate score; sweeping the evasion weight traces
the discrete tradeoff. (ii) The agentic task-execution evaluation above
(`run_agentic.py`) reports the resist-AND-complete metric the reviewer asked for,
with an LLM judge scoring both injection-resistance and task completion on
document-embedded indirect injections. We position these honestly as a discrete
optimized attack and a lightweight agentic proxy; full AgentDojo/BIPIA tool-use
suites remain complementary future work.

## W4 — Propositions deferred to a companion paper

Done. Proposition 1 (minimal intervention) is proved in-text (it is short and
self-contained). Proposition 2 (ISS) has been **demoted to a Conjecture** and is
explicitly motivated empirically rather than claimed; we removed the "guarantees
deferred to a companion treatment" framing. The paper now states plainly which
result is proved and which is conjectured.

## W5 — Minor

- **(a) Abstract length.** Trimmed substantially (~40% shorter); it no longer
  enumerates every sub-result.
- **(b) Soatto et al. author list.** Fixed. The correct list is Soatto,
  Tabuada, Chaudhari, and Tian Yu Liu (verified against the arXiv/DBLP record);
  the earlier draft's list was erroneous. (We also re-verified every cited
  reference's title, venue, year, and authors against primary sources, and added
  Arditi et al. 2024 and ARGUS 2025.)
- **(c) CI on the 0.00 FPR.** Added. We now report a one-sided Wilson 95% upper
  bound ($\approx 1\%$ on the 400-prompt pool) rather than a bare $0.00$, in both
  the operating-curve text and the table caption.

## Summary of new material

| Reviewer item | New artifact | What it shows |
|---|---|---|
| W1 / Q4 | `run_agentic.py` (LLM judge) | ASR down with task-completion preserved; clean utility untouched |
| W2 | `run_persist.py` | internal model beats static cancellation on persistent injections |
| W3 | `run_gcg.py` + agentic eval | discrete optimized attack + resist-AND-complete metric |
| W4 | paper edit | Prop 1 proved; Prop 2 → Conjecture |
| W5a/b/c | paper edits | abstract trimmed; citation fixed; Wilson CI on FPR |

We believe these changes close the two gating items and convert the framing from
"elegant motivation" to a tested, load-bearing component.

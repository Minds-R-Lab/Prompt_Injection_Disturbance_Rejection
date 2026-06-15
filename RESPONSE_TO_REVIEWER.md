# Response to Reviewer (third round / camera-ready)

We thank the reviewer for the generous reading and the Accept. The decisive step
was the one the reviewer pressed for in both prior rounds: scoring the defense
with an LLM judge rather than string markers. It overturned our headline
"detect-then-cancel" result, and we rebuilt the paper around that finding.

## The pivot (what changed since draft 2)

- Re-scored the exact detect-then-cancel defense with a behavioral judge
  (hijacked? task completed?). It provides essentially no protection
  (hijacked 0.88->0.96, task 0.12->0.04); the marker metric overstated it ~3-4x.
- The paper no longer claims a defense. New title: "...A Detector, and the
  Localization Barrier to Activation-Space Mitigation."
- Old marker-based defense tables (Tables 6-7 in draft 2) REMOVED; replaced by a
  single LLM-judge table (Table 6 now) including an oracle row.
- Oracle (remove the known injection): hijacked 0.88->0.04, task ->1.00 ->
  isolates faithful localization, not the removal mechanism, as the open problem.
- Detector results unchanged (none depended on the defense): LOFO 0.98, SEM
  0.997, scaling 0.5-14B, Gemma-2 cross-family, beats trained probe, 0.00
  hard-neg FPR, adaptive causality argument.

## Remaining points

1. **Power up Section 7 (the insisted item).** Agreed; this is the camera-ready
   priority. The harness (`run_localize.py`) takes `--n-eval`, `--placement`
   {append,mid,document}, and `--model`, and now reports a 95% Wilson CI on the
   hijacked rate. For camera-ready we repeat Table 6 at n=60, on a second model
   (Qwen2.5-14B-Instruct), and at appended placement, and report CIs.
2. **The 3-4x multiplier.** Restated as an observed ratio on this task set, not a
   constant; the qualitative overstatement is robust, the multiplier is not.
   CIs are now produced by the harness and will accompany the powered-up numbers.
3. **GCG / agentic suite.** `run_gcg.py` (discrete optimized attack) and
   `run_agentic.py` (LLM-judged task-execution) are in the repo; we will fold a
   discrete attack and an agentic-style benchmark into camera-ready so our own
   evaluation meets the behavioral standard we urge on others.
4. **Internal-model term (Eq. 6).** Trimmed; kept minimal and explicitly marked
   not load-bearing (cancellation fails with or without it).
5. Minor: (a) abstract pivot tightened; (b) "(marker ASR 0.18)" parenthetical
   kept in Table 6; (c) clean (no-injection) baseline now a row in Table 6.

## Camera-ready checklist
- [ ] Table 6 at n=60, +Qwen2.5-14B-Instruct, +appended placement, with CIs.
- [ ] One discrete (GCG) attack + one agentic benchmark.
- [x] 3-4x stated as observed ratio + CI support.
- [x] Eq. 6 trimmed.
- [x] Clean baseline as a Table-6 row.

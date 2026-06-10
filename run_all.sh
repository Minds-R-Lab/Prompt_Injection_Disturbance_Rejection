#!/usr/bin/env bash
# Foundation-locking battery. Run on the H100; logs + JSON land in results/.
# Usage: bash run_all.sh
set -e
mkdir -p results
DEV=cuda
PY="python -m"

echo "=== [1/4] Phase 1 detection (seed-robust check across 3 model families) ==="
for M in Qwen/Qwen2.5-7B Qwen/Qwen2.5-1.5B EleutherAI/pythia-1.4b; do
  tag=$(echo $M | tr '/' '_')
  $PY dobsteer.run_phase1 --model $M --device $DEV \
      --out results/phase1_${tag}.pt 2>&1 | tee results/phase1_${tag}.log
done

echo "=== [2/4] Phase 3 gated detect-then-cancel: seed stability + target-FPR ROC ==="
$PY dobsteer.run_sweep --model Qwen/Qwen2.5-7B-Instruct --device $DEV \
    --seeds 0,1,2,3,4 --fprs 0,0.01,0.02,0.05,0.1 --k 32 --alpha 3 --tau 1.0 \
    --out results/sweep_qwen7b_instruct.json 2>&1 | tee results/sweep_qwen7b_instruct.log

echo "=== [3/4] Phase 2 ASR-vs-KL Pareto frontier (always-on cancellation) ==="
$PY dobsteer.run_phase2 --model Qwen/Qwen2.5-7B-Instruct --device $DEV \
    --taus 1,2,3,4,6 --alphas 1,3,8 --k 32 --layers all \
    --out results/phase2_pareto_qwen7b_instruct.json 2>&1 | tee results/phase2_pareto.log

echo "=== [4/4] Phase 3 single-point sanity (zero-FPR operating point) ==="
$PY dobsteer.run_phase3 --model Qwen/Qwen2.5-7B-Instruct --device $DEV \
    --k 32 --alpha 3 --tau 1.0 --target-fpr 0.02 \
    --out results/phase3_point.json 2>&1 | tee results/phase3_point.log

echo "=== DONE. Results in results/ ==="

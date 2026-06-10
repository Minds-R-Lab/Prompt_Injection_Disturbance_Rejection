#!/usr/bin/env bash
# Full scale-up battery for the H100. Detection (cheap, cached) across the
# model ladder + control sweep on the largest instruct model.
# Edit LADDER / N_BENIGN as desired. Logs+JSON -> results_scale/.
set -e
mkdir -p results_scale
DEV=cuda
N_BENIGN=${N_BENIGN:-400}
SEEDS=${SEEDS:-0,1,2}

# Scaling ladder (one family) + cross-family generality. Gated models need a
# HF token (huggingface-cli login); ungated ones run as-is.
LADDER=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
)

echo "=== SCALE DETECTION (LOFO + semantic generalization), cached ==="
for M in "${LADDER[@]}"; do
  tag=$(echo $M | tr '/' '_')
  echo "--- $M ---"
  python -m dobsteer.run_scale_detect --model "$M" --device $DEV \
      --n-benign $N_BENIGN --seeds $SEEDS --k 32 \
      --out results_scale/detect_${tag}.json 2>&1 | tee results_scale/detect_${tag}.log || \
      echo "SKIP $M (load/access failed)"
done

echo "=== SCALE CONTROL sweep (held-out calibration, large benign) ==="
python -m dobsteer.run_sweep --model Qwen/Qwen2.5-7B-Instruct --device $DEV \
    --use-data --n-benign 300 --seeds $SEEDS --fprs 0,0.01,0.02,0.05,0.1 \
    --k 32 --alpha 3 --tau 1.0 --max-test 40 \
    --out results_scale/control_sweep_qwen7b.json 2>&1 | tee results_scale/control_sweep.log

echo "=== DONE -> results_scale/ ==="

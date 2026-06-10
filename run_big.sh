#!/usr/bin/env bash
# Bigger-model detection runs on a single H100 (80GB).
#   <=32B : fp16, no extra deps.
#   70B+  : 4-bit, needs `pip install bitsandbytes` (and HF login for Llama).
# Cached + generation-free, so these are tractable despite size.
set -e
mkdir -p results_scale
DEV=cuda; N_BENIGN=${N_BENIGN:-400}; SEEDS=${SEEDS:-0,1,2}
run() { # model  quant
  tag=$(echo "$1" | tr '/' '_')
  echo "=== $1  (quant=$2) ==="
  python -m dobsteer.run_scale_detect --model "$1" --device $DEV \
      --n-benign $N_BENIGN --seeds $SEEDS --k 32 --quant "$2" \
      --out results_scale/detect_${tag}.json 2>&1 | tee results_scale/detect_${tag}.log \
      || echo "SKIP $1"
}

# clean fp16 extension of the in-family ladder (no quantization confound)
run "Qwen/Qwen2.5-32B-Instruct" none

# frontier points via 4-bit (footnote the quantization in the paper)
run "Qwen/Qwen2.5-72B-Instruct" 4bit
run "meta-llama/Llama-3.3-70B-Instruct" 4bit

echo "=== DONE -> results_scale/ ==="

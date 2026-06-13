#!/usr/bin/env bash
# Cross-family detection ladder: Gemma-2 (gated -> huggingface-cli login + accept
# license on each model page first). 2b/9b fp16, 27b 4-bit. Cached, no generation.
set -e
mkdir -p results_scale
DEV=cuda; N_BENIGN=${N_BENIGN:-400}; SEEDS=${SEEDS:-0,1,2}
run() {  # model  quant
  tag=$(echo "$1" | tr '/' '_')
  echo "=== $1 (quant=$2) ==="
  python -m dobsteer.run_scale_detect --model "$1" --device $DEV \
      --n-benign $N_BENIGN --seeds $SEEDS --k 32 --quant "$2" \
      --out results_scale/detect_${tag}.json 2>&1 | tee results_scale/detect_${tag}.log \
      || echo "SKIP $1 (load/access failed -- check HF login + license)"
}
run "google/gemma-2-2b-it"  none
run "google/gemma-2-9b-it"  none
run "google/gemma-2-27b-it" 4bit
echo "=== DONE -> results_scale/ ==="

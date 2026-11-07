#!/usr/bin/env bash
set -euo pipefail

# Qwen3-14B: evaluates all seven OOD benchmarks.
# Usage: ./ood_eval/scripts/eval_ood_qwen3_14b.sh [gpu_ranks] [model_or_checkpoint_path]

model_family="qwen3"
default_model="Qwen/Qwen3-14B"

source "./ood_eval/scripts/eval_ood_vllm.sh" "$@"

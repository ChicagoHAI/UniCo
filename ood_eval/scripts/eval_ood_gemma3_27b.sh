#!/usr/bin/env bash
set -euo pipefail

# Gemma-3-27B-IT: evaluates all seven OOD benchmarks.
# Usage: ./ood_eval/scripts/eval_ood_gemma3_27b.sh [gpu_ranks] [model_or_checkpoint_path]

model_family="gemma3"
default_model="google/gemma-3-27b-it"

source "./ood_eval/scripts/eval_ood_vllm.sh" "$@"

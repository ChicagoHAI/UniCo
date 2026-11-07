#!/usr/bin/env bash
set -euo pipefail

# Olmo-3-7B-Instruct: evaluates all seven OOD benchmarks.
# Usage: ./ood_eval/scripts/eval_ood_olmo3_7b.sh [gpu_ranks] [model_or_checkpoint_path]

model_family="olmo3"
default_model="allenai/Olmo-3-7B-Instruct"

source "./ood_eval/scripts/eval_ood_vllm.sh" "$@"

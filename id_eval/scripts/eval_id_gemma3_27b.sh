#!/usr/bin/env bash
set -euo pipefail

# Usage: ./id_eval/scripts/eval_id_gemma3_27b.sh [gpu_ranks] [model_or_checkpoint_path]

model_family="gemma3"
default_model="google/gemma-3-27b-it"

source "./id_eval/scripts/eval_id_vllm.sh" "$@"

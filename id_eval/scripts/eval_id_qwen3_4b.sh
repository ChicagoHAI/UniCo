#!/usr/bin/env bash
set -euo pipefail

# Qwen3-4B: evaluates every record of the 44 prompt files.
# Usage: ./id_eval/scripts/eval_id_qwen3_4b.sh [gpu_ranks] [model_or_checkpoint_path]

model_family="qwen3"
default_model="Qwen/Qwen3-4B"

source "./id_eval/scripts/eval_id_vllm.sh" "$@"

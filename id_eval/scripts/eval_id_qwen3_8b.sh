#!/usr/bin/env bash
set -euo pipefail

# Qwen3-8B: evaluates every record of the 44 prompt files.
# Usage: ./id_eval/scripts/eval_id_qwen3_8b.sh [gpu_ranks] [model_or_checkpoint_path]

model_family="qwen3"
default_model="Qwen/Qwen3-8B"

source "./id_eval/scripts/eval_id_vllm.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail

# Usage: ./id_eval/scripts/eval_id_qwen3_14b.sh [gpu_ranks] [model_or_checkpoint_path]

model_family="qwen3"
default_model="Qwen/Qwen3-14B"

source "./id_eval/scripts/eval_id_vllm.sh" "$@"

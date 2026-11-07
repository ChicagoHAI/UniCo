#!/usr/bin/env bash
set -euo pipefail

# Olmo-3-7B-Instruct: evaluates every record of the 44 prompt files.
# Usage: ./id_eval/scripts/eval_id_olmo3_7b.sh [gpu_ranks] [model_or_checkpoint_path]

model_family="olmo3"
default_model="allenai/Olmo-3-7B-Instruct"

source "./id_eval/scripts/eval_id_vllm.sh" "$@"

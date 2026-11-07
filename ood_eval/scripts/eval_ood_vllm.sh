#!/usr/bin/env bash
set -euo pipefail

# Shared runner of the unified evaluation over the seven out-of-distribution benchmarks.
# The model wrappers source it and set model_family and default_model. Run from the repository root.
#
# Usage: <wrapper>.sh [gpu_ranks] [model_or_checkpoint_path]

if [[ -z "${model_family:-}" || -z "${default_model:-}" ]]; then
    echo "Run a model wrapper such as ./ood_eval/scripts/eval_ood_qwen3_4b.sh" >&2
    exit 1
fi
if [[ $# -gt 2 ]]; then
    echo "Usage: $0 [gpu_ranks] [model_or_checkpoint_path]" >&2
    exit 1
fi

gpu_ranks="${1:-0}"
model="${2:-${default_model}}"
export CUDA_VISIBLE_DEVICES="${gpu_ranks}"
export OMP_NUM_THREADS=1

prompts_path_list=(
    "./ood_eval/benchmarks/bbeh_causal_understanding/data/task.jsonl"
    "./ood_eval/benchmarks/calm_lite/data/calm_lite_aggregated.jsonl"
    "./ood_eval/benchmarks/cladder/data/CLadder-full_v1.5_default_processed.jsonl"
    "./ood_eval/benchmarks/com_2/data/main_filtered_nointerv.jsonl"
    "./ood_eval/benchmarks/corr2cause/data/corr2cause-test-processed.jsonl"
    "./ood_eval/benchmarks/counterbench/data/data_balanced_all.jsonl"
    "./ood_eval/benchmarks/exec_cf/data/val_updated_fields_processed.jsonl"
)

max_new_tokens=16384
results_base_dir="./results/ood_eval"

# Base seed 1 with budget 3 gives round seeds 1, 2, 3: the paper's avg@3.
sampling_seed_list=(
    1
)
budget_list=(
    3
)
start_round=1

infer_model_name() {
    local model_ref="$1"
    local model_basename
    model_basename="$(basename "${model_ref}")"
    if [[ "${model_basename}" == checkpoint-* ]]; then
        basename "$(dirname "$(realpath -m "${model_ref}")")"
    else
        echo "${model_basename}"
    fi
}
model_name="$(infer_model_name "${model}")"

for sampling_seed in "${sampling_seed_list[@]}"; do
for budget in "${budget_list[@]}"; do
    timestamp="$(date +%Y%m%d_%H%M%S)"
    logs_dir="${results_base_dir}/logs/${model_name}/${timestamp}"
    mkdir -p "${logs_dir}"
    log_file="${logs_dir}/${timestamp}-seed${sampling_seed}-budget${budget}-start${start_round}.log"

    echo "=========================================="
    echo "UniCo out-of-distribution evaluation"
    echo "  Model: ${model} (family: ${model_family})"
    echo "  Prompt files: ${#prompts_path_list[@]}"
    echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "  Seed: ${sampling_seed}; budget: ${budget}; start round: ${start_round}"
    echo "  Logging to: ${log_file}"
    echo "=========================================="

    python3 -m ood_eval.eval_ood_vllm \
        --model_family "${model_family}" \
        --model "${model}" \
        --prompts_path_list "${prompts_path_list[@]}" \
        --results_base_dir "${results_base_dir}" \
        --max_new_tokens "${max_new_tokens}" \
        --sampling_seed "${sampling_seed}" \
        --budget "${budget}" \
        --start_round "${start_round}" \
        --run_timestamp "${timestamp}" \
    2>&1 | tee -a "${log_file}"
done
done

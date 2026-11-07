#!/usr/bin/env bash
set -euo pipefail

# Shared runner of the unified in-distribution evaluation over all 44 UniCo prompt files.
# The model wrappers source it and set model_family and default_model. Run from the repository root.
#
# Usage: <wrapper>.sh [gpu_ranks] [model_or_checkpoint_path]

if [[ -z "${model_family:-}" || -z "${default_model:-}" ]]; then
    echo "Run a model wrapper such as ./id_eval/scripts/eval_id_qwen3_4b.sh" >&2
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
    ## symbolic
    "./data/unico/ate/ate-eval.jsonl"
    "./data/unico/cate/cate-eval.jsonl"
    "./data/unico/jate/jate-eval.jsonl"
    "./data/unico/identification/identification-eval.jsonl"
    "./data/unico/backdoor/backdoor-eval.jsonl"
    "./data/unico/frontdoor/frontdoor-eval.jsonl"
    "./data/unico/independence_test/independence_test-eval.jsonl"
    "./data/unico/markov_blanket/markov_blanket-eval.jsonl"
    "./data/unico/att/att-eval.jsonl"
    "./data/unico/cf/cf-eval.jsonl"
    "./data/unico/conditional_prob/conditional_prob-eval.jsonl"
    "./data/unico/joint_prob/joint_prob-eval.jsonl"
    "./data/unico/marginal_prob/marginal_prob-eval.jsonl"
    "./data/unico/nde/nde-eval.jsonl"
    "./data/unico/nie/nie-eval.jsonl"
    "./data/unico/observed_difference/observed_difference-eval.jsonl"
    "./data/unico/pn/pn-eval.jsonl"
    "./data/unico/ps/ps-eval.jsonl"

    ## code
    "./data/unico/ate/ate-code-eval.jsonl"
    "./data/unico/cate/cate-code-eval.jsonl"
    "./data/unico/jate/jate-code-eval.jsonl"
    "./data/unico/att/att-code-eval.jsonl"
    "./data/unico/cf/cf-code-eval.jsonl"
    "./data/unico/conditional_prob/conditional_prob-code-eval.jsonl"
    "./data/unico/joint_prob/joint_prob-code-eval.jsonl"
    "./data/unico/marginal_prob/marginal_prob-code-eval.jsonl"
    "./data/unico/nde/nde-code-eval.jsonl"
    "./data/unico/nie/nie-code-eval.jsonl"
    "./data/unico/observed_difference/observed_difference-code-eval.jsonl"
    "./data/unico/pn/pn-code-eval.jsonl"
    "./data/unico/ps/ps-code-eval.jsonl"

    ## natural language
    "./data/unico/ate/ate-nl-eval.jsonl"
    "./data/unico/cate/cate-nl-eval.jsonl"
    "./data/unico/jate/jate-nl-eval.jsonl"
    "./data/unico/att/att-nl-eval.jsonl"
    "./data/unico/cf/cf-nl-eval.jsonl"
    "./data/unico/conditional_prob/conditional_prob-nl-eval.jsonl"
    "./data/unico/joint_prob/joint_prob-nl-eval.jsonl"
    "./data/unico/marginal_prob/marginal_prob-nl-eval.jsonl"
    "./data/unico/nde/nde-nl-eval.jsonl"
    "./data/unico/nie/nie-nl-eval.jsonl"
    "./data/unico/observed_difference/observed_difference-nl-eval.jsonl"
    "./data/unico/pn/pn-nl-eval.jsonl"
    "./data/unico/ps/ps-nl-eval.jsonl"
)

max_new_tokens=16384
results_base_dir="./results/id_eval"

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
    echo "UniCo in-distribution evaluation"
    echo "  Model: ${model} (family: ${model_family})"
    echo "  Prompt files: ${#prompts_path_list[@]}"
    echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "  Seed: ${sampling_seed}; budget: ${budget}; start round: ${start_round}"
    echo "  Logging to: ${log_file}"
    echo "=========================================="

    python3 -m id_eval.eval_id_vllm \
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

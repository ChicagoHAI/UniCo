#!/usr/bin/env bash
set -euo pipefail

version="v1.1"

# --- Distribution: parallel arrays (#nodes <-> num_examples per split) ---
ate_num_nodes_list=(3 4 5 6 7 8 9 10)
ate_train_num_examples_list=(120 600 960 1200 1200 960 600 360)
ate_eval_num_examples_list=(16 80 128 160 160 128 80 48)

cate_jate_num_nodes_list=(4 5 6 7 8 9 10)
cate_jate_train_num_examples_list=(600 960 1200 1200 960 600 360)
cate_jate_eval_num_examples_list=(80 128 160 160 128 80 48)

# --- Fixed hyperparameters ---
train_seed=0
eval_seed=1000000
dp_values=(1 2 3 4)
query_type_list=(
    "ate"
    "cate"
    "jate"
)
split_list=(
    "train"
    "eval"
)

# --- Nested loops ---
for query_type in "${query_type_list[@]}"; do
    if [[ "$query_type" == "ate" ]]; then
        base_dir="./data_generation/generated_datasets/${version}/ate"
        num_nodes_list=("${ate_num_nodes_list[@]}")
        train_examples=("${ate_train_num_examples_list[@]}")
        eval_examples=("${ate_eval_num_examples_list[@]}")
    else
        base_dir="./data_generation/generated_datasets/${version}/cate_jate"
        num_nodes_list=("${cate_jate_num_nodes_list[@]}")
        train_examples=("${cate_jate_train_num_examples_list[@]}")
        eval_examples=("${cate_jate_eval_num_examples_list[@]}")
    fi

    for split in "${split_list[@]}"; do
        if [[ "$split" == "train" ]]; then
            seed=$train_seed
            num_examples_list=("${train_examples[@]}")
        else
            seed=$eval_seed
            num_examples_list=("${eval_examples[@]}")
        fi

        timestamp=$(date +%Y%m%d_%H%M%S)
        log_dir="${base_dir}/.logs"
        mkdir -p "${log_dir}"

        echo "=== Integrating: ${query_type} ${split} ==="

        python3 -m data_generation.integration_pipeline \
            --query_type "${query_type}" \
            --split "${split}" \
            --base_dir "${base_dir}" \
            --run_timestamp "${timestamp}" \
            --mode both \
            --seed "${seed}" \
            --dp_values "${dp_values[@]}" \
            --num_nodes_list "${num_nodes_list[@]}" \
            --num_examples_list "${num_examples_list[@]}" \
        2>&1 | tee -a "${log_dir}/${timestamp}-integrate-${query_type}-${split}.log"
    done
done

#!/usr/bin/env bash
set -euo pipefail

# identification/backdoor/frontdoor/nie/nde runners write under v2.1; the rest write under v2.0.
version_for_query() {
    case "$1" in
        identification|backdoor|frontdoor|nie|nde) echo "v2.1" ;;
        *) echo "v2.0" ;;
    esac
}

# --- Domain ---
domains=(
    "symbolic"
    "code"
)

# --- Distribution: parallel arrays (#nodes <-> num_examples per split) ---
cf_num_nodes_list=(4 5 6 7 8 9 10)
cf_train_num_examples_list=(600 960 1200 1200 960 600 360)
cf_eval_num_examples_list=(80 128 160 160 128 80 48)

pn_ps_num_nodes_list=(4 5 6 7 8 9 10)
pn_ps_train_num_examples_list=(600 960 1200 1200 960 600 360)
pn_ps_eval_num_examples_list=(80 128 160 160 128 80 48)

att_num_nodes_list=(3 4 5 6 7 8 9 10)
att_train_num_examples_list=(120 600 960 1200 1200 960 600 360)
att_eval_num_examples_list=(16 80 128 160 160 128 80 48)

nie_nde_num_nodes_list=(4 5 6 7 8 9 10)
nie_nde_train_num_examples_list=(600 960 1200 1200 960 600 360)
nie_nde_eval_num_examples_list=(80 128 160 160 128 80 48)

id_num_nodes_list=(4 5 6 7 8 9 10)
id_train_num_examples_list=(153 246 306 306 246 150 93)
id_eval_num_examples_list=(30 48 63 63 48 30 18)

bd_num_nodes_list=(4 5 6 7 8 9 10)
bd_train_num_examples_list=(153 246 306 306 246 150 93)
bd_eval_num_examples_list=(30 48 63 63 48 30 18)

fd_num_nodes_list=(4 5 6 7 8 9 10)
fd_train_num_examples_list=(153 246 306 306 246 150 93)
fd_eval_num_examples_list=(30 48 63 63 48 30 18)

mp_num_nodes_list=(3 4 5 6 7 8 9 10)
mp_train_num_examples_list=(30 150 240 300 300 240 150 90)
mp_eval_num_examples_list=(6 30 48 60 60 48 30 18)

cp_num_nodes_list=(3 4 5 6 7 8 9 10)
cp_train_num_examples_list=(30 150 240 300 300 240 150 90)
cp_eval_num_examples_list=(6 30 48 60 60 48 30 18)

jp_num_nodes_list=(3 4 5 6 7 8 9 10)
jp_train_num_examples_list=(30 150 240 300 300 240 150 90)
jp_eval_num_examples_list=(6 30 48 60 60 48 30 18)

od_num_nodes_list=(3 4 5 6 7 8 9 10)
od_train_num_examples_list=(30 150 240 300 300 240 150 90)
od_eval_num_examples_list=(6 30 48 60 60 48 30 18)

it_num_nodes_list=(4 5 6 7 8 9 10)
it_train_num_examples_list=(153 246 306 306 246 150 93)
it_eval_num_examples_list=(30 48 63 63 48 30 18)

mb_num_nodes_list=(4 5 6 7 8 9 10)
mb_train_num_examples_list=(153 246 306 306 246 150 93)
mb_eval_num_examples_list=(30 48 63 63 48 30 18)

# --- Fixed hyperparameters ---
train_seed=0
eval_seed=1000000
default_dp_values=(1 2 3 4)
query_type_list=(
    "marginal_prob"
    "conditional_prob"
    "joint_prob"
    "observed_difference"
    "independence_test"
    "markov_blanket"
    "identification"
    "backdoor"
    "frontdoor"
    "att"
    "cf"
    "nie"
    "nde"
    "pn"
    "ps"
)
# Graph-only query types have no code rendering.
graph_only_query_types=" independence_test markov_blanket identification backdoor frontdoor "
split_list=(
    "train"
    "eval"
)

# --- Nested loops ---
for domain in "${domains[@]}"; do
for query_type in "${query_type_list[@]}"; do
    if [[ "${domain}" == "code" && "${graph_only_query_types}" == *" ${query_type} "* ]]; then
        continue
    fi
    version="$(version_for_query "${query_type}")"
    current_dp_values=("${default_dp_values[@]}")
    if [[ "$query_type" == "cf" ]]; then
        base_dir="./data_generation/generated_datasets/${version}/cf"
        num_nodes_list=("${cf_num_nodes_list[@]}")
        train_examples=("${cf_train_num_examples_list[@]}")
        eval_examples=("${cf_eval_num_examples_list[@]}")
    elif [[ "$query_type" == "att" ]]; then
        base_dir="./data_generation/generated_datasets/${version}/att"
        num_nodes_list=("${att_num_nodes_list[@]}")
        train_examples=("${att_train_num_examples_list[@]}")
        eval_examples=("${att_eval_num_examples_list[@]}")
    elif [[ "$query_type" == "nie" ]]; then
        base_dir="./data_generation/generated_datasets/${version}/nie"
        num_nodes_list=("${nie_nde_num_nodes_list[@]}")
        train_examples=("${nie_nde_train_num_examples_list[@]}")
        eval_examples=("${nie_nde_eval_num_examples_list[@]}")
    elif [[ "$query_type" == "nde" ]]; then
        base_dir="./data_generation/generated_datasets/${version}/nde"
        num_nodes_list=("${nie_nde_num_nodes_list[@]}")
        train_examples=("${nie_nde_train_num_examples_list[@]}")
        eval_examples=("${nie_nde_eval_num_examples_list[@]}")
    elif [[ "$query_type" == "pn" ]]; then
        base_dir="./data_generation/generated_datasets/${version}/pn"
        num_nodes_list=("${pn_ps_num_nodes_list[@]}")
        train_examples=("${pn_ps_train_num_examples_list[@]}")
        eval_examples=("${pn_ps_eval_num_examples_list[@]}")
    elif [[ "$query_type" == "ps" ]]; then
        base_dir="./data_generation/generated_datasets/${version}/ps"
        num_nodes_list=("${pn_ps_num_nodes_list[@]}")
        train_examples=("${pn_ps_train_num_examples_list[@]}")
        eval_examples=("${pn_ps_eval_num_examples_list[@]}")
    elif [[ "$query_type" == "identification" ]]; then
        base_dir="./data_generation/generated_datasets/${version}/identification"
        num_nodes_list=("${id_num_nodes_list[@]}")
        train_examples=("${id_train_num_examples_list[@]}")
        eval_examples=("${id_eval_num_examples_list[@]}")
        current_dp_values=(0)
    elif [[ "$query_type" == "backdoor" ]]; then
        base_dir="./data_generation/generated_datasets/${version}/backdoor"
        num_nodes_list=("${bd_num_nodes_list[@]}")
        train_examples=("${bd_train_num_examples_list[@]}")
        eval_examples=("${bd_eval_num_examples_list[@]}")
        current_dp_values=(0)
    elif [[ "$query_type" == "frontdoor" ]]; then
        base_dir="./data_generation/generated_datasets/${version}/frontdoor"
        num_nodes_list=("${fd_num_nodes_list[@]}")
        train_examples=("${fd_train_num_examples_list[@]}")
        eval_examples=("${fd_eval_num_examples_list[@]}")
        current_dp_values=(0)
    elif [[ "$query_type" == "marginal_prob" ]]; then
        base_dir="./data_generation/generated_datasets/${version}/marginal_prob"
        num_nodes_list=("${mp_num_nodes_list[@]}")
        train_examples=("${mp_train_num_examples_list[@]}")
        eval_examples=("${mp_eval_num_examples_list[@]}")
    elif [[ "$query_type" == "conditional_prob" ]]; then
        base_dir="./data_generation/generated_datasets/${version}/conditional_prob"
        num_nodes_list=("${cp_num_nodes_list[@]}")
        train_examples=("${cp_train_num_examples_list[@]}")
        eval_examples=("${cp_eval_num_examples_list[@]}")
    elif [[ "$query_type" == "joint_prob" ]]; then
        base_dir="./data_generation/generated_datasets/${version}/joint_prob"
        num_nodes_list=("${jp_num_nodes_list[@]}")
        train_examples=("${jp_train_num_examples_list[@]}")
        eval_examples=("${jp_eval_num_examples_list[@]}")
    elif [[ "$query_type" == "observed_difference" ]]; then
        base_dir="./data_generation/generated_datasets/${version}/observed_difference"
        num_nodes_list=("${od_num_nodes_list[@]}")
        train_examples=("${od_train_num_examples_list[@]}")
        eval_examples=("${od_eval_num_examples_list[@]}")
    elif [[ "$query_type" == "independence_test" ]]; then
        base_dir="./data_generation/generated_datasets/${version}/independence_test"
        num_nodes_list=("${it_num_nodes_list[@]}")
        train_examples=("${it_train_num_examples_list[@]}")
        eval_examples=("${it_eval_num_examples_list[@]}")
        current_dp_values=(2)
    elif [[ "$query_type" == "markov_blanket" ]]; then
        base_dir="./data_generation/generated_datasets/${version}/markov_blanket"
        num_nodes_list=("${mb_num_nodes_list[@]}")
        train_examples=("${mb_train_num_examples_list[@]}")
        eval_examples=("${mb_eval_num_examples_list[@]}")
        current_dp_values=(0)
    else
        echo "Unknown query type: ${query_type}" >&2
        exit 1
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

        echo "=== Integrating: ${query_type} ${domain} ${split} ==="

        python3 -m data_generation.integration_pipeline \
            --query_type "${query_type}" \
            --split "${split}" \
            --base_dir "${base_dir}" \
            --run_timestamp "${timestamp}" \
            --domain "${domain}" \
            --mode both \
            --seed "${seed}" \
            --dp_values "${current_dp_values[@]}" \
            --num_nodes_list "${num_nodes_list[@]}" \
            --num_examples_list "${num_examples_list[@]}" \
        2>&1 | tee -a "${log_dir}/${timestamp}-integrate-${query_type}-${domain}-${split}.log"
    done
done
done

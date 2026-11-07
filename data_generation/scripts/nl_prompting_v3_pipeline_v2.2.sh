#!/usr/bin/env bash
set -euo pipefail

# v2.2 NL domain translation, prompting v3 (entity assign, interpretation, question).
# v3 is intended for num_nodes >= 7; the train split makes many paid API calls.


version="v2.2"
pipeline_version="v3"
manifest_timestamp="${1:?Usage: $0 <manifest_timestamp>}"
manifest_dir="./data_generation/generated_datasets/${version}/manifests"
subcorpus_base="./data_generation/generated_datasets/${version}/nl_subcorpora/${manifest_timestamp}"
passage_corpus="./data_generation/reference_datasets/reference_datasets_aggregated_6k.jsonl"

variant_assignment="manifest"

query_types=(
    "conditional_prob"
    "joint_prob"
    "marginal_prob"
    "observed_difference"
    "att"
    "cf"
    "nie"
    "nde"
    "pn"
    "ps"
)

split_list=(
    "train"
    "eval"
)

num_nodes_list=(
    7
    8
    9
    10
)

dp_list=(1 2 3 4)
train_seed=0
eval_seed=1000000

model_provider_presets=(
    "gpt-5.4-mini:openai|gpt-5.4-mini:openai|gemini-3-flash-preview:vertex"
)

sampling_presets=(
    "low,8192"
)

retry_presets=(
    "3,3,3"
)

other_presets=(
    "8,100,5,10.0"
)

subcorpus_dir="./data_generation/generated_datasets/${version}/nl_subcorpora/${manifest_timestamp}"
if [[ ! -d "${subcorpus_dir}" ]]; then
    echo "No sub-corpus at ${subcorpus_dir}. Pass the timestamp printed by the redistribution step." >&2
    exit 1
fi

for mp_preset in "${model_provider_presets[@]}"; do
for samp_preset in "${sampling_presets[@]}"; do
for ret_preset in "${retry_presets[@]}"; do
for oth_preset in "${other_presets[@]}"; do

    IFS='|' read -r step1_mp step2_mp step3_mp <<< "${mp_preset}"
    IFS=':' read -r entity_assign_model entity_assign_provider <<< "${step1_mp}"
    IFS=':' read -r entity_interp_model entity_interp_provider <<< "${step2_mp}"
    IFS=':' read -r question_model question_provider <<< "${step3_mp}"
    IFS=',' read -r reasoning_effort max_completion_tokens <<< "${samp_preset}"
    IFS=',' read -r entity_assign_retries entity_interp_retries question_retries <<< "${ret_preset}"
    IFS=',' read -r num_processes checkpoint_interval api_retries api_retry_sleep <<< "${oth_preset}"

for split in "${split_list[@]}"; do
    if [[ "${split}" == "train" ]]; then
        seed="${train_seed}"
    else
        seed="${eval_seed}"
    fi

for query_index in "${!query_types[@]}"; do
    qt="${query_types[$query_index]}"
    query_type_offset=$((query_index * 1000))
    manifest_path="${manifest_dir}/${manifest_timestamp}-${qt}-${split}-manifest.jsonl"
    output_base="./data_generation/generated_datasets/${version}/${qt}"

    config_fingerprint=$(printf '%s\n' \
        "${qt}" "${entity_assign_model}" "${entity_assign_provider}" \
        "${entity_interp_model}" "${entity_interp_provider}" \
        "${question_model}" "${question_provider}" \
        "${reasoning_effort}" "${max_completion_tokens}" \
        "${entity_assign_retries}" "${entity_interp_retries}" "${question_retries}" \
        "${num_processes}" "${checkpoint_interval}" "${api_retries}" "${api_retry_sleep}" \
        "${passage_corpus}" "${query_type_offset}" \
        "${variant_assignment}" "${manifest_timestamp}" "${pipeline_version}" \
        | md5sum | cut -c1-8)

for num_nodes in "${num_nodes_list[@]}"; do
    node_dir="${output_base}/nodes${num_nodes}-prompting_${pipeline_version}"
    intermed_dir="${node_dir}/intermediates-nl"
    log_dir="${node_dir}/.logs"
    mkdir -p "${intermed_dir}" "${log_dir}"

for dp in "${dp_list[@]}"; do
    triplet_dir="${subcorpus_base}/${qt}/nodes${num_nodes}/triplets/jsonls"
    if [[ ! -d "${triplet_dir}" ]]; then
        echo "WARN: no triplet dir for ${qt} nodes=${num_nodes}; skipping"
        continue
    fi
    corpus_path=$(find "${triplet_dir}" \
        -name "*-dp${dp}-seed${seed}*.jsonl" | sort | tail -1)
    if [[ -z "${corpus_path}" ]]; then
        echo "WARN: no corpus for ${qt} nodes=${num_nodes} dp=${dp} seed=${seed}; skipping"
        continue
    fi

    timestamp=$(date +%Y%m%d_%H%M%S)
    corpus_basename=$(basename "${corpus_path}" .jsonl)
    src_hash=$(printf '%s' "${corpus_basename}" | md5sum | cut -c1-8)
    checkpoint_tag="qt${qt}-m${manifest_timestamp}-n${num_nodes}-dp${dp}-sd${seed}-${split}-src${src_hash}-cfg${config_fingerprint}"

    echo ""
    echo "====== ${qt} ${split} nodes=${num_nodes} dp=${dp} ======"

    assign_output="${intermed_dir}/${checkpoint_tag}-entity_assign.jsonl"
    python3 -m data_generation.nl.entity_assign \
        --query_type "${qt}" \
        --triplet_corpus_path "${corpus_path}" \
        --passage_corpus_path "${passage_corpus}" \
        --output_path "${assign_output}" \
        --model "${entity_assign_model}" \
        --provider "${entity_assign_provider}" \
        --max_completion_tokens "${max_completion_tokens}" \
        --reasoning_effort "${reasoning_effort}" \
        --num_processes "${num_processes}" \
        --max_retries "${entity_assign_retries}" \
        --api_retries "${api_retries}" \
        --api_retry_sleep "${api_retry_sleep}" \
        --checkpoint_interval "${checkpoint_interval}" \
        --start_idx 0 \
        --end_idx -1 \
        --query_type_offset "${query_type_offset}" \
        2>&1 | tee "${log_dir}/${timestamp}-${checkpoint_tag}-step1_assign.log"

    interp_output="${intermed_dir}/${checkpoint_tag}-entity_interp.jsonl"
    python3 -m data_generation.nl.entity_interpret \
        --query_type "${qt}" \
        --triplet_corpus_path "${corpus_path}" \
        --entity_assign_path "${assign_output}" \
        --output_path "${interp_output}" \
        --model "${entity_interp_model}" \
        --provider "${entity_interp_provider}" \
        --max_completion_tokens "${max_completion_tokens}" \
        --reasoning_effort "${reasoning_effort}" \
        --num_processes "${num_processes}" \
        --max_retries "${entity_interp_retries}" \
        --api_retries "${api_retries}" \
        --api_retry_sleep "${api_retry_sleep}" \
        --checkpoint_interval "${checkpoint_interval}" \
        --start_idx 0 \
        --end_idx -1 \
        2>&1 | tee "${log_dir}/${timestamp}-${checkpoint_tag}-step2_interp.log"

    python3 -m data_generation.nl.nl_step_question_gen_v2 \
        --query_type "${qt}" \
        --triplet_corpus_path "${corpus_path}" \
        --entity_interp_path "${interp_output}" \
        --output_dir "${node_dir}" \
        --run_timestamp "${timestamp}" \
        --pipeline_version "${pipeline_version}" \
        --model "${question_model}" \
        --provider "${question_provider}" \
        --max_completion_tokens "${max_completion_tokens}" \
        --reasoning_effort "${reasoning_effort}" \
        --num_processes "${num_processes}" \
        --max_retries "${question_retries}" \
        --api_retries "${api_retries}" \
        --api_retry_sleep "${api_retry_sleep}" \
        --checkpoint_interval "${checkpoint_interval}" \
        --start_idx 0 \
        --end_idx -1 \
        --checkpoint_name "${checkpoint_tag}" \
        --variant_assignment "${variant_assignment}" \
        --config_fingerprint "${config_fingerprint}" \
        --manifest_path "${manifest_path}" \
        --split "${split}" \
        2>&1 | tee "${log_dir}/${timestamp}-${checkpoint_tag}-step3_question.log"

    for variant in full query_ancestor_prune surgery_ancestor_prune; do
        variant_dir="${node_dir}/${variant}-nl/jsonls"
        if [[ ! -d "${variant_dir}" ]]; then
            continue
        fi
        nl_jsonl=$(find "${variant_dir}" \
            -name "*-qt${qt}-n${num_nodes}-dp${dp}-sd${seed}-va_${variant_assignment}-cfg${config_fingerprint}.jsonl" \
            | sort | tail -1)
        if [[ -n "${nl_jsonl}" ]]; then
            report_name=$(basename "${nl_jsonl}" .jsonl)
            report_path="${node_dir}/${variant}-nl/stats/${report_name}-validation.json"
            python3 -m data_generation.dataset_utils.nl_post_validator \
                --input_path "${nl_jsonl}" \
                --triplet_corpus_path "${corpus_path}" \
                --report_path "${report_path}" \
                2>&1 | tee -a "${log_dir}/${timestamp}-${checkpoint_tag}-postval.log"
        fi
    done

done
done
done
done
done
done
done
done

echo ""
echo "=== v2.2 NL prompting v3 complete ==="

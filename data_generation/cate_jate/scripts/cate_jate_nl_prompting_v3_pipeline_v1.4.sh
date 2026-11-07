#!/usr/bin/env bash
set -euo pipefail

# ==========================================================================
# CATE/JATE NL domain translation — prompting pipeline v3 (three-step)
# Version: v1.4 (redistributed NL sub-corpus from v1.1)
#
# Uses triplet sub-corpora under nl_subcorpora-v1.1/<manifest_timestamp>/
# which preserve original per-DP idx offsets. Sub-corpus records are NOT
# re-indexed; start_idx/end_idx use original per-DP block offsets.
#
# Covers nodes >= 7; nodes <= 6 are handled by the v2 pipeline.
#
# ==========================================================================

# --- Manifest timestamp printed by ./data_generation/scripts/redistribute_manifest_v1.4.sh ---
manifest_timestamp="${1:?Usage: $0 <manifest_timestamp>}"

# --- Distribution: v3 handles nodes 7-10 (CATE/JATE start at 4) ---
num_nodes_list=(
    # 4
    # 5
    # 6
    7
    8
    9
    10
)

# Original per_combo (= num_examples // 8) — used for start_idx/end_idx
# offsets so output filenames match integration_pipeline expectations
orig_train_per_combo_list=(
    # 75
    # 120
    # 150
    150
    120
    75
    45
)
orig_eval_per_combo_list=(
    # 10
    # 16
    # 20
    20
    16
    10
    6
)

# NL per_combo = original // 3 (train) or original // 2 (eval)
train_per_combo_list=(
    # 25     # 75/3 (nodes4: 200 NL total)
    # 40     # 120/3 (nodes5: 320 NL total)
    # 50     # 150/3 (nodes6: 400 NL total)
    50     # 150/3 (nodes7: 400 NL total)
    40     # 120/3 (nodes8: 320 NL total)
    25     # 75/3 (nodes9: 200 NL total)
    15     # 45/3 (nodes10: 120 NL total)
)

eval_per_combo_list=(
    # 5      # 10/2 (nodes4: 40 NL total)
    # 8      # 16/2 (nodes5: 64 NL total)
    # 10     # 20/2 (nodes6: 80 NL total)
    10     # 20/2 (nodes7: 80 NL total)
    8      # 16/2 (nodes8: 64 NL total)
    5      # 10/2 (nodes9: 40 NL total)
    3      # 6/2 (nodes10: 24 NL total)
)

dp_list=(1 2 3 4)
train_seed=0
eval_seed=1000000

# --- Split selection (comment/uncomment as needed) ---
split_list=(
    "train"
    "eval"
)

# --- Version and path settings ---
version="v1.4"
triplet_version="v1.4"
pipeline_version="v3"
output_base="./data_generation/generated_datasets/${version}/cate_jate"
passage_corpus="./data_generation/reference_datasets/reference_datasets_aggregated_6k.jsonl"

# --- Query types ---
query_types=("cate" "jate")

# --- Variant assignment ---
variant_assignment="split"

# ==========================================================================
# Preset arrays (iterated as cross-product)
# ==========================================================================

# Model/provider presets: pipe-delimited steps
# Format: step1_model:step1_provider|step2_model:step2_provider|step3_model:step3_provider
model_provider_presets=(
    # "gpt-5.4-mini:openai|gpt-5.4-mini:openai|gemini-3-flash-preview:openrouter"
    "gpt-5.4-mini:openai|gpt-5.4-mini:openai|gemini-3-flash-preview:vertex"
    # "gpt-5.4-mini:openrouter|gpt-5.4-mini:openrouter|gemini-3-flash-preview:vertex"
)

# Sampling presets: reasoning_effort,max_completion_tokens
sampling_presets=(
    "low,8192"
)

# Retry presets: entity_assign_retries,entity_interp_retries,question_retries
retry_presets=(
    "3,3,3"
)

# Other settings: num_processes,checkpoint_interval,api_retries,api_retry_sleep
other_presets=(
    "8,100,5,10.0"
)

# Seed presets: query_type_seed_enabled,seed_offset_base
# For CATE/JATE the offset is per-query-type when enabled (CATE=1*base, JATE=2*base)
seed_presets=(
    # "false,0"
    "true,1"
)

# ==========================================================================
# Main loop: preset cross-product → split → query_type → num_nodes → dp
# ==========================================================================

subcorpus_dir="./data_generation/generated_datasets/${triplet_version}/nl_subcorpora-v1.1/${manifest_timestamp}"
if [[ ! -d "${subcorpus_dir}" ]]; then
    echo "No sub-corpus at ${subcorpus_dir}. Pass the timestamp printed by the redistribution step." >&2
    exit 1
fi

for mp_preset in "${model_provider_presets[@]}"; do
for samp_preset in "${sampling_presets[@]}"; do
for ret_preset in "${retry_presets[@]}"; do
for oth_preset in "${other_presets[@]}"; do
for seed_preset in "${seed_presets[@]}"; do

    # --- Parse model/provider preset ---
    IFS='|' read -r step1_mp step2_mp step3_mp <<< "${mp_preset}"
    IFS=':' read -r entity_assign_model entity_assign_provider <<< "${step1_mp}"
    IFS=':' read -r entity_interp_model entity_interp_provider <<< "${step2_mp}"
    IFS=':' read -r question_model question_provider <<< "${step3_mp}"

    # --- Parse sampling preset ---
    IFS=',' read -r reasoning_effort max_completion_tokens <<< "${samp_preset}"

    # --- Parse retry preset ---
    IFS=',' read -r entity_assign_retries entity_interp_retries question_retries <<< "${ret_preset}"

    # --- Parse other settings ---
    IFS=',' read -r num_processes checkpoint_interval api_retries api_retry_sleep <<< "${oth_preset}"

    # --- Parse seed preset ---
    IFS=',' read -r query_type_seed_enabled seed_offset_base <<< "${seed_preset}"

    for split in "${split_list[@]}"; do
    triplet_base="./data_generation/generated_datasets/${triplet_version}/nl_subcorpora-v1.1/${manifest_timestamp}/cate_jate"

    # --- Config fingerprint ---
    config_fingerprint=$(printf '%s\n' \
        "${entity_assign_model}" "${entity_assign_provider}" \
        "${entity_interp_model}" "${entity_interp_provider}" \
        "${question_model}" "${question_provider}" \
        "${reasoning_effort}" "${max_completion_tokens}" \
        "${entity_assign_retries}" "${entity_interp_retries}" "${question_retries}" \
        "${num_processes}" "${checkpoint_interval}" "${api_retries}" "${api_retry_sleep}" \
        "${passage_corpus}" "${query_type_seed_enabled}" "${seed_offset_base}" \
        "${variant_assignment}" \
        "${manifest_timestamp}" \
        | md5sum | cut -c1-8)

    echo ""
    echo "===== Preset: mp=${mp_preset} samp=${samp_preset} ret=${ret_preset} cfg=${config_fingerprint} ====="

    for qt in "${query_types[@]}"; do

        if [[ "$query_type_seed_enabled" == "true" ]]; then
            if [[ "$qt" == "cate" ]]; then
                query_type_offset=$((1 * seed_offset_base))
            elif [[ "$qt" == "jate" ]]; then
                query_type_offset=$((2 * seed_offset_base))
            else
                query_type_offset=0
            fi
        else
            query_type_offset=0
        fi

    for i in "${!num_nodes_list[@]}"; do
        num_nodes="${num_nodes_list[$i]}"
        if [[ "$split" == "train" ]]; then
            per_combo="${train_per_combo_list[$i]}"
            orig_per_combo="${orig_train_per_combo_list[$i]}"
            seed=$train_seed
        else
            per_combo="${eval_per_combo_list[$i]}"
            orig_per_combo="${orig_eval_per_combo_list[$i]}"
            seed=$eval_seed
        fi

        node_dir="${output_base}/nodes${num_nodes}-prompting_${pipeline_version}"
        intermed_dir="${node_dir}/intermediates-nl"
        log_dir="${node_dir}/.logs"
        mkdir -p "${intermed_dir}" "${log_dir}"

        block_size=$((per_combo * 2))
        orig_block_size=$((orig_per_combo * 2))
        timestamp=$(date +%Y%m%d_%H%M%S)

        for dp_idx in 0 1 2 3; do
            dp="${dp_list[$dp_idx]}"

            # Use original per-DP offsets so output filenames match
            # integration_pipeline.find_assembly_file() expectations.
            # Sub-corpus records preserve their original idx values.
            start_idx=$((dp_idx * orig_block_size))
            end_idx=$(((dp_idx + 1) * orig_block_size))

            # Discover triplet corpus from v1.4 sub-corpus
            triplet_dir="${triplet_base}/nodes${num_nodes}/triplets/jsonls"
            if [[ ! -d "$triplet_dir" ]]; then
                echo "WARN: No triplet dir for nodes=${num_nodes}; skipping"
                continue
            fi
            corpus_path=$(find "${triplet_dir}" \
                -name "*${qt}*-dp${dp}-seed${seed}*.jsonl" | sort | tail -1)

            if [[ -z "$corpus_path" ]]; then
                echo "WARN: No corpus for ${qt} nodes=${num_nodes}, dp=${dp}, seed=${seed}; skipping"
                continue
            fi

            corpus_basename=$(basename "${corpus_path}" .jsonl)
            checkpoint_tag="${qt}-m${manifest_timestamp}-n${num_nodes}-dp${dp}-seed${seed}-${split}-${corpus_basename}-cfg${config_fingerprint}"

            echo ""
            echo "====== ${qt} nodes=${num_nodes} dp=${dp} split=${split} [${start_idx},${end_idx}) ======"

            # --- Step 1: Entity assignment ---
            assign_output="${intermed_dir}/${checkpoint_tag}-entity_assign.jsonl"
            echo "  Step 1: Entity assignment → ${assign_output}"

            python3 -m data_generation.nl.entity_assign \
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
                --start_idx "${start_idx}" \
                --end_idx "${end_idx}" \
                --query_type_offset "${query_type_offset}" \
                2>&1 | tee "${log_dir}/${timestamp}-${checkpoint_tag}-step1_assign.log"

            # --- Step 2: Entity interpretation ---
            interp_output="${intermed_dir}/${checkpoint_tag}-entity_interp.jsonl"
            echo "  Step 2: Entity interpretation → ${interp_output}"

            python3 -m data_generation.nl.entity_interpret \
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
                --start_idx "${start_idx}" \
                --end_idx "${end_idx}" \
                2>&1 | tee "${log_dir}/${timestamp}-${checkpoint_tag}-step2_interp.log"

            # --- Step 3: Question generation ---
            echo "  Step 3: Question generation"

            python3 -m data_generation.cate_jate.cate_jate_nl_step_question_gen \
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
                --start_idx "${start_idx}" \
                --end_idx "${end_idx}" \
                --checkpoint_name "${checkpoint_tag}" \
                --variant_assignment "${variant_assignment}" \
                --config_fingerprint "${config_fingerprint}" \
                2>&1 | tee "${log_dir}/${timestamp}-${checkpoint_tag}-step3_question.log"

            # --- Post-validation ---
            nl_filename="${timestamp}-start${start_idx}-end${end_idx}-from-${corpus_basename}-va_${variant_assignment}-cfg${config_fingerprint}"
            if [[ "${variant_assignment}" == "split" ]]; then
                expected_per_variant=$((per_combo))
            else
                expected_per_variant=$((block_size))
            fi
            for variant in full surgery_ancestor_prune; do
                nl_jsonl="${node_dir}/${variant}-nl/jsonls/${nl_filename}.jsonl"
                report_path="${node_dir}/${variant}-nl/stats/${nl_filename}-validation.json"
                if [[ -f "${nl_jsonl}" ]]; then
                    echo "  Post-validation: ${variant} (expected_records=${expected_per_variant})"
                    python3 -m data_generation.dataset_utils.nl_post_validator \
                        --input_path "${nl_jsonl}" \
                        --triplet_corpus_path "${corpus_path}" \
                        --report_path "${report_path}" \
                        --expected_records "${expected_per_variant}" \
                        2>&1 | tee -a "${log_dir}/${timestamp}-${checkpoint_tag}-postval.log"
                fi
            done

        done  # dp
    done  # num_nodes
    done  # query_type
    done  # split

done; done; done; done; done  # preset loops

echo ""
echo "=== Pipeline v3 (v1.4 CATE/JATE) complete ==="

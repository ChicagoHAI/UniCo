# Data generation

This pipeline generates the UniCo dataset. It samples causal graphs and queries, computes exact answers with pgmpy, and renders each question in symbolic, code, and natural-language form. The published dataset is on [Hugging Face](https://huggingface.co/datasets/ChicagoHAI/UniCo).

Run every command from the repository root. Outputs go to `./data_generation/generated_datasets/`.

## 1. Sample queries and render them in symbolic and code form

```bash
# ATE, CATE, JATE.
for family in ate cate_jate; do
    ./data_generation/${family}/scripts/${family}_query_sampling_pipeline_v1.1.sh
    ./data_generation/${family}/scripts/${family}_question_assembly_pipeline_v1.1.sh
    ./data_generation/${family}/scripts/${family}_question_assembly_pipeline_code_v1.2.sh
done

# Query types with symbolic and code forms.
for family in att cf conditional_prob joint_prob marginal_prob observed_difference pn_ps; do
    ./data_generation/${family}/scripts/${family}_query_sampling_pipeline_v2.0.sh
    ./data_generation/${family}/scripts/${family}_question_assembly_pipeline_v2.0.sh
    ./data_generation/${family}/scripts/${family}_question_assembly_pipeline_code_v2.0.sh
done
./data_generation/nie_nde/scripts/nie_nde_query_sampling_pipeline_v2.1.sh
./data_generation/nie_nde/scripts/nie_nde_question_assembly_pipeline_v2.1.sh
./data_generation/nie_nde/scripts/nie_nde_question_assembly_pipeline_code_v2.1.sh

# Graph-only query types, symbolic form only.
for family in independence_test markov_blanket; do
    ./data_generation/${family}/scripts/${family}_query_sampling_pipeline_v2.0.sh
    ./data_generation/${family}/scripts/${family}_question_assembly_pipeline_v2.0.sh
done
for family in identification backdoor frontdoor; do
    ./data_generation/${family}/scripts/${family}_query_sampling_pipeline_v2.1.sh
    ./data_generation/${family}/scripts/${family}_question_assembly_pipeline_v2.1.sh
done
```

A `pn_ps` or `nie_nde` run samples both query types of its name and writes each one to its own directory.

## 2. Integrate per-combination files into one file per query type and split

```bash
./data_generation/scripts/integrate_dataset_v1.1.sh           # ATE, CATE, JATE, symbolic.
./data_generation/scripts/integrate_dataset_v1.2.sh           # ATE, CATE, JATE, code.
./data_generation/scripts/integrate_dataset_new_queries.sh    # All other query types.
```

## 3. Divide records among the representation forms

```bash
./data_generation/scripts/redistribute_dataset_v1.4.sh    # ATE, CATE, JATE. Prints a manifest timestamp.
./data_generation/scripts/redistribute_dataset_v2.2.sh    # Ten other query types. Prints a manifest timestamp.
```

## 4. Render the natural-language form

This step calls LLM APIs. Set `OPENAI_API_KEY` and Google application default credentials with `GCP_PROJECT_ID`, in the environment or in a `.env` file at the repository root. Each runner's `model_provider_presets` array selects the models. Pass each runner the manifest timestamp printed by step 3.

```bash
# ATE, CATE, JATE.
for family in ate cate_jate; do
    ./data_generation/${family}/scripts/${family}_nl_prompting_v2_pipeline_v1.4.sh <v1.4 manifest timestamp>
    ./data_generation/${family}/scripts/${family}_nl_prompting_v3_pipeline_v1.4.sh <v1.4 manifest timestamp>
done
./data_generation/scripts/integrate_dataset_v1.4.sh

# Ten other query types.
./data_generation/scripts/nl_prompting_v2_pipeline_v2.2.sh <v2.2 manifest timestamp>
./data_generation/scripts/nl_prompting_v3_pipeline_v2.2.sh <v2.2 manifest timestamp>
./data_generation/scripts/integrate_dataset_v2.2_nl.sh <v2.2 manifest timestamp>
```

The runners read a passage corpus built from BBC News, NarraSum, and Wikipedia paragraphs. To build it, put these files under `./data_generation/reference_datasets/`: `bbc_news_alltime/bbc_news_alltime.jsonl` (passage in `content`), `narrasum/sampled_data.jsonl` (`summary`), and `wikipedia-paragraphs/wikipedia-paragraphs.jsonl` (`text`). Then run:

```bash
python3 -m data_generation.data_processing.unify_reference_content
python3 -m data_generation.data_processing.filter_reference_by_length
python3 -m data_generation.data_processing.aggregate_reference_datasets
```

## Final files

| Query types | Directory under `./data_generation/generated_datasets/` | Forms |
| --- | --- | --- |
| `ate` | `v1.4/ate/` | symbolic, code, NL |
| `cate`, `jate` | `v1.4/cate_jate/` | symbolic, code, NL |
| `att`, `cf`, `conditional_prob`, `joint_prob`, `marginal_prob`, `observed_difference`, `pn`, `ps`, `nde`, `nie` | `v2.2/<query_type>/` | symbolic, code, NL |
| `independence_test`, `markov_blanket` | `v2.0/<query_type>/` | symbolic |
| `identification`, `backdoor`, `frontdoor` | `v2.1/<query_type>/` | symbolic |

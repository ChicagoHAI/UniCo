# In-distribution evaluation

`./id_eval/scripts/eval_id_qwen3_4b.sh 0` sends all 13,321 prompts of UniCo's 44 evaluation files to Qwen3-4B for three rounds. It writes four tables, one for all forms together and one per form, with a score per query type and their average weighted by record count. Download the files first with `./id_eval/scripts/download_unico_data.sh`.

| Causal ladder level | Query type | Symbolic | Code | Natural language |
| --- | --- | --- | --- | --- |
| Association | `conditional_prob` | 100 | 100 | 100 |
| Association | `joint_prob` | 100 | 100 | 100 |
| Association | `marginal_prob` | 100 | 100 | 100 |
| Association | `observed_difference` | 100 | 100 | 100 |
| Association | `independence_test` | 300 | - | - |
| Association | `markov_blanket` | 300 | - | - |
| Intervention | `ate` | 400 | 400 | 400 |
| Intervention | `cate` | 392 | 392 | 392 |
| Intervention | `jate` | 392 | 392 | 392 |
| Intervention | `identification` | 300 | - | - |
| Intervention | `backdoor` | 300 | - | - |
| Intervention | `frontdoor` | 300 | - | - |
| Counterfactual | `cf` | 392 | 392 | 387 |
| Counterfactual | `att` | 400 | 400 | 400 |
| Counterfactual | `nde` | 392 | 392 | 392 |
| Counterfactual | `nie` | 392 | 392 | 392 |
| Counterfactual | `pn` | 392 | 392 | 391 |
| Counterfactual | `ps` | 392 | 392 | 387 |

`grading.py` holds the grading rules. To also download the training split, run `python3 -m id_eval.download_unico_data --splits test train`.

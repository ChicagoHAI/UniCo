# Out-of-distribution evaluation

`./ood_eval/scripts/eval_ood_qwen3_4b.sh 0` sends all 18,065 prompts of the seven benchmarks below to Qwen3-4B for three rounds. It writes one table with a primary metric per benchmark and their mean. The prompt files are in this repository, so no download is needed.

| Benchmark | Directory under `./ood_eval/benchmarks/` | Records | Primary metric |
| --- | --- | --- | --- |
| [BBEH](https://github.com/google-deepmind/bbeh), causal understanding | `bbeh_causal_understanding` | 200 | Accuracy |
| [CaLM-Lite](https://github.com/OpenCausaLab/CaLM) | `calm_lite` | 3,900 | Accuracy |
| [CLadder](https://github.com/causalNLP/cladder), full v1.5 | `cladder` | 10,112 | Accuracy |
| [Com2](https://github.com/Waste-Wood/Com2), counterfactual and decision | `com_2` | 991 | F1 |
| [Corr2Cause](https://github.com/causalNLP/corr2cause), test split | `corr2cause` | 1,162 | F1 |
| [CounterBench](https://huggingface.co/datasets/CounterBench/CounterBench) | `counterbench` | 1,200 | Accuracy |
| [Executable Counterfactuals](https://huggingface.co/datasets/Raidriar-Dai/executable-counterfactuals), if-else | `exec_cf` | 500 | F1 |

Each benchmark directory holds its prompt file in `data/`, its grading rules in `eval_utils.py`, and the script that built the prompt file in `data_processing/`. Please cite the original papers when you use these benchmarks.

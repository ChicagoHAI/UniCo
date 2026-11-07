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

Each benchmark directory holds its prompt file in `data/`, its grading rules in `eval_utils.py`, and the script that built the prompt file in `data_processing/`.

## License and citation

The Apache License 2.0 covers the code in this repository. The benchmark files in `./ood_eval/benchmarks/*/data/` follow the license of each original work.

If you use this evaluation code on a benchmark, please cite its original work.

<details>
<summary>BibTeX for the seven original works</summary>

```bibtex
@inproceedings{kazemi2025big,
  title={Big-bench extra hard},
  author={Kazemi, Mehran and Fatemi, Bahare and Bansal, Hritik and Palowitch, John and Anastasiou, Chrysovalantis and Mehta, Sanket Vaibhav and Jain, Lalit K and Aglietti, Virginia and Jindal, Disha and Chen, Yuanzhu Peter and others},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={26473--26501},
  year={2025}
}

@article{chen2024causal,
  title={Causal evaluation of language models},
  author={Chen, Sirui and Peng, Bo and Chen, Meiqi and Wang, Ruiqi and Xu, Mengying and Zeng, Xingyu and Zhao, Rui and Zhao, Shengjie and Qiao, Yu and Lu, Chaochao},
  journal={arXiv preprint arXiv:2405.00622},
  year={2024}
}

@inproceedings{jin2023cladder,
  title={{CL}adder: A Benchmark to Assess Causal Reasoning Capabilities of Language Models},
  author={Zhijing Jin and Yuen Chen and Felix Leeb and Luigi Gresele and Ojasv Kamal and Zhiheng LYU and Kevin Blin and Fernando Gonzalez Adauto and Max Kleiman-Weiner and Mrinmaya Sachan and Bernhard Sch{\"o}lkopf},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=e2wtjx0Yqu}
}

@inproceedings{xiong2025com2,
  title={Com2: A causal-guided benchmark for exploring complex commonsense reasoning in large language models},
  author={Xiong, Kai and Ding, Xiao and Cao, Yixin and Yan, Yuxiong and Du, Li and Zhang, Yufei and Gao, Jinglong and Liu, Jiaqian and Qin, Bing and Liu, Ting},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={16119--16140},
  year={2025}
}

@inproceedings{jin2024can,
  title={Can Large Language Models Infer Causation from Correlation?},
  author={Zhijing Jin and Jiarui Liu and Zhiheng LYU and Spencer Poff and Mrinmaya Sachan and Rada Mihalcea and Mona T. Diab and Bernhard Sch{\"o}lkopf},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=vqIH0ObdqL}
}

@article{chen2025counterbench,
  title={Counterbench: A benchmark for counterfactuals reasoning in large language models},
  author={Chen, Yuefei and Singh, Vivek K and Ma, Jing and Tang, Ruxiang},
  journal={arXiv preprint arXiv:2502.11008},
  year={2025}
}

@inproceedings{vashishtha2026executable,
  title={Executable Counterfactuals: Improving {LLM}s' Causal Reasoning Through Code},
  author={Aniket Vashishtha and Qirun Dai and Hongyuan Mei and Amit Sharma and Chenhao Tan and Hao Peng},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=Lm46gJA0q8}
}
```

</details>

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import networkx as nx
from tqdm import tqdm

from data_generation.markov_blanket.markov_blanket_query_sampling import MarkovBlanketResult
from data_generation.markov_blanket.markov_blanket_question_assembly import assemble_all_questions, format_answer_for_export

VARIANT_NAMES = ["full"]


def ensure_output_dirs(output_dir: str) -> None:
    for variant_name in VARIANT_NAMES:
        for subdir in ('jsonls', 'csvs', 'stats'):
            Path(output_dir, variant_name, subdir).mkdir(parents=True, exist_ok=True)


def reconstruct_from_triplet(record: Dict):
    dag = nx.DiGraph(); dag.add_nodes_from(range(record['num_nodes'])); dag.add_edges_from([tuple(edge) for edge in record['dag_edges']])
    return dag, MarkovBlanketResult.from_json_dict(record['markov_blanket_result']), record['example_seed'], record['idx']


def main() -> None:
    parser = argparse.ArgumentParser(description='Markov Blanket question assembly pipeline')
    parser.add_argument('--triplet_corpus_path', type=str, required=True)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--run_timestamp', type=str, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in open(args.triplet_corpus_path, 'r', encoding='utf-8') if line.strip()]
    ensure_output_dirs(args.output_dir)
    end_idx = args.end_idx if args.end_idx >= 0 else len(records)
    selected = [r for r in records if args.start_idx <= r['idx'] < end_idx]
    outputs = {variant: [] for variant in VARIANT_NAMES}
    for record in tqdm(selected, desc='Assembling questions'):
        dag, result, example_seed, example_idx = reconstruct_from_triplet(record)
        for question in assemble_all_questions(dag, result, example_seed=example_seed, example_idx=example_idx):
            question.metadata['num_nodes'] = record['num_nodes']
            question.metadata['dag_edges'] = record['dag_edges']
            outputs[question.variant_name].append({'idx': record['idx'], 'question': question.full_question_text, 'answer': format_answer_for_export(question.metadata['query_format'], question.answer), 'metadata': question.metadata})
    filename = f"{args.run_timestamp}-start{args.start_idx}-end{end_idx}-from-{Path(args.triplet_corpus_path).stem}"
    for variant, rows in outputs.items():
        jsonl_path = Path(args.output_dir, variant, 'jsonls', f"{filename}.jsonl")
        csv_path = Path(args.output_dir, variant, 'csvs', f"{filename}.csv")
        stats_path = Path(args.output_dir, variant, 'stats', f"{filename}.json")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['idx', 'question', 'answer', 'metadata']); writer.writeheader()
            for row in rows: writer.writerow({'idx': row['idx'], 'question': row['question'], 'answer': row['answer'], 'metadata': json.dumps(row['metadata'], ensure_ascii=False)})
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({'query_type': 'markov_blanket', 'num_examples': len(rows), 'source_triplet_corpus': args.triplet_corpus_path}, f, indent=2, ensure_ascii=False); f.write("\n")


if __name__ == '__main__':
    main()

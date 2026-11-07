from __future__ import annotations

import json
from pathlib import Path
from typing import List

import networkx as nx


DAG_CACHE_DIR = Path(__file__).resolve().parent / "dag_cache"


def _cache_path(num_nodes: int) -> Path:
    return DAG_CACHE_DIR / f"nodes{num_nodes}-complete.json"


def load_complete_unique_dags(num_nodes: int) -> List[nx.DiGraph] | None:
    path = _cache_path(num_nodes)
    if not path.exists():
        return None

    data = json.loads(path.read_text())
    if data.get("num_nodes") != num_nodes:
        raise ValueError(
            f"DAG cache mismatch for num_nodes={num_nodes}: found {data.get('num_nodes')}"
        )
    if not data.get("is_complete", False):
        raise ValueError(f"DAG cache is not marked complete: {path}")

    dags = []
    for edge_list in data["dag_edges_list"]:
        dag = nx.DiGraph()
        dag.add_nodes_from(range(num_nodes))
        dag.add_edges_from((int(u), int(v)) for u, v in edge_list)
        dags.append(dag)
    return dags


def save_complete_unique_dags(num_nodes: int, dags: List[nx.DiGraph]) -> Path:
    DAG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(num_nodes)
    payload = {
        "num_nodes": num_nodes,
        "num_dags": len(dags),
        "is_complete": True,
        "dag_edges_list": [
            sorted([[int(u), int(v)] for u, v in dag.edges()])
            for dag in dags
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path

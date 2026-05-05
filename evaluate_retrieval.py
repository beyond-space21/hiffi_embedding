import json
from pathlib import Path

from qdrant_service import search


def recall_at_k(results: list[str], relevant: set[str], k: int) -> float:
    top_k = set(results[:k])
    if not relevant:
        return 0.0
    return len(top_k.intersection(relevant)) / len(relevant)


def run_eval(dataset_path: str, k: int = 10) -> None:
    rows = json.loads(Path(dataset_path).read_text())
    recalls = []
    for row in rows:
        query = row["query"]
        relevant = set(row["relevant_video_ids"])
        resp = search(query, limit=k)
        ranked_ids = [item[0] for item in resp["results"]]
        recalls.append(recall_at_k(ranked_ids, relevant, k))

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    print(f"queries={len(recalls)} recall@{k}={avg_recall:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval quality from labeled dataset.")
    parser.add_argument("--dataset", required=True, help="Path to JSON file with query/relevant ids.")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    run_eval(args.dataset, k=args.k)

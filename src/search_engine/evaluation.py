from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class EvaluationResult:
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    queries_evaluated: int


def precision_at_k(relevant_ids: set[int], retrieved_ids: Sequence[int], k: int) -> float:
    top_k = list(retrieved_ids[:k])
    if not top_k:
        return 0.0

    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(relevant_ids: set[int], retrieved_ids: Sequence[int], k: int) -> float:
    if not relevant_ids:
        return 0.0

    top_k = set(retrieved_ids[:k])
    hits = len(top_k.intersection(relevant_ids))
    return hits / len(relevant_ids)


def evaluate_engine(
    engine,
    query_texts: Sequence[str],
    query_labels: Sequence[int],
    k_values: Iterable[int],
    use_query_expansion: bool = True,
) -> EvaluationResult:
    if len(query_texts) != len(query_labels):
        raise ValueError("query_texts and query_labels must have same length.")

    cleaned_k_values = sorted({int(k) for k in k_values if int(k) > 0})
    if not cleaned_k_values:
        raise ValueError("At least one positive k value is required.")

    max_k = cleaned_k_values[-1]
    precision_sums = {k: 0.0 for k in cleaned_k_values}
    recall_sums = {k: 0.0 for k in cleaned_k_values}

    evaluated_queries = 0
    for query, label in zip(query_texts, query_labels):
        relevant = engine.relevant_doc_ids(label)
        if not relevant:
            continue

        retrieved = engine.search(
            query,
            k=max_k,
            use_query_expansion=use_query_expansion,
        )
        retrieved_ids = [item.doc_id for item in retrieved]

        for k in cleaned_k_values:
            precision_sums[k] += precision_at_k(relevant, retrieved_ids, k)
            recall_sums[k] += recall_at_k(relevant, retrieved_ids, k)

        evaluated_queries += 1

    if evaluated_queries == 0:
        return EvaluationResult(
            precision_at_k={k: 0.0 for k in cleaned_k_values},
            recall_at_k={k: 0.0 for k in cleaned_k_values},
            queries_evaluated=0,
        )

    return EvaluationResult(
        precision_at_k={k: precision_sums[k] / evaluated_queries for k in cleaned_k_values},
        recall_at_k={k: recall_sums[k] / evaluated_queries for k in cleaned_k_values},
        queries_evaluated=evaluated_queries,
    )

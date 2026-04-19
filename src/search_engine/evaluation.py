from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class EvaluationResult:
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    map_score: float
    queries_requested: int
    queries_evaluated: int
    queries_skipped: int


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


def average_precision_at_k(relevant_ids: set[int], retrieved_ids: Sequence[int], k: int) -> float:
    if not relevant_ids:
        return 0.0

    top_k = list(retrieved_ids[:k])
    if not top_k:
        return 0.0

    hit_count = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id not in relevant_ids:
            continue

        hit_count += 1
        precision_sum += hit_count / rank

    if hit_count == 0:
        return 0.0

    normalizer = min(len(relevant_ids), len(top_k))
    if normalizer == 0:
        return 0.0
    return precision_sum / normalizer


def ndcg_at_k(relevant_ids: set[int], retrieved_ids: Sequence[int], k: int) -> float:
    if not relevant_ids:
        return 0.0

    top_k = list(retrieved_ids[:k])
    if not top_k:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), len(top_k))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0

    return dcg / idcg


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
    ndcg_sums = {k: 0.0 for k in cleaned_k_values}
    map_sum = 0.0
    queries_requested = len(query_texts)

    evaluated_queries = 0
    skipped_queries = 0
    for query, label in zip(query_texts, query_labels):
        relevant = engine.relevant_doc_ids(label)
        if not relevant:
            skipped_queries += 1
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
            ndcg_sums[k] += ndcg_at_k(relevant, retrieved_ids, k)
        map_sum += average_precision_at_k(relevant, retrieved_ids, max_k)

        evaluated_queries += 1

    if evaluated_queries == 0:
        return EvaluationResult(
            precision_at_k={k: 0.0 for k in cleaned_k_values},
            recall_at_k={k: 0.0 for k in cleaned_k_values},
            ndcg_at_k={k: 0.0 for k in cleaned_k_values},
            map_score=0.0,
            queries_requested=queries_requested,
            queries_evaluated=0,
            queries_skipped=skipped_queries,
        )

    return EvaluationResult(
        precision_at_k={k: precision_sums[k] / evaluated_queries for k in cleaned_k_values},
        recall_at_k={k: recall_sums[k] / evaluated_queries for k in cleaned_k_values},
        ndcg_at_k={k: ndcg_sums[k] / evaluated_queries for k in cleaned_k_values},
        map_score=map_sum / evaluated_queries,
        queries_requested=queries_requested,
        queries_evaluated=evaluated_queries,
        queries_skipped=skipped_queries,
    )

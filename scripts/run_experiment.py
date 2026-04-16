from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from search_engine.data import load_text_classification_dataset
from search_engine.engine import SearchEngineConfig, WordEmbeddingSearchEngine
from search_engine.evaluation import EvaluationResult, evaluate_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Word embedding based search on AG News.")
    parser.add_argument("--dataset", type=str, default="sh0416/ag_news")
    parser.add_argument("--embedding", type=str, choices=["word2vec", "bert"], default="word2vec")
    parser.add_argument("--max-train", type=int, default=30000)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--eval-queries", type=int, default=1000)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-examples", type=int, default=3)

    parser.add_argument("--ann-n-neighbors", type=int, default=30)
    parser.add_argument("--ann-random-state", type=int, default=42)

    parser.add_argument("--expansion-per-term", type=int, default=3)

    parser.add_argument("--w2v-vector-size", type=int, default=200)
    parser.add_argument("--w2v-window", type=int, default=5)
    parser.add_argument("--w2v-min-count", type=int, default=2)
    parser.add_argument("--w2v-workers", type=int, default=4)
    parser.add_argument("--w2v-epochs", type=int, default=8)

    parser.add_argument(
        "--bert-model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    return parser.parse_args()


def short_text(text: str, max_len: int = 180) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 3] + "..."


def print_metrics(title: str, result: EvaluationResult) -> None:
    print(f"\n{title}")
    print(f"Queries evaluated: {result.queries_evaluated}")
    print("Precision@k:")
    for k in sorted(result.precision_at_k):
        print(f"  P@{k}: {result.precision_at_k[k]:.4f}")
    print("Recall@k:")
    for k in sorted(result.recall_at_k):
        print(f"  R@{k}: {result.recall_at_k[k]:.4f}")


def sample_query_indices(total: int, n: int, seed: int) -> list[int]:
    n = min(total, n)
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    return indices[:n]


def show_examples(engine, dataset, indices: list[int], top_k: int, show_count: int) -> None:
    if show_count <= 0:
        return

    print("\nSample queries and retrieval outputs")
    for idx in indices[:show_count]:
        query = dataset.test.texts[idx]
        label = dataset.test.labels[idx]
        label_name = dataset.label_names.get(label, str(label))

        print("\n" + "=" * 80)
        print(f"Query label: {label} ({label_name})")
        print(f"Query text: {short_text(query)}")

        if engine.supports_query_expansion():
            expanded_terms = engine.expand_query_terms(query)
            print(f"Expanded terms: {expanded_terms[:12]}")

        results = engine.search(query, k=top_k, use_query_expansion=True)
        for rank, result in enumerate(results, start=1):
            result_label_name = dataset.label_names.get(result.label, str(result.label))
            print(
                f"  {rank:>2}. doc_id={result.doc_id} label={result.label} ({result_label_name}) "
                f"score={result.score:.4f} text={short_text(result.text, max_len=110)}"
            )


def main() -> None:
    args = parse_args()

    print("Loading dataset...")
    dataset = load_text_classification_dataset(
        dataset_name=args.dataset,
        max_train=args.max_train,
        max_test=args.max_test,
    )
    print(f"Train size: {len(dataset.train.texts)}")
    print(f"Test size:  {len(dataset.test.texts)}")

    config = SearchEngineConfig(
        embedding_backend=args.embedding,
        ann_n_neighbors=args.ann_n_neighbors,
        ann_random_state=args.ann_random_state,
        expansion_per_term=args.expansion_per_term,
        w2v_vector_size=args.w2v_vector_size,
        w2v_window=args.w2v_window,
        w2v_min_count=args.w2v_min_count,
        w2v_workers=args.w2v_workers,
        w2v_epochs=args.w2v_epochs,
        bert_model_name=args.bert_model_name,
    )

    print(f"Building engine with backend={args.embedding}...")
    engine = WordEmbeddingSearchEngine(config)
    engine.fit(dataset.train.texts, dataset.train.labels)

    query_indices = sample_query_indices(
        total=len(dataset.test.texts),
        n=args.eval_queries,
        seed=args.seed,
    )
    query_texts = [dataset.test.texts[idx] for idx in query_indices]
    query_labels = [dataset.test.labels[idx] for idx in query_indices]

    if args.embedding == "word2vec":
        baseline = evaluate_engine(
            engine,
            query_texts=query_texts,
            query_labels=query_labels,
            k_values=args.k,
            use_query_expansion=False,
        )
        expanded = evaluate_engine(
            engine,
            query_texts=query_texts,
            query_labels=query_labels,
            k_values=args.k,
            use_query_expansion=True,
        )

        print_metrics("Word2Vec Retrieval (no query expansion)", baseline)
        print_metrics("Word2Vec Retrieval (with query expansion)", expanded)

        print("\nDelta from expansion")
        for k in sorted(expanded.precision_at_k):
            p_delta = expanded.precision_at_k[k] - baseline.precision_at_k[k]
            r_delta = expanded.recall_at_k[k] - baseline.recall_at_k[k]
            print(f"  k={k}: delta P@k={p_delta:+.4f}, delta R@k={r_delta:+.4f}")
    else:
        bert_result = evaluate_engine(
            engine,
            query_texts=query_texts,
            query_labels=query_labels,
            k_values=args.k,
            use_query_expansion=False,
        )
        print_metrics("BERT Retrieval", bert_result)

    show_examples(
        engine=engine,
        dataset=dataset,
        indices=query_indices,
        top_k=max(args.k),
        show_count=args.show_examples,
    )


if __name__ == "__main__":
    main()

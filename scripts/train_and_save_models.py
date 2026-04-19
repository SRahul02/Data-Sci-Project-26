from __future__ import annotations

import os
import pickle
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


def _local_venv_site_packages(project_root: Path) -> Path | None:
    if os.name == "nt":
        candidate = project_root / ".venv" / "Lib" / "site-packages"
        return candidate if candidate.exists() else None

    lib_dir = project_root / ".venv" / "lib"
    if not lib_dir.exists():
        return None
    matches = sorted(lib_dir.glob("python*/site-packages"))
    return matches[0] if matches else None


def _prefer_local_venv_packages(project_root: Path) -> None:
    site_packages = _local_venv_site_packages(project_root)
    if site_packages is None:
        return

    site_packages_str = str(site_packages)
    if site_packages_str not in sys.path:
        sys.path.insert(0, site_packages_str)

    if os.name == "nt":
        scripts_dir = project_root / ".venv" / "Scripts"
        if scripts_dir.exists():
            os.environ["PATH"] = f"{scripts_dir}{os.pathsep}{os.environ.get('PATH', '')}"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_prefer_local_venv_packages(PROJECT_ROOT)
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from search_engine.data import load_text_classification_dataset
from search_engine.engine import SearchEngineConfig, WordEmbeddingSearchEngine
from search_engine.evaluation import EvaluationResult, evaluate_engine


# Editable constants
DATASET_NAME = "sh0416/ag_news"
MODEL_NAME = "bert"  # "word2vec" or "bert"
MAX_TRAIN = 50000
EVAL_QUERIES = 1000

# Save one checkpoint per epoch for Word2Vec. Highest epoch is the latest/best checkpoint.
MODEL_EPOCHS = [2000]

# Optional constants
MAX_TEST = None
K_VALUES = [5, 10, 20]
SEED = 42
USE_ANN = False
ANN_N_NEIGHBORS = 30
ANN_RANDOM_STATE = 42
EXPANSION_PER_TERM = 3

W2V_VECTOR_SIZE = 200
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_WORKERS = 4
BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _sample_query_indices(total: int, n: int, seed: int) -> list[int]:
    n = min(total, n)
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    return indices[:n]


def _training_epochs() -> list[int]:
    epochs = sorted({int(epoch) for epoch in MODEL_EPOCHS if int(epoch) > 0})
    if not epochs:
        raise ValueError("MODEL_EPOCHS must contain at least one positive integer.")

    if MODEL_NAME.lower() != "word2vec":
        return [max(epochs)]
    return epochs


def _build_config(epoch: int) -> SearchEngineConfig:
    backend = MODEL_NAME.lower()
    if backend not in {"word2vec", "bert"}:
        raise ValueError("MODEL_NAME must be either 'word2vec' or 'bert'.")

    return SearchEngineConfig(
        embedding_backend=backend,
        use_ann=USE_ANN,
        ann_n_neighbors=ANN_N_NEIGHBORS,
        ann_random_state=ANN_RANDOM_STATE,
        expansion_per_term=EXPANSION_PER_TERM,
        w2v_vector_size=W2V_VECTOR_SIZE,
        w2v_window=W2V_WINDOW,
        w2v_min_count=W2V_MIN_COUNT,
        w2v_workers=W2V_WORKERS,
        w2v_epochs=epoch,
        w2v_seed=SEED,
        bert_model_name=BERT_MODEL_NAME,
    )


def _result_to_dict(result: EvaluationResult) -> dict[str, object]:
    return {
        "precision_at_k": dict(result.precision_at_k),
        "recall_at_k": dict(result.recall_at_k),
        "ndcg_at_k": dict(result.ndcg_at_k),
        "map_score": result.map_score,
        "queries_requested": result.queries_requested,
        "queries_evaluated": result.queries_evaluated,
        "queries_skipped": result.queries_skipped,
    }


def _save_checkpoint(
    engine: WordEmbeddingSearchEngine,
    config: SearchEngineConfig,
    epoch: int,
    metrics: EvaluationResult,
) -> Path:
    model_dir = PROJECT_ROOT / "models" / MODEL_NAME.lower()
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir / f"epoch_{epoch:03d}.pkl"

    bert_embedder_backup = engine.bert_embedder
    doc_index_backend_backup = engine.doc_index._index if engine.doc_index is not None else None
    term_index_backend_backup = engine.term_index._index if engine.term_index is not None else None

    payload = {
        "engine": engine,
        "config": asdict(config),
        "dataset_name": DATASET_NAME,
        "max_train": MAX_TRAIN,
        "max_test": MAX_TEST,
        "eval_queries": EVAL_QUERIES,
        "k_values": list(K_VALUES),
        "seed": SEED,
        "trained_epoch": epoch,
        "metrics": _result_to_dict(metrics),
    }

    try:
        # Strip runtime-only index/model handles to make checkpoint loading more robust.
        if engine.config.embedding_backend.lower() == "bert":
            engine.bert_embedder = None
        if engine.doc_index is not None:
            engine.doc_index._index = None
        if engine.term_index is not None:
            engine.term_index._index = None

        with checkpoint_path.open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        if engine.config.embedding_backend.lower() == "bert":
            engine.bert_embedder = bert_embedder_backup
        if engine.doc_index is not None:
            engine.doc_index._index = doc_index_backend_backup
        if engine.term_index is not None:
            engine.term_index._index = term_index_backend_backup

    return checkpoint_path


def _print_eval_result(epoch: int, result: EvaluationResult) -> None:
    print(f"\nEpoch {epoch} evaluation")
    print(f"Queries requested: {result.queries_requested}")
    print(f"Queries evaluated: {result.queries_evaluated}")
    print(f"Queries skipped:   {result.queries_skipped}")
    print(f"MAP: {result.map_score:.4f}")
    for k in sorted(result.precision_at_k):
        precision = result.precision_at_k[k]
        recall = result.recall_at_k[k]
        ndcg = result.ndcg_at_k[k]
        print(f"  P@{k}: {precision:.4f} | R@{k}: {recall:.4f} | nDCG@{k}: {ndcg:.4f}")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    print("Loading dataset...")
    dataset = load_text_classification_dataset(
        dataset_name=DATASET_NAME,
        max_train=MAX_TRAIN,
        max_test=MAX_TEST,
    )
    print(f"Train size: {len(dataset.train.texts)}")
    print(f"Test size:  {len(dataset.test.texts)}")

    query_indices = _sample_query_indices(len(dataset.test.texts), EVAL_QUERIES, SEED)
    query_texts = [dataset.test.texts[index] for index in query_indices]
    query_labels = [dataset.test.labels[index] for index in query_indices]

    epochs = _training_epochs()
    saved_checkpoints: list[tuple[int, Path]] = []

    for epoch in epochs:
        config = _build_config(epoch)
        print(f"\nTraining {MODEL_NAME} model for epoch={epoch}...")
        engine = WordEmbeddingSearchEngine(config)
        engine.fit(dataset.train.texts, dataset.train.labels)

        use_query_expansion = MODEL_NAME.lower() == "word2vec"
        result = evaluate_engine(
            engine,
            query_texts=query_texts,
            query_labels=query_labels,
            k_values=K_VALUES,
            use_query_expansion=use_query_expansion,
        )

        checkpoint_path = _save_checkpoint(engine, config, epoch, result)
        saved_checkpoints.append((epoch, checkpoint_path))

        _print_eval_result(epoch, result)
        print(f"Saved checkpoint: {checkpoint_path}")

    best_epoch, best_path = max(saved_checkpoints, key=lambda item: item[0])
    print("\nTraining complete.")
    print(f"Highest epoch checkpoint: epoch={best_epoch}")
    print(f"Path: {best_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import pickle
import random
import re
import sys
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
from search_engine.engine import SearchEngineConfig
from search_engine.evaluation import evaluate_engine


# Editable constants
MODEL_NAME = "word2vec"  # folder name under models/, e.g. "word2vec" or "bert"
DATASET_NAME = "sh0416/ag_news"
QUERY_TEXT = "Oil prices are rising"
K_VALUES = [20]

# Optional constants
MAX_TRAIN = 50000
MAX_TEST = None
EVAL_QUERIES = 1000
SEED = 42
TOP_K_FOR_DIRECT_SEARCH = 20
USE_QUERY_EXPANSION = True


def _sample_query_indices(total: int, n: int, seed: int) -> list[int]:
    n = min(total, n)
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    return indices[:n]


def _extract_epoch(path: Path) -> int:
    match = re.match(r"epoch_(\d+)\.pkl$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def _latest_checkpoint(model_name: str) -> Path:
    model_dir = PROJECT_ROOT / "models" / model_name.lower()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    checkpoints = [path for path in model_dir.glob("epoch_*.pkl") if path.is_file()]
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {model_dir}. Run train_and_save_models.py first."
        )

    best_path = max(checkpoints, key=_extract_epoch)
    if _extract_epoch(best_path) < 0:
        raise RuntimeError(f"No valid epoch_* checkpoints found in {model_dir}.")
    return best_path


def _rebuild_indexes(engine) -> None:
    # Rebuild ANN backends after unpickling; vectors remain available in the checkpoint.
    if engine.doc_index is not None and engine.doc_index._vectors is not None:
        engine.doc_index.build(engine.doc_index._vectors)

    if engine.term_index is not None and engine.term_index._vectors is not None:
        engine.term_index.build(engine.term_index._vectors)


def _restore_runtime_state(engine, checkpoint_payload: dict[str, object]) -> None:
    config_dict = checkpoint_payload.get("config")
    if isinstance(config_dict, dict):
        config = SearchEngineConfig(**config_dict)
        engine.config = config

        if config.embedding_backend.lower() == "bert":
            from search_engine.embeddings import BertSentenceEmbedder

            engine.bert_embedder = BertSentenceEmbedder(config.bert_model_name)

    _rebuild_indexes(engine)


def _print_saved_metrics(payload: dict[str, object]) -> None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return

    precision = metrics.get("precision_at_k", {})
    recall = metrics.get("recall_at_k", {})
    ndcg = metrics.get("ndcg_at_k", {})
    map_score = metrics.get("map_score")
    queries_requested = metrics.get("queries_requested", "?")
    queries_evaluated = metrics.get("queries_evaluated", "?")
    queries_skipped = metrics.get("queries_skipped", "?")

    print("\nMetrics recorded at training time")
    print(f"Queries requested: {queries_requested}")
    print(f"Queries evaluated: {queries_evaluated}")
    print(f"Queries skipped:   {queries_skipped}")
    if isinstance(map_score, (int, float)):
        print(f"MAP: {float(map_score):.4f}")

    if isinstance(precision, dict) and isinstance(recall, dict):
        for key in sorted(precision, key=int):
            p_value = precision[key]
            r_value = recall.get(key, 0.0)
            n_value = ndcg.get(key, 0.0) if isinstance(ndcg, dict) else 0.0
            print(
                f"  P@{key}: {float(p_value):.4f} | "
                f"R@{key}: {float(r_value):.4f} | "
                f"nDCG@{key}: {float(n_value):.4f}"
            )


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    checkpoint_path = _latest_checkpoint(MODEL_NAME)
    trained_epoch = _extract_epoch(checkpoint_path)
    print(f"Loading highest epoch checkpoint: {checkpoint_path.name} (epoch={trained_epoch})")

    with checkpoint_path.open("rb") as file:
        payload = pickle.load(file)

    if not isinstance(payload, dict) or "engine" not in payload:
        raise RuntimeError("Checkpoint format is invalid. Missing engine object.")

    engine = payload["engine"]
    _restore_runtime_state(engine, payload)

    _print_saved_metrics(payload)

    print("\nLoading dataset for fresh evaluation...")
    dataset = load_text_classification_dataset(
        dataset_name=DATASET_NAME,
        max_train=MAX_TRAIN,
        max_test=MAX_TEST,
    )

    query_indices = _sample_query_indices(len(dataset.test.texts), EVAL_QUERIES, SEED)
    query_texts = [dataset.test.texts[index] for index in query_indices]
    query_labels = [dataset.test.labels[index] for index in query_indices]

    use_query_expansion = USE_QUERY_EXPANSION and engine.supports_query_expansion()
    result = evaluate_engine(
        engine,
        query_texts=query_texts,
        query_labels=query_labels,
        k_values=K_VALUES,
        use_query_expansion=use_query_expansion,
    )

    print("\nFresh evaluation on current constants")
    print(f"Queries requested: {result.queries_requested}")
    print(f"Queries evaluated: {result.queries_evaluated}")
    print(f"Queries skipped:   {result.queries_skipped}")
    print(f"MAP: {result.map_score:.4f}")
    for k in sorted(result.precision_at_k):
        precision = result.precision_at_k[k]
        recall = result.recall_at_k[k]
        ndcg = result.ndcg_at_k[k]
        print(f"  P@{k}: {precision:.4f} | R@{k}: {recall:.4f} | nDCG@{k}: {ndcg:.4f}")

    top_k = max(1, TOP_K_FOR_DIRECT_SEARCH)
    print(f"\nDirect search for text: {QUERY_TEXT}")
    results = engine.search(
        QUERY_TEXT,
        k=top_k,
        use_query_expansion=use_query_expansion,
    )

    label_names = dataset.label_names
    for rank, item in enumerate(results, start=1):
        label_name = label_names.get(item.label, str(item.label))
        print(
            f"  {rank:>2}. doc_id={item.doc_id} label={item.label} ({label_name}) "
            f"score={item.score:.4f} text={item.text[:120].replace(os.linesep, ' ')}"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

from collections import defaultdict
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import streamlit as st


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


PROJECT_ROOT = Path(__file__).resolve().parent
_prefer_local_venv_packages(PROJECT_ROOT)
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from search_engine.data import load_text_classification_dataset
from search_engine.engine import SearchEngineConfig
from search_engine.evaluation import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


DEFAULT_QUERY = "Iran and Israel conflict news"
DEFAULT_K = 10
DEFAULT_USE_QUERY_EXPANSION = True


def _extract_epoch(path: Path) -> int:
    match = re.match(r"epoch_(\d+)\.pkl$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def _available_model_names(models_root: Path) -> list[str]:
    if not models_root.exists():
        return []

    model_names: list[str] = []
    for model_dir in sorted(path for path in models_root.iterdir() if path.is_dir()):
        has_checkpoint = any(model_dir.glob("epoch_*.pkl"))
        if has_checkpoint:
            model_names.append(model_dir.name)

    return model_names


def _latest_checkpoint_path(model_name: str) -> Path:
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


def _rebuild_indexes(engine: Any) -> None:
    if engine.doc_index is not None and engine.doc_index._vectors is not None:
        engine.doc_index.build(engine.doc_index._vectors)

    if engine.term_index is not None and engine.term_index._vectors is not None:
        engine.term_index.build(engine.term_index._vectors)


def _restore_runtime_state(engine: Any, payload: dict[str, object]) -> None:
    config_dict = payload.get("config")
    if isinstance(config_dict, dict):
        config = SearchEngineConfig(**config_dict)
        engine.config = config

        if config.embedding_backend.lower() == "bert":
            from search_engine.embeddings import BertSentenceEmbedder

            engine.bert_embedder = BertSentenceEmbedder(config.bert_model_name)

    _rebuild_indexes(engine)


def _to_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric_dict(metric_obj: object) -> dict[int, float]:
    if not isinstance(metric_obj, dict):
        return {}

    normalized: dict[int, float] = {}
    for key, value in metric_obj.items():
        try:
            k = int(key)
            normalized[k] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def _resolve_query_label(
    query_text: str,
    dataset,
    fallback_label: int | None,
) -> tuple[int | None, str]:
    normalized_query = _normalize_text(query_text)
    if not normalized_query:
        return None, "query is empty"

    for text, label in zip(dataset.test.texts, dataset.test.labels):
        if _normalize_text(text) == normalized_query:
            return int(label), "exact match in test split"

    for text, label in zip(dataset.train.texts, dataset.train.labels):
        if _normalize_text(text) == normalized_query:
            return int(label), "exact match in train split"

    if fallback_label is not None:
        return int(fallback_label), "inferred from top retrieved result"

    return None, "no matching label found"


def _compute_query_metrics(
    engine: Any,
    dataset,
    query_text: str,
    retrieved_ids: list[int],
    k: int,
    fallback_label: int | None,
) -> dict[str, object]:
    query_label, label_source = _resolve_query_label(
        query_text=query_text,
        dataset=dataset,
        fallback_label=fallback_label,
    )

    if query_label is None:
        return {
            "available": False,
            "message": "Could not determine a label for this query, so query-level metrics cannot be computed.",
        }

    relevant_ids = engine.relevant_doc_ids(query_label)
    if not relevant_ids:
        return {
            "available": False,
            "message": "No relevant documents exist for the resolved query label.",
        }

    return {
        "available": True,
        "query_label": query_label,
        "label_name": dataset.label_names.get(query_label, str(query_label)),
        "label_source": label_source,
        "k": int(k),
        "precision": precision_at_k(relevant_ids, retrieved_ids, int(k)),
        "recall": recall_at_k(relevant_ids, retrieved_ids, int(k)),
        "ndcg": ndcg_at_k(relevant_ids, retrieved_ids, int(k)),
        "ap": average_precision_at_k(relevant_ids, retrieved_ids, int(k)),
        "relevant_count": len(relevant_ids),
    }


def _short_title(text: str, max_len: int = 110) -> str:
    normalized = " ".join(str(text).split())
    if not normalized:
        return "Untitled result"
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3] + "..."


@st.cache_data(show_spinner=False)
def _load_train_string_columns(
    dataset_name: str,
    max_train: int | None,
) -> dict[str, object]:
    try:
        from datasets import DatasetDict, load_dataset
    except Exception as exc:
        raise RuntimeError("Failed to import datasets package for description loading.") from exc

    dataset = load_dataset(dataset_name)
    if not isinstance(dataset, DatasetDict):
        raise RuntimeError(f"Expected DatasetDict for {dataset_name}, got {type(dataset)!r}.")

    if "train" in dataset:
        train_split = dataset["train"]
    else:
        first_split_name = next(iter(dataset.keys()))
        train_split = dataset[first_split_name]

    if max_train is not None and max_train > 0 and max_train < len(train_split):
        train_split = train_split.select(range(max_train))

    if len(train_split) == 0:
        return {"string_columns": [], "columns": {}}

    sample = train_split[0]
    string_columns: list[str] = []
    for column_name in train_split.column_names:
        if isinstance(sample.get(column_name), str):
            string_columns.append(column_name)

    columns: dict[str, list[str]] = {}
    for column_name in string_columns:
        columns[column_name] = [
            "" if value is None else str(value)
            for value in train_split[column_name]
        ]

    return {
        "string_columns": string_columns,
        "columns": columns,
    }


def _attach_descriptions_from_dataset(engine: Any, payload: dict[str, object]) -> str:
    if getattr(engine, "_descriptions_loaded_from_dataset", False):
        return str(getattr(engine, "_description_status", "Descriptions already loaded."))

    if not getattr(engine, "train_texts", None):
        engine.set_doc_descriptions(None)
        status = "Descriptions unavailable: engine has no training texts."
        setattr(engine, "_descriptions_loaded_from_dataset", True)
        setattr(engine, "_description_status", status)
        return status

    dataset_name = str(payload.get("dataset_name", "")).strip()
    if not dataset_name:
        engine.set_doc_descriptions(None)
        status = "Descriptions unavailable: checkpoint has no dataset_name."
        setattr(engine, "_descriptions_loaded_from_dataset", True)
        setattr(engine, "_description_status", status)
        return status

    payload_max_train = _to_optional_int(payload.get("max_train"))
    data = _load_train_string_columns(dataset_name, payload_max_train)
    string_columns = data.get("string_columns", [])
    columns = data.get("columns", {})

    if not isinstance(string_columns, list) or not isinstance(columns, dict):
        engine.set_doc_descriptions(None)
        status = "Descriptions unavailable: failed to read train split string columns."
        setattr(engine, "_descriptions_loaded_from_dataset", True)
        setattr(engine, "_description_status", status)
        return status

    description_column = None
    for column_name in string_columns:
        if str(column_name).lower() == "description":
            description_column = column_name
            break

    if description_column is None:
        engine.set_doc_descriptions(None)
        status = "Descriptions unavailable: no description column found in dataset."
        setattr(engine, "_descriptions_loaded_from_dataset", True)
        setattr(engine, "_description_status", status)
        return status

    candidate_title_columns = [
        column_name
        for column_name in string_columns
        if column_name != description_column
    ]

    if not candidate_title_columns:
        engine.set_doc_descriptions(None)
        status = "Descriptions unavailable: no title/text column found to align descriptions."
        setattr(engine, "_descriptions_loaded_from_dataset", True)
        setattr(engine, "_description_status", status)
        return status

    engine_text_keys = {_normalize_text(text) for text in engine.train_texts}

    best_title_column = candidate_title_columns[0]
    best_match_count = -1
    for candidate in candidate_title_columns:
        values = columns.get(candidate, [])
        match_count = sum(1 for value in values if _normalize_text(value) in engine_text_keys)
        if match_count > best_match_count:
            best_match_count = match_count
            best_title_column = candidate

    title_values = columns.get(best_title_column, [])
    description_values = columns.get(description_column, [])

    description_map: dict[str, list[str]] = defaultdict(list)
    for title_value, description_value in zip(title_values, description_values):
        description_map[_normalize_text(title_value)].append(description_value)

    aligned_descriptions: list[str] = []
    matched_titles = 0
    non_empty_descriptions = 0
    for train_text in engine.train_texts:
        key = _normalize_text(train_text)
        bucket = description_map.get(key)
        if bucket:
            description = bucket.pop(0)
            matched_titles += 1
        else:
            description = ""

        if description.strip():
            non_empty_descriptions += 1
        aligned_descriptions.append(description)

    engine.set_doc_descriptions(aligned_descriptions)

    status = (
        f"Descriptions loaded from dataset column '{description_column}' aligned with "
        f"title column '{best_title_column}'. "
        f"Matched titles: {matched_titles}/{len(engine.train_texts)}; "
        f"non-empty descriptions: {non_empty_descriptions}/{len(engine.train_texts)}."
    )
    setattr(engine, "_descriptions_loaded_from_dataset", True)
    setattr(engine, "_description_status", status)
    return status


@st.cache_resource(show_spinner=False)
def _load_engine_payload(model_name: str) -> tuple[Any, dict[str, object], Path, int]:
    checkpoint_path = _latest_checkpoint_path(model_name)
    with checkpoint_path.open("rb") as file:
        payload = pickle.load(file)

    if not isinstance(payload, dict) or "engine" not in payload:
        raise RuntimeError("Checkpoint format is invalid. Missing engine object.")

    engine = payload["engine"]
    _restore_runtime_state(engine, payload)
    epoch = _extract_epoch(checkpoint_path)
    return engine, payload, checkpoint_path, epoch


@st.cache_resource(show_spinner=False)
def _load_dataset(dataset_name: str, max_train: int | None, max_test: int | None):
    return load_text_classification_dataset(
        dataset_name=dataset_name,
        max_train=max_train,
        max_test=max_test,
    )


def _render_query_metrics(query_metrics: dict[str, object]) -> None:
    if not query_metrics.get("available"):
        st.info(str(query_metrics.get("message", "Query-level metrics are unavailable.")))
        return

    query_label = int(query_metrics["query_label"])
    label_name = str(query_metrics["label_name"])
    label_source = str(query_metrics["label_source"])
    k = int(query_metrics["k"])
    relevant_count = int(query_metrics["relevant_count"])

    st.caption(
        f"Label used: {query_label} ({label_name}) | source: {label_source} | "
        f"relevant documents: {relevant_count}"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"P@{k}", f"{float(query_metrics['precision']):.4f}")
    c2.metric(f"R@{k}", f"{float(query_metrics['recall']):.4f}")
    c3.metric(f"nDCG@{k}", f"{float(query_metrics['ndcg']):.4f}")
    c4.metric(f"AP@{k}", f"{float(query_metrics['ap']):.4f}")


def _render_model_metrics(metrics: dict[str, object]) -> None:
    precision = _metric_dict(metrics.get("precision_at_k"))
    recall = _metric_dict(metrics.get("recall_at_k"))
    ndcg = _metric_dict(metrics.get("ndcg_at_k"))
    map_score = _to_float(metrics.get("map_score"), default=0.0)

    queries_requested = str(metrics.get("queries_requested", "?"))
    queries_evaluated = str(metrics.get("queries_evaluated", "?"))
    queries_skipped = str(metrics.get("queries_skipped", "?"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Queries requested", queries_requested)
    c2.metric("Queries evaluated", queries_evaluated)
    c3.metric("Queries skipped", queries_skipped)
    c4.metric("MAP", f"{map_score:.4f}")

    all_k = sorted(set(precision) | set(recall) | set(ndcg))
    if not all_k:
        st.info("No per-k model metrics are available in this checkpoint.")
        return

    rows = []
    for k in all_k:
        rows.append(
            {
                "k": k,
                "Precision@k": f"{precision.get(k, 0.0):.4f}",
                "Recall@k": f"{recall.get(k, 0.0):.4f}",
                "nDCG@k": f"{ndcg.get(k, 0.0):.4f}",
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Semantic Search App", layout="wide")
    st.title("Semantic Search App")
    st.caption(
        "Select a saved model, type your query, and choose k. "
        "Click each retrieved text to open its description."
    )

    model_names = _available_model_names(PROJECT_ROOT / "models")
    if not model_names:
        st.error("No saved model checkpoints found in the models folder.")
        st.stop()

    top_left, top_mid, top_right = st.columns([2, 5, 1])
    with top_left:
        selected_model = st.selectbox("Model", options=model_names)
    with top_mid:
        query_text = st.text_input("Query", value=DEFAULT_QUERY)
    with top_right:
        k_value = st.number_input("k", min_value=1, max_value=100, value=DEFAULT_K, step=1)

    try:
        with st.spinner("Loading selected model checkpoint..."):
            engine, payload, checkpoint_path, epoch = _load_engine_payload(selected_model)
    except Exception as exc:  # pragma: no cover - UI error path
        st.error(f"Failed to load model: {exc}")
        st.stop()

    st.caption(f"Loaded checkpoint: {checkpoint_path.name} (epoch={epoch})")

    try:
        with st.spinner("Loading descriptions from dataset..."):
            description_status = _attach_descriptions_from_dataset(engine, payload)
    except Exception as exc:  # pragma: no cover - UI error path
        description_status = f"Descriptions could not be attached from dataset: {exc}"
    st.caption(description_status)

    supports_query_expansion = bool(engine.supports_query_expansion())
    use_query_expansion = st.checkbox(
        "Use query expansion",
        value=DEFAULT_USE_QUERY_EXPANSION,
        disabled=not supports_query_expansion,
    )
    if not supports_query_expansion:
        st.info("Query expansion is only available for Word2Vec checkpoints.")

    if "last_search_output" not in st.session_state:
        st.session_state.last_search_output = None

    if st.button("Search", type="primary", use_container_width=True):
        clean_query = query_text.strip()
        if not clean_query:
            st.warning("Please enter a query before searching.")
        else:
            with st.spinner("Retrieving results..."):
                detailed_results = engine.search_with_description(
                    clean_query,
                    k=int(k_value),
                    use_query_expansion=use_query_expansion,
                )

            retrieved_ids = [entry.item.doc_id for entry in detailed_results]
            fallback_label = detailed_results[0].item.label if detailed_results else None

            dataset_name = str(payload.get("dataset_name", "sh0416/ag_news"))
            payload_max_train = _to_optional_int(payload.get("max_train"))
            payload_max_test = _to_optional_int(payload.get("max_test"))

            try:
                dataset = _load_dataset(dataset_name, payload_max_train, payload_max_test)
                query_metrics = _compute_query_metrics(
                    engine=engine,
                    dataset=dataset,
                    query_text=clean_query,
                    retrieved_ids=retrieved_ids,
                    k=int(k_value),
                    fallback_label=fallback_label,
                )
            except Exception as exc:  # pragma: no cover - UI error path
                query_metrics = {
                    "available": False,
                    "message": f"Could not compute query metrics: {exc}",
                }

            st.session_state.last_search_output = {
                "query": clean_query,
                "model": selected_model,
                "results": detailed_results,
                "query_metrics": query_metrics,
            }

    st.divider()
    st.subheader("Metrics for Loaded Model")
    model_metrics = payload.get("metrics")
    if isinstance(model_metrics, dict):
        st.caption("Metrics saved with the loaded highest-epoch checkpoint")
        _render_model_metrics(model_metrics)
    else:
        st.info("No model-level metrics are available in this checkpoint.")

    search_output = st.session_state.get("last_search_output")
    if not search_output or search_output.get("model") != selected_model:
        st.divider()
        st.subheader("Metrics for This Query")
        st.info("Run a search to compute query-level metrics for your current query.")
        return

    st.subheader(f"Top {len(search_output['results'])} retrieved texts")
    st.caption(f"Query: {search_output['query']}")

    results = search_output["results"]
    if not results:
        st.info("No results found.")
    else:
        for entry in results:
            item = entry.item
            expander_title = f"{entry.rank}. {_short_title(item.text)}"
            with st.expander(expander_title, expanded=False):
                st.write(entry.description)
                st.caption("Retrieved title")
                st.write(item.text)
                st.caption(
                    f"doc_id={item.doc_id} | label={item.label} | score={item.score:.4f}"
                )

    st.divider()
    st.subheader("Metrics for This Query")
    _render_query_metrics(search_output["query_metrics"])


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from datasets import Dataset


@dataclass(frozen=True)
class CorpusSplit:
    texts: list[str]
    labels: list[int]


@dataclass(frozen=True)
class LoadedTextDataset:
    train: CorpusSplit
    test: CorpusSplit
    label_names: dict[int, str]
    text_column: str
    label_column: str


def _import_hf_datasets() -> tuple[type, type, object]:
    try:
        from datasets import Dataset, DatasetDict, load_dataset

        return Dataset, DatasetDict, load_dataset
    except KeyboardInterrupt as exc:
        raise RuntimeError(
            "Import of Hugging Face 'datasets' was interrupted. "
            "Please re-run and avoid interrupting during first-time dependency load."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Failed to import 'datasets'. Ensure dependencies are installed in the local .venv."
        ) from exc


def _subset(split: Dataset, max_items: Optional[int]) -> Dataset:
    if max_items is None or max_items >= len(split):
        return split
    return split.select(range(max_items))


def _infer_columns(split: Dataset) -> Tuple[str, str]:
    sample = split[0]
    text_column = "text" if "text" in split.column_names else None
    label_column = "label" if "label" in split.column_names else None

    if text_column is None:
        for name, value in sample.items():
            if isinstance(value, str):
                text_column = name
                break

    if label_column is None:
        for name, value in sample.items():
            if isinstance(value, int):
                label_column = name
                break

    if text_column is None or label_column is None:
        raise ValueError(
            "Could not infer text and label columns. Please ensure your dataset has one string column and one integer label column."
        )

    return text_column, label_column


def _to_corpus_split(split: Dataset, text_column: str, label_column: str) -> CorpusSplit:
    texts = [str(record[text_column]) for record in split]
    labels = [int(record[label_column]) for record in split]
    return CorpusSplit(texts=texts, labels=labels)


def _get_label_names(split: Dataset, label_column: str) -> Dict[int, str]:
    feature = split.features.get(label_column)
    if feature is not None and hasattr(feature, "names") and feature.names:
        return {idx: str(name) for idx, name in enumerate(feature.names)}

    unique_labels = sorted({int(record[label_column]) for record in split})
    return {label: str(label) for label in unique_labels}


def load_text_classification_dataset(
    dataset_name: str,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
) -> LoadedTextDataset:
    Dataset, DatasetDict, load_dataset = _import_hf_datasets()

    dataset = load_dataset(dataset_name)
    if not isinstance(dataset, DatasetDict):
        raise ValueError(f"Expected DatasetDict from {dataset_name}, got {type(dataset)!r}.")

    if "train" in dataset:
        train_split = dataset["train"]
    else:
        first_split_name = next(iter(dataset.keys()))
        train_split = dataset[first_split_name]

    if "test" in dataset:
        test_split = dataset["test"]
    elif "validation" in dataset:
        test_split = dataset["validation"]
    else:
        split_names = list(dataset.keys())
        if len(split_names) < 2:
            raise ValueError("Dataset must contain at least two splits for indexing and evaluation.")
        test_split = dataset[split_names[1]]

    text_column, label_column = _infer_columns(train_split)

    train_subset = _subset(train_split, max_train)
    test_subset = _subset(test_split, max_test)

    return LoadedTextDataset(
        train=_to_corpus_split(train_subset, text_column, label_column),
        test=_to_corpus_split(test_subset, text_column, label_column),
        label_names=_get_label_names(train_split, label_column),
        text_column=text_column,
        label_column=label_column,
    )

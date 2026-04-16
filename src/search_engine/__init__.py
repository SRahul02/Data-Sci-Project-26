"""Semantic search engine components for AG News experiments.

This module intentionally uses lazy attribute loading so importing
`search_engine` does not eagerly import heavy optional dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "CorpusSplit",
    "LoadedTextDataset",
    "SearchEngineConfig",
    "SearchResult",
    "WordEmbeddingSearchEngine",
    "EvaluationResult",
    "load_text_classification_dataset",
    "evaluate_engine",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CorpusSplit": (".data", "CorpusSplit"),
    "LoadedTextDataset": (".data", "LoadedTextDataset"),
    "load_text_classification_dataset": (".data", "load_text_classification_dataset"),
    "SearchEngineConfig": (".engine", "SearchEngineConfig"),
    "SearchResult": (".engine", "SearchResult"),
    "WordEmbeddingSearchEngine": (".engine", "WordEmbeddingSearchEngine"),
    "EvaluationResult": (".evaluation", "EvaluationResult"),
    "evaluate_engine": (".evaluation", "evaluate_engine"),
}

if TYPE_CHECKING:
    from .data import CorpusSplit, LoadedTextDataset, load_text_classification_dataset
    from .engine import SearchEngineConfig, SearchResult, WordEmbeddingSearchEngine
    from .evaluation import EvaluationResult, evaluate_engine


def __getattr__(name: str) -> Any:
    export = _LAZY_EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = export
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

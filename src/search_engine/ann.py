from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple
import warnings

import numpy as np

_NN_DESCENT_CLASS: Any | None = None
_NN_DESCENT_IMPORT_ERROR: BaseException | None = None


def _load_nndescent_class() -> Any | None:
    global _NN_DESCENT_CLASS, _NN_DESCENT_IMPORT_ERROR

    if _NN_DESCENT_CLASS is not None:
        return _NN_DESCENT_CLASS
    if _NN_DESCENT_IMPORT_ERROR is not None:
        return None

    try:
        from pynndescent import NNDescent as _NNDescent

        _NN_DESCENT_CLASS = _NNDescent
        return _NN_DESCENT_CLASS
    except BaseException as exc:
        _NN_DESCENT_IMPORT_ERROR = exc
        return None


@dataclass
class ANNConfig:
    enabled: bool = False
    n_neighbors: int = 30
    metric: str = "cosine"
    random_state: int = 42


class ANNVectorIndex:
    def __init__(self, vector_size: int, config: ANNConfig | None = None) -> None:
        self.vector_size = vector_size
        self.config = config or ANNConfig()
        self._vectors: np.ndarray | None = None
        self._index: Any | None = None

    def build(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.vector_size:
            raise ValueError(
                f"Expected vectors with shape (n, {self.vector_size}), got {vectors.shape}."
            )

        self._vectors = vectors.astype(np.float32)
        if len(self._vectors) < 2:
            self._index = None
            return

        if not self.config.enabled:
            self._index = None
            return

        nn_descent_class = _load_nndescent_class()
        if nn_descent_class is None:
            reason = _NN_DESCENT_IMPORT_ERROR
            warnings.warn(
                "PyNNDescent import unavailable/interrupted; falling back to exact search. "
                f"Reason: {reason}",
                RuntimeWarning,
            )
            self._index = None
            return

        n_neighbors = min(max(2, self.config.n_neighbors), len(self._vectors) - 1)
        try:
            self._index = nn_descent_class(
                self._vectors,
                n_neighbors=n_neighbors,
                metric=self.config.metric,
                random_state=self.config.random_state,
            )
            self._index.prepare()
        except KeyboardInterrupt as exc:
            warnings.warn(
                "NNDescent initialization interrupted; falling back to exact search. "
                f"Reason: {exc}",
                RuntimeWarning,
            )
            self._index = None
        except Exception as exc:
            warnings.warn(
                "NNDescent failed to initialize; falling back to exact search. "
                f"Reason: {exc}",
                RuntimeWarning,
            )
            self._index = None

    def _exact_query(self, vector: np.ndarray, k: int) -> Tuple[list[int], list[float]]:
        assert self._vectors is not None

        if self.config.metric == "cosine":
            distances = 1.0 - (self._vectors @ vector)
        else:
            differences = self._vectors - vector.reshape(1, -1)
            distances = np.linalg.norm(differences, axis=1)

        ids = np.argsort(distances)[:k]
        return ids.tolist(), distances[ids].astype(float).tolist()

    def query(self, vector: np.ndarray, k: int) -> Tuple[list[int], list[float]]:
        if self._vectors is None:
            raise RuntimeError("ANN index has not been built yet.")

        if vector.shape[0] != self.vector_size:
            raise ValueError(
                f"Expected query vector of length {self.vector_size}, got {vector.shape[0]}."
            )

        k = min(k, len(self._vectors))
        if k <= 0:
            return [], []

        if self._index is None:
            return self._exact_query(vector, k)

        try:
            ids, distances = self._index.query(
                vector.reshape(1, -1).astype(np.float32),
                k=k,
            )
            return ids[0].tolist(), distances[0].astype(float).tolist()
        except KeyboardInterrupt as exc:
            warnings.warn(
                "NNDescent query interrupted; falling back to exact search. "
                f"Reason: {exc}",
                RuntimeWarning,
            )
            self._index = None
            return self._exact_query(vector, k)
        except Exception as exc:
            warnings.warn(
                "NNDescent query failed; falling back to exact search. "
                f"Reason: {exc}",
                RuntimeWarning,
            )
            self._index = None
            return self._exact_query(vector, k)

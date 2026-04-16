from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from pynndescent import NNDescent


@dataclass
class ANNConfig:
    n_neighbors: int = 30
    metric: str = "cosine"
    random_state: int = 42


class ANNVectorIndex:
    def __init__(self, vector_size: int, config: ANNConfig | None = None) -> None:
        self.vector_size = vector_size
        self.config = config or ANNConfig()
        self._vectors: np.ndarray | None = None
        self._index: NNDescent | None = None

    def build(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.vector_size:
            raise ValueError(
                f"Expected vectors with shape (n, {self.vector_size}), got {vectors.shape}."
            )

        self._vectors = vectors.astype(np.float32)
        if len(self._vectors) < 2:
            self._index = None
            return

        n_neighbors = min(max(2, self.config.n_neighbors), len(self._vectors) - 1)
        self._index = NNDescent(
            self._vectors,
            n_neighbors=n_neighbors,
            metric=self.config.metric,
            random_state=self.config.random_state,
        )
        self._index.prepare()

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

        ids, distances = self._index.query(
            vector.reshape(1, -1).astype(np.float32),
            k=k,
        )
        return ids[0].tolist(), distances[0].astype(float).tolist()

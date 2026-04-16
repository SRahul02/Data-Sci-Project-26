from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from gensim.models import Word2Vec

from .text_utils import tokenize


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (vectors / norms).astype(np.float32)


@dataclass
class Word2VecConfig:
    vector_size: int = 200
    window: int = 5
    min_count: int = 2
    workers: int = 4
    epochs: int = 8


class Word2VecEmbedder:
    def __init__(self, config: Word2VecConfig) -> None:
        self.config = config
        self._model: Word2Vec | None = None

    @property
    def vector_size(self) -> int:
        return self.config.vector_size

    def train(self, texts: Sequence[str]) -> None:
        tokenized = [tokenize(text) for text in texts]
        tokenized = [tokens for tokens in tokenized if tokens]
        if not tokenized:
            raise ValueError("No valid tokens found to train Word2Vec.")

        self._model = Word2Vec(
            sentences=tokenized,
            vector_size=self.config.vector_size,
            window=self.config.window,
            min_count=self.config.min_count,
            workers=self.config.workers,
            sg=1,
            epochs=self.config.epochs,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = [self.encode_text(text) for text in texts]
        return np.vstack(vectors).astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Word2Vec model is not trained.")

        tokens = tokenize(text)
        token_vectors = [self._model.wv[token] for token in tokens if token in self._model.wv]

        if not token_vectors:
            return np.zeros(self.config.vector_size, dtype=np.float32)

        matrix = np.asarray(token_vectors, dtype=np.float32)
        return matrix.mean(axis=0).astype(np.float32)

    def has_token(self, token: str) -> bool:
        if self._model is None:
            return False
        return token in self._model.wv

    def token_vector(self, token: str) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Word2Vec model is not trained.")
        return np.asarray(self._model.wv[token], dtype=np.float32)

    def vocabulary(self) -> list[str]:
        if self._model is None:
            return []
        return list(self._model.wv.key_to_index.keys())


class BertSentenceEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embedding_backend='bert'. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(model_name)
        self.vector_size = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode([text], batch_size=1)[0]

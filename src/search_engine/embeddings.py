from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence
import warnings

import numpy as np

from .text_utils import tokenize

_WORD2VEC_CLASS: Any | None = None
_WORD2VEC_IMPORT_ERROR: BaseException | None = None


def _load_gensim_word2vec_class() -> Any | None:
    global _WORD2VEC_CLASS, _WORD2VEC_IMPORT_ERROR

    if _WORD2VEC_CLASS is not None:
        return _WORD2VEC_CLASS
    if _WORD2VEC_IMPORT_ERROR is not None:
        return None

    try:
        from gensim.models import Word2Vec as _Word2Vec

        _WORD2VEC_CLASS = _Word2Vec
        return _WORD2VEC_CLASS
    except BaseException as exc:
        _WORD2VEC_IMPORT_ERROR = exc
        return None


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
    seed: int = 42


class Word2VecEmbedder:
    def __init__(self, config: Word2VecConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._fallback_vectors: dict[str, np.ndarray] = {}

    @property
    def vector_size(self) -> int:
        return self.config.vector_size

    def train(self, texts: Sequence[str]) -> None:
        tokenized = [tokenize(text) for text in texts]
        tokenized = [tokens for tokens in tokenized if tokens]
        if not tokenized:
            raise ValueError("No valid tokens found to train Word2Vec.")

        word2vec_class = _load_gensim_word2vec_class()
        if word2vec_class is None:
            warnings.warn(
                "Gensim Word2Vec import unavailable/interrupted; using deterministic fallback vectors. "
                f"Reason: {_WORD2VEC_IMPORT_ERROR}",
                RuntimeWarning,
            )
            self._train_fallback_vectors(tokenized)
            return

        try:
            self._model = word2vec_class(
                sentences=tokenized,
                vector_size=self.config.vector_size,
                window=self.config.window,
                min_count=self.config.min_count,
                workers=self.config.workers,
                sg=1,
                epochs=self.config.epochs,
                seed=self.config.seed,
            )
            self._fallback_vectors = {}
        except KeyboardInterrupt as exc:
            warnings.warn(
                "Word2Vec training was interrupted; using deterministic fallback vectors. "
                f"Reason: {exc}",
                RuntimeWarning,
            )
            self._train_fallback_vectors(tokenized)
        except Exception as exc:
            warnings.warn(
                "Word2Vec training failed; using deterministic fallback vectors. "
                f"Reason: {exc}",
                RuntimeWarning,
            )
            self._train_fallback_vectors(tokenized)

    def _train_fallback_vectors(self, tokenized_texts: Sequence[Sequence[str]]) -> None:
        self._model = None
        counts = Counter(token for tokens in tokenized_texts for token in tokens)
        self._fallback_vectors = {}

        for token, count in counts.items():
            if count < self.config.min_count:
                continue

            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            token_seed = int.from_bytes(digest, byteorder="little", signed=False) ^ self.config.seed
            rng = np.random.default_rng(token_seed)
            vector = rng.standard_normal(self.config.vector_size).astype(np.float32)
            self._fallback_vectors[token] = vector

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = [self.encode_text(text) for text in texts]
        return np.vstack(vectors).astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        if self._model is None and not self._fallback_vectors:
            raise RuntimeError("Word2Vec model is not trained.")

        tokens = tokenize(text)
        if self._model is not None:
            token_vectors = [self._model.wv[token] for token in tokens if token in self._model.wv]
        else:
            token_vectors = [self._fallback_vectors[token] for token in tokens if token in self._fallback_vectors]

        if not token_vectors:
            return np.zeros(self.config.vector_size, dtype=np.float32)

        matrix = np.asarray(token_vectors, dtype=np.float32)
        return matrix.mean(axis=0).astype(np.float32)

    def has_token(self, token: str) -> bool:
        if self._model is not None:
            return token in self._model.wv
        return token in self._fallback_vectors

    def token_vector(self, token: str) -> np.ndarray:
        if self._model is not None:
            return np.asarray(self._model.wv[token], dtype=np.float32)
        if token not in self._fallback_vectors:
            raise RuntimeError("Word2Vec fallback vectors are not available for this token.")
        return np.asarray(self._fallback_vectors[token], dtype=np.float32)

    def vocabulary(self) -> list[str]:
        if self._model is not None:
            return list(self._model.wv.key_to_index.keys())
        return list(self._fallback_vectors.keys())


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
        if hasattr(self.model, "get_embedding_dimension"):
            self.vector_size = int(self.model.get_embedding_dimension())
        else:
            self.vector_size = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return vectors.astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode([text], batch_size=1)[0]

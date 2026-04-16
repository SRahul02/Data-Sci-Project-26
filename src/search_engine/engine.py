from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .ann import ANNConfig, ANNVectorIndex
from .embeddings import BertSentenceEmbedder, Word2VecConfig, Word2VecEmbedder, l2_normalize
from .text_utils import tokenize


@dataclass
class SearchEngineConfig:
    embedding_backend: str = "word2vec"
    use_ann: bool = False
    ann_n_neighbors: int = 30
    ann_random_state: int = 42
    expansion_per_term: int = 3
    w2v_vector_size: int = 200
    w2v_window: int = 5
    w2v_min_count: int = 2
    w2v_workers: int = 4
    w2v_epochs: int = 8
    w2v_seed: int = 42
    bert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class SearchResult:
    doc_id: int
    label: int
    score: float
    text: str


class WordEmbeddingSearchEngine:
    def __init__(self, config: SearchEngineConfig) -> None:
        self.config = config

        self.train_texts: list[str] = []
        self.train_labels: list[int] = []
        self.label_to_doc_ids: dict[int, set[int]] = defaultdict(set)

        self.word_embedder: Word2VecEmbedder | None = None
        self.bert_embedder: BertSentenceEmbedder | None = None

        self.doc_index: ANNVectorIndex | None = None
        self.term_index: ANNVectorIndex | None = None
        self.vocab_terms: list[str] = []

    def fit(self, texts: list[str], labels: list[int]) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length.")
        if not texts:
            raise ValueError("At least one training document is required.")

        self.train_texts = list(texts)
        self.train_labels = list(labels)

        self.label_to_doc_ids = defaultdict(set)
        for doc_id, label in enumerate(self.train_labels):
            self.label_to_doc_ids[label].add(doc_id)

        backend = self.config.embedding_backend.lower()
        if backend not in {"word2vec", "bert"}:
            raise ValueError("embedding_backend must be one of: word2vec, bert")

        if backend == "word2vec":
            self.word_embedder = Word2VecEmbedder(
                Word2VecConfig(
                    vector_size=self.config.w2v_vector_size,
                    window=self.config.w2v_window,
                    min_count=self.config.w2v_min_count,
                    workers=self.config.w2v_workers,
                    epochs=self.config.w2v_epochs,
                    seed=self.config.w2v_seed,
                )
            )
            self.word_embedder.train(self.train_texts)
            doc_vectors = self.word_embedder.encode(self.train_texts)
            self._build_term_index()
            self.bert_embedder = None
        else:
            self.bert_embedder = BertSentenceEmbedder(self.config.bert_model_name)
            doc_vectors = self.bert_embedder.encode(self.train_texts)
            self.word_embedder = None
            self.term_index = None
            self.vocab_terms = []

        normalized_doc_vectors = l2_normalize(doc_vectors)
        self.doc_index = ANNVectorIndex(
            vector_size=normalized_doc_vectors.shape[1],
            config=ANNConfig(
                enabled=self.config.use_ann,
                n_neighbors=self.config.ann_n_neighbors,
                metric="cosine",
                random_state=self.config.ann_random_state,
            ),
        )
        self.doc_index.build(normalized_doc_vectors)

    def _build_term_index(self) -> None:
        if self.word_embedder is None:
            self.term_index = None
            self.vocab_terms = []
            return

        self.vocab_terms = self.word_embedder.vocabulary()
        if not self.vocab_terms:
            self.term_index = None
            return

        term_vectors = np.vstack(
            [self.word_embedder.token_vector(token) for token in self.vocab_terms]
        ).astype(np.float32)
        term_vectors = l2_normalize(term_vectors)

        term_neighbors = max(10, self.config.ann_n_neighbors // 2)
        self.term_index = ANNVectorIndex(
            vector_size=term_vectors.shape[1],
            config=ANNConfig(
                enabled=self.config.use_ann,
                n_neighbors=term_neighbors,
                metric="cosine",
                random_state=self.config.ann_random_state,
            ),
        )
        self.term_index.build(term_vectors)

    def supports_query_expansion(self) -> bool:
        return (
            self.config.embedding_backend.lower() == "word2vec"
            and self.term_index is not None
            and self.word_embedder is not None
        )

    def expand_query_terms(self, query: str) -> list[str]:
        if not self.supports_query_expansion():
            return []
        if self.config.expansion_per_term <= 0:
            return []

        assert self.word_embedder is not None
        assert self.term_index is not None

        tokens = tokenize(query)
        seen_terms = set(tokens)
        expanded_terms: list[str] = []

        for token in tokens:
            if not self.word_embedder.has_token(token):
                continue

            token_vec = self.word_embedder.token_vector(token)
            token_vec = l2_normalize(token_vec.reshape(1, -1))[0]

            # Ask for a few extra neighbors because we drop duplicates and the original token.
            neighbor_ids, _ = self.term_index.query(token_vec, self.config.expansion_per_term + 3)

            added_for_token = 0
            for idx in neighbor_ids:
                candidate = self.vocab_terms[idx]
                if candidate in seen_terms:
                    continue

                expanded_terms.append(candidate)
                seen_terms.add(candidate)
                added_for_token += 1

                if added_for_token >= self.config.expansion_per_term:
                    break

        return expanded_terms

    def _query_vector(self, query: str, use_query_expansion: bool) -> np.ndarray:
        backend = self.config.embedding_backend.lower()
        if backend == "word2vec":
            if self.word_embedder is None:
                raise RuntimeError("Word2Vec embedder is not initialized. Call fit() first.")

            if use_query_expansion and self.supports_query_expansion():
                expanded_terms = self.expand_query_terms(query)
                merged_query = " ".join([query] + expanded_terms)
            else:
                merged_query = query

            query_vector = self.word_embedder.encode_text(merged_query)
        else:
            if self.bert_embedder is None:
                raise RuntimeError("BERT embedder is not initialized. Call fit() first.")
            query_vector = self.bert_embedder.encode_text(query)

        return l2_normalize(query_vector.reshape(1, -1))[0]

    def search(
        self,
        query: str,
        k: int = 10,
        use_query_expansion: bool = True,
    ) -> list[SearchResult]:
        if self.doc_index is None:
            raise RuntimeError("Document index not built. Call fit() first.")
        if k <= 0:
            return []

        query_vector = self._query_vector(query, use_query_expansion=use_query_expansion)
        k = min(k, len(self.train_texts))
        doc_ids, distances = self.doc_index.query(query_vector, k=k)

        results: list[SearchResult] = []
        for doc_id, distance in zip(doc_ids, distances):
            cosine_similarity = 1.0 - distance
            cosine_similarity = float(np.clip(cosine_similarity, -1.0, 1.0))
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    label=self.train_labels[doc_id],
                    score=cosine_similarity,
                    text=self.train_texts[doc_id],
                )
            )

        return results

    def relevant_doc_ids(self, label: int) -> set[int]:
        return set(self.label_to_doc_ids.get(label, set()))

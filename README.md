# Word Embedding Search Engine on AG News

This project builds a semantic search engine for the `sh0416/ag_news` dataset using:

- Word embeddings (Word2Vec or BERT sentence embeddings)
- Approximate nearest neighbor (ANN) retrieval with PyNNDescent
- Query expansion using nearest neighbor terms (Word2Vec mode)
- Standard retrieval metrics: Precision@k and Recall@k

## What this implements

1. Train embeddings on AG News text
2. Build ANN index over document vectors
3. Expand queries using nearest neighbor terms from a word-level ANN index
4. Retrieve top-k documents for each query
5. Evaluate retrieval quality with Precision@k and Recall@k

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_experiment.py --dataset sh0416/ag_news --embedding word2vec --max-train 30000 --eval-queries 1000 --k 5 10 20
```

For BERT sentence embeddings:

```bash
python scripts/run_experiment.py --dataset sh0416/ag_news --embedding bert --max-train 30000 --eval-queries 1000 --k 5 10 20
```

## Evaluation setup

- Index corpus: train split
- Query corpus: sampled documents from test split
- Relevance definition: train documents with the same class label as query label

This gives a consistent label-based relevance benchmark on AG News.

## Project structure

- `src/search_engine/data.py`: dataset loading and split handling
- `src/search_engine/embeddings.py`: Word2Vec and BERT embedding backends
- `src/search_engine/ann.py`: PyNNDescent ANN wrappers
- `src/search_engine/engine.py`: search engine + query expansion
- `src/search_engine/evaluation.py`: Precision@k and Recall@k
- `scripts/run_experiment.py`: end-to-end experiment runner

## Notes

- Word2Vec mode supports query expansion with nearest-neighbor terms.
- BERT mode gives strong semantic document embeddings but does not do token-level expansion.
- PyNNDescent uses graph-based approximate nearest neighbor retrieval and works well on Windows without compiler toolchains.

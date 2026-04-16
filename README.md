# Word Embedding Search Engine (AG News)

This repository implements a full retrieval pipeline for AG News using word embeddings and approximate nearest-neighbor search.

It is designed to satisfy the assignment requirement from `question.txt`:

1. Build a word-embedding based search engine
2. Use Word2Vec or BERT embeddings
3. Use ANN retrieval
4. Evaluate using Precision@k and Recall@k

## Features

- Word2Vec retrieval backend
- BERT sentence-embedding retrieval backend
- ANN retrieval using PyNNDescent
- Word2Vec query expansion using nearest-neighbor terms
- Precision@k and Recall@k evaluation
- Query accounting in evaluation output:
	- Queries requested
	- Queries evaluated
	- Queries skipped

## Repository Layout

- `scripts/run_experiment.py`: End-to-end CLI runner
- `src/search_engine/data.py`: Dataset loading and split/column inference
- `src/search_engine/embeddings.py`: Word2Vec and BERT embedding models
- `src/search_engine/ann.py`: ANN index wrapper and exact-search fallback
- `src/search_engine/engine.py`: Search engine fit/search/query-expansion logic
- `src/search_engine/evaluation.py`: Precision@k and Recall@k computation
- `AUDIT_AND_TEST_RESULTS.md`: Local audit log and run results

## Setup

### 1) Create virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Linux/macOS:

```bash
python3 -m venv .venv
```

### 2) Activate virtual environment

Windows PowerShell:

```powershell
& ".\.venv\Scripts\Activate.ps1"
```

If activation is blocked by PowerShell policy:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
& ".\.venv\Scripts\Activate.ps1"
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Optional: Hugging Face login (recommended)

For higher rate limits and faster dataset/model access:

```bash
huggingface-cli login
```

## Quick Start

### Word2Vec default experiment

```bash
python scripts/run_experiment.py --embedding word2vec --max-train 30000 --eval-queries 1000 --k 5 10 20
```

### BERT experiment

```bash
python scripts/run_experiment.py --embedding bert --max-train 30000 --eval-queries 1000 --k 5 10 20
```

### Fast smoke test (recommended first run)

```bash
python scripts/run_experiment.py --embedding word2vec --max-train 1200 --max-test 220 --eval-queries 40 --k 5 10 20 --show-examples 1
```

## CLI Reference

Main options from `scripts/run_experiment.py`:

- `--dataset` (default: `sh0416/ag_news`)
- `--embedding` (`word2vec` or `bert`)
- `--max-train` (positive int)
- `--max-test` (positive int, optional)
- `--eval-queries` (positive int)
- `--k` (one or more positive ints, example: `--k 5 10 20`)
- `--seed` (global seed for sampling and NumPy)
- `--show-examples` (`>= 0`)

ANN options:

- `--ann-n-neighbors` (`>= 2`)
- `--ann-random-state`

Query expansion options:

- `--expansion-per-term` (`>= 0`)
	- `0` disables expansion in Word2Vec mode

Word2Vec options:

- `--w2v-vector-size`
- `--w2v-window`
- `--w2v-min-count`
- `--w2v-workers`
- `--w2v-epochs`
- `--w2v-seed` (defaults to `--seed` when not set)

BERT option:

- `--bert-model-name` (default: `sentence-transformers/all-MiniLM-L6-v2`)

## How Evaluation Works

- Index corpus: train split
- Query corpus: sampled items from test split
- Relevance rule: a train document is relevant if its class label equals the query label

Metrics shown:

- `Precision@k`
- `Recall@k`

Query accounting shown for transparency:

- `Queries requested`
- `Queries evaluated`
- `Queries skipped`

If `Queries skipped > 0`, metrics were computed on the evaluated subset only.

## Example Command Recipes

### Compare Word2Vec with and without expansion impact

```bash
python scripts/run_experiment.py --embedding word2vec --max-train 5000 --max-test 1200 --eval-queries 300 --k 5 10 20 --show-examples 0
python scripts/run_experiment.py --embedding word2vec --max-train 5000 --max-test 1200 --eval-queries 300 --k 5 10 20 --show-examples 0 --expansion-per-term 0
```

### BERT smoke run

```bash
python scripts/run_experiment.py --embedding bert --max-train 300 --max-test 80 --eval-queries 20 --k 5 10 20 --show-examples 0
```

### Reproducible Word2Vec run

```bash
python scripts/run_experiment.py --embedding word2vec --max-train 5000 --max-test 1200 --eval-queries 300 --seed 42 --w2v-seed 42
```

## Troubleshooting

### 1) PowerShell activation blocked

Error mentions `Activate.ps1 cannot be loaded` and execution policy.

Fix:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
```

### 2) `ModuleNotFoundError: sentence_transformers`

Install into your active venv:

```bash
pip install sentence-transformers
```

### 3) Hugging Face warning about unauthenticated requests

Login with token:

```bash
huggingface-cli login
```

### 4) BERT run is slow

- First run downloads model files
- CPU-only inference is slower
- Use smaller `--max-train`, `--max-test`, and `--eval-queries` for quick checks

### 5) All metrics are 0.0 on small runs

Check query accounting. If all queries are skipped, your sampled train split may not contain relevant labels for sampled test queries.

## Reproducibility Notes

- Word2Vec randomness is controlled by `--w2v-seed`.
- Query sampling and NumPy randomness are controlled by `--seed`.
- ANN has `--ann-random-state`.
- Exact bitwise reproducibility can still vary across OS/hardware/library versions.

## Known Limitations

1. Relevance is label-based and may not perfectly reflect semantic relevance judgments.
2. Query expansion can hurt quality on some subsets/hyperparameters.
3. If ANN backend fails internally, the code falls back to exact search with warnings.

## Validation Artifacts

See `AUDIT_AND_TEST_RESULTS.md` for:

- Audit findings
- Commands executed locally
- Runtime and metric outputs
- Pass/fail outcomes

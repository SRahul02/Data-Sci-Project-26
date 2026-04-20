# Word Embedding Search Engine (AG News)

This repository implements a semantic retrieval pipeline for AG News using word embeddings.
It supports Word2Vec and BERT backends, exact or ANN retrieval, checkpoint save/load,
and an interactive Streamlit app.

The project covers the assignment requirements:

1. Build a word-embedding based search engine
2. Use Word2Vec or BERT embeddings
3. Use ANN retrieval
4. Evaluate retrieval quality

## What Is Implemented

- Word2Vec retrieval backend
- BERT sentence-embedding retrieval backend
- Exact search (default) and optional ANN with PyNNDescent
- Word2Vec query expansion using nearest-neighbor terms
- Retrieval metrics:
	- Precision@k
	- Recall@k
	- nDCG@k
	- MAP
- Query accounting in evaluation output:
	- Queries requested
	- Queries evaluated
	- Queries skipped
- Saved checkpoint workflow:
	- train and save models by epoch
	- reload and evaluate saved checkpoint
- Streamlit app for interactive querying with result descriptions and query-level metrics

## Repository Layout

- `app.py`: Streamlit app to load saved checkpoints and run interactive search
- `scripts/run_experiment.py`: End-to-end CLI experiment runner
- `scripts/train_and_save_models.py`: Train model(s) and save epoch checkpoints
- `scripts/evaluate_saved_model.py`: Load highest checkpoint and evaluate/search
- `models/word2vec/`: Saved Word2Vec checkpoints (`epoch_XXX.pkl`)
- `models/bert/`: Saved BERT checkpoints (`epoch_XXX.pkl`)
- `src/search_engine/data.py`: Dataset loading and split/column inference
- `src/search_engine/embeddings.py`: Word2Vec/BERT embedding logic and normalization
- `src/search_engine/ann.py`: ANN wrapper with exact-search fallback
- `src/search_engine/engine.py`: Search engine fit/search/query-expansion logic
- `src/search_engine/evaluation.py`: Evaluation metrics and query accounting
- `results.txt`: Example output from a run
- `FILE_WISE_EXPLANATION.txt`: Detailed file-wise project explanation

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

### A) Word2Vec experiment (CLI)

```bash
python scripts/run_experiment.py --embedding word2vec --max-train 30000 --eval-queries 1000 --k 5 10 20
```

### B) BERT experiment (CLI)

```bash
python scripts/run_experiment.py --embedding bert --max-train 30000 --eval-queries 1000 --k 5 10 20
```

### C) ANN-enabled run (CLI)

```bash
python scripts/run_experiment.py --embedding word2vec --use-ann --ann-n-neighbors 30 --max-train 10000 --eval-queries 400 --k 5 10 20
```

### D) Train and save checkpoints

Edit constants at the top of `scripts/train_and_save_models.py`, then run:

```bash
python scripts/train_and_save_models.py
```

### E) Evaluate saved checkpoint

Edit constants at the top of `scripts/evaluate_saved_model.py`, then run:

```bash
python scripts/evaluate_saved_model.py
```

### F) Launch Streamlit app

```bash
streamlit run app.py
```

The app loads the highest available checkpoint for the selected model and shows:

- Model-level metrics saved in checkpoint
- Query-level metrics (P@k, R@k, nDCG@k, AP@k)
- Expandable top-k result cards with descriptions (when available)

## CLI Reference (`scripts/run_experiment.py`)

Core options:

- `--dataset` (default: `sh0416/ag_news`)
- `--embedding` (`word2vec` or `bert`)
- `--max-train` (positive int)
- `--max-test` (positive int, optional)
- `--eval-queries` (positive int)
- `--k` (one or more positive ints, example: `--k 5 10 20`)
- `--seed` (global seed for sampling and NumPy)
- `--show-examples` (`>= 0`)

ANN options:

- `--use-ann` (enables ANN; otherwise exact search is used)
- `--ann-n-neighbors` (`>= 2`)
- `--ann-random-state`

Query expansion option:

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

## Evaluation Protocol

- Index corpus: train split
- Query corpus: sampled test split items
- Relevance definition: train document is relevant if label matches query label

Reported metrics:

- `Precision@k`
- `Recall@k`
- `nDCG@k`
- `MAP`

Query accounting:

- `Queries requested`
- `Queries evaluated`
- `Queries skipped`

If `Queries skipped > 0`, all averages are computed over evaluated queries only.

In Word2Vec mode, `run_experiment.py` prints both:

- without query expansion
- with query expansion

and reports metric deltas between them.

## Example Command Recipes

### Fast smoke test

```bash
python scripts/run_experiment.py --embedding word2vec --max-train 1200 --max-test 220 --eval-queries 40 --k 5 10 20 --show-examples 1
```

### Compare Word2Vec with and without expansion

```bash
python scripts/run_experiment.py --embedding word2vec --max-train 5000 --max-test 1200 --eval-queries 300 --k 5 10 20 --show-examples 0
python scripts/run_experiment.py --embedding word2vec --max-train 5000 --max-test 1200 --eval-queries 300 --k 5 10 20 --show-examples 0 --expansion-per-term 0
```

### Compare exact vs ANN

```bash
python scripts/run_experiment.py --embedding word2vec --max-train 10000 --eval-queries 400 --k 5 10 20
python scripts/run_experiment.py --embedding word2vec --use-ann --ann-n-neighbors 30 --max-train 10000 --eval-queries 400 --k 5 10 20
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

Install into active virtual environment:

```bash
pip install sentence-transformers
```

### 3) Hugging Face warning about unauthenticated requests

Login with token:

```bash
huggingface-cli login
```

### 4) Streamlit app or saved-model evaluator says no checkpoints found

Run training first to create checkpoints:

```bash
python scripts/train_and_save_models.py
```

### 5) BERT run is slow

- First run downloads model files
- CPU-only inference is slower
- Use smaller `--max-train`, `--max-test`, and `--eval-queries` for quick checks

### 6) Metrics become 0.0 on tiny subsets

Check query accounting. If all sampled queries are skipped, your train subset may not include
relevant labels for sampled test queries.

### 7) Descriptions not visible in app

The app can show fallback text when dataset description alignment is unavailable.
This does not block search results or metrics.

## Reproducibility Notes

- Word2Vec randomness is controlled by `--w2v-seed`
- Query sampling and NumPy randomness are controlled by `--seed`
- ANN has `--ann-random-state`
- Exact bitwise reproducibility may still vary across OS/hardware/library versions

## Known Limitations

1. Relevance is label-based, not human-judged semantic relevance.
2. Query expansion can improve or hurt quality depending on subset/hyperparameters.
3. If ANN backend import/build/query fails, code falls back to exact search with warning.

## Useful Artifacts

- `results.txt`: Sample run logs and retrieval outputs
- `FILE_WISE_EXPLANATION.txt`: Simple file-by-file explanation of the full project

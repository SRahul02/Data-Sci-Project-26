# Audit And Local Test Results

Date: 2026-04-16
Workspace: Data-Sci-Project-26

## 1) Assignment Requirements Extracted From question.txt

The question requires:

1. Build a word-embedding based search engine.
2. Use Word2Vec or BERT for embeddings.
3. Use approximate nearest neighbor algorithms for retrieval.
4. Evaluate with Precision@k and Recall@k.

## 2) Codebase Context (What The Project Does)

Current implementation provides:

- Dataset loading from Hugging Face AG News via `sh0416/ag_news`.
- Embedding backends:
  - Word2Vec (token average sentence representation)
  - BERT sentence embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- ANN indexing and querying via PyNNDescent.
- Query expansion in Word2Vec mode using nearest-neighbor terms from a term ANN index.
- Retrieval scoring via cosine similarity.
- Evaluation via Precision@k and Recall@k using label-based relevance.

## 3) Audit Checklist Against Requirements

- Word embeddings implemented: PASS
- Both Word2Vec and BERT paths implemented: PASS
- ANN retrieval implemented: PASS
- Precision@k and Recall@k implemented: PASS
- End-to-end runnable script provided: PASS

## 4) Fixes Applied During Audit

The following correctness/robustness changes were applied:

1. ANN stability fallback already in place
- If NNDescent initialization/query fails, code now falls back to exact search with warning.

2. Better argument validation in `scripts/run_experiment.py`
- Validates positive integer constraints for core arguments.
- Validates `--k` values are positive.
- Prevents invalid values such as `--k 0`.

3. Better evaluation transparency in `src/search_engine/evaluation.py`
- Added:
  - `queries_requested`
  - `queries_evaluated`
  - `queries_skipped`
- Makes skipped-query behavior explicit in outputs.

4. Query expansion disable logic fixed in `src/search_engine/engine.py`
- `--expansion-per-term 0` now truly disables term expansion.

5. Reproducibility improvements
- Added Word2Vec seed in config and wiring:
  - `Word2VecConfig.seed`
  - `SearchEngineConfig.w2v_seed`
  - CLI support: `--w2v-seed`
- Uses `--seed` by default when `--w2v-seed` is not set.
- Seeds Python and NumPy in runner.

6. BERT embedding compatibility cleanup
- Removed redundant normalization at encode-time; normalization remains centralized in engine flow.
- Updated embedding dimension getter to support newer API (`get_embedding_dimension`) while keeping backward compatibility.

## 5) Local Test Matrix And Results

Interpreter used for project runs:

- `.venv\\Scripts\\python.exe`

### Test A: Word2Vec Smoke

Command:

```powershell
.venv\Scripts\python.exe scripts/run_experiment.py --embedding word2vec --max-train 1200 --max-test 220 --eval-queries 40 --k 5 10 20 --show-examples 1 --seed 42
```

Result:

- Exit code: 0
- Elapsed: 69.79 sec
- Metrics:
  - No expansion: P@5 0.4650, P@10 0.4375, P@20 0.3887; R@5 0.0088, R@10 0.0166, R@20 0.0285
  - With expansion: P@5 0.3150, P@10 0.3100, P@20 0.2937; R@5 0.0061, R@10 0.0118, R@20 0.0216
  - Delta indicates expansion hurt on this sample.

### Test B: Word2Vec With Expansion Disabled (`--expansion-per-term 0`)

Command:

```powershell
.venv\Scripts\python.exe scripts/run_experiment.py --embedding word2vec --max-train 1200 --max-test 220 --eval-queries 40 --k 5 10 20 --show-examples 0 --expansion-per-term 0 --seed 42
```

Result:

- Exit code: 0
- Elapsed: 76.18 sec
- Metrics with and without expansion are identical (all deltas 0.0000), confirming the fix works.

### Test C: BERT Smoke (Before Dependency Fix)

Command:

```powershell
.venv\Scripts\python.exe scripts/run_experiment.py --embedding bert --max-train 300 --max-test 80 --eval-queries 20 --k 5 10 20 --show-examples 0 --seed 42
```

Initial result:

- Exit code: 1
- Error: `ModuleNotFoundError: No module named 'sentence_transformers'`

Action taken:

```powershell
.venv\Scripts\python.exe -m pip install sentence-transformers
```

### Test D: BERT Smoke (After Dependency Fix)

Command:

```powershell
.venv\Scripts\python.exe scripts/run_experiment.py --embedding bert --max-train 300 --max-test 80 --eval-queries 20 --k 5 10 20 --show-examples 0 --seed 42
```

Result:

- Exit code: 0
- Elapsed: 271.98 sec
- Metrics:
  - P@5 0.8000, P@10 0.7778, P@20 0.7389
  - R@5 0.0226, R@10 0.0443, R@20 0.0823
- Query accounting:
  - Requested: 20
  - Evaluated: 9
  - Skipped: 11

### Test E: Tiny Corpus + Large k

Command:

```powershell
.venv\Scripts\python.exe scripts/run_experiment.py --embedding word2vec --max-train 30 --max-test 20 --eval-queries 10 --k 5 50 --show-examples 0 --seed 42
```

Result:

- Exit code: 0
- Elapsed: 67.85 sec
- Query accounting:
  - Requested: 10
  - Evaluated: 0
  - Skipped: 10
- All metrics are 0.0, which is now clearly explained by skipped-query reporting.

### Test F: Invalid Argument Validation

Command:

```powershell
.venv\Scripts\python.exe scripts/run_experiment.py --k 0
```

Result:

- Exit code: 2
- Correct parser error returned:
  - `--k requires one or more positive integers.`

### Test G: Medium-Scale Word2Vec Benchmark

Command:

```powershell
.venv\Scripts\python.exe scripts/run_experiment.py --embedding word2vec --max-train 5000 --max-test 1200 --eval-queries 300 --k 5 10 20 --show-examples 0 --seed 42
```

Result:

- Exit code: 0
- Elapsed: 1562.94 sec
- Metrics:
  - No expansion: P@5 0.4553, P@10 0.4247, P@20 0.4070; R@5 0.0019, R@10 0.0035, R@20 0.0066
  - With expansion: P@5 0.3980, P@10 0.3967, P@20 0.3703; R@5 0.0016, R@10 0.0032, R@20 0.0060

## 6) Final Audit Conclusion

Status: PASS with improvements applied.

- The project now fully satisfies the assignment requirements.
- Both backend paths (Word2Vec and BERT) are runnable locally in the project venv.
- Evaluation reporting is more transparent and argument validation is safer.
- Local test matrix confirms core functionality and edge-case behavior.

## 7) Important Notes For Interpretation

1. Relevance is label-based, not semantic ground-truth judged.
2. Query expansion can reduce precision/recall depending on corpus and hyperparameters.
3. Small or imbalanced train subsets can produce skipped queries; this is now reported.
4. BERT first-run may be slower due model download and weight initialization.

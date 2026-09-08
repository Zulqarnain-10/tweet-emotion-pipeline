# Proof: where every public number comes from

Every number a reader meets in the README, on the demo page, in the model card, or in the data
card is produced by the pipeline and written to a file. This page maps each one to that file, the
JSON key, and the command that wrote it. The Value column reads `[todo]` until the first
`dvc repro`; the integrator copies each value from the file named on its row, and after a retrain
the files win again. Values that are declared in `params.yaml` or fixed in code are filled in
already.

Commands run from the repository root inside the activated virtual environment. Stage names
refer to `dvc.yaml`; `dvc repro` runs them all in order.

## README, synced block

Rendered by `python -m tweet_emotion.sync_readme`. `--check` exits 1 when the block and the
files disagree. The block itself is the receipt for these values, so this table names the keys
and does not repeat the numbers.

| Number | File | Key | Command |
|---|---|---|---|
| Test rows | `reports/metrics.json` | `n_test` | `python -m tweet_emotion.evaluate` (stage `evaluate`) |
| Share labelled happiness on test | `reports/metrics.json` | `positive_rate_test`, `positive_label` | stage `evaluate` |
| Decision threshold | `reports/metrics.json` | `threshold` (declared in `params.yaml` `evaluate.threshold`) | stage `evaluate` |
| Accuracy, and the majority-baseline accuracy | `reports/metrics.json` | `accuracy`, `majority_baseline_accuracy` | stage `evaluate` |
| Precision, recall, F1 | `reports/metrics.json` | `precision`, `recall`, `f1` | stage `evaluate` |
| ROC-AUC, PR-AUC | `reports/metrics.json` | `roc_auc`, `pr_auc` | stage `evaluate` |
| Brier, log loss | `reports/metrics.json` | `brier`, `log_loss` | stage `evaluate` |
| Shipped model name in the table header | `models/version.json` | `model` | `python -m tweet_emotion.train` (stage `train`) |
| Folds and selection metric | `models/version.json` | `cv_folds`, `selection_metric` (declared in `params.yaml` `train`) | stage `train` |
| Per-candidate cv ROC-AUC mean and std, cv accuracy, fit seconds | `reports/metrics.json` (`candidates_cv`, a copy of `models/version.json` `candidates`) | `candidates_cv.<name>.cv_roc_auc_mean`, `.cv_roc_auc_std`, `.cv_accuracy_mean`, `.fit_seconds` | stage `train`, copied by stage `evaluate` |
| p95 latency, requests, concurrency, requests per second, host | `reports/loadtest.json` | `p95_ms`, `requests`, `concurrency`, `rps`, `host` | `python -m tweet_emotion.loadtest --url http://127.0.0.1:8000 --requests 300 --concurrency 10 --host "local uvicorn, Windows 11, Python 3.12, single process"` |
| Trained at | `models/version.json` | `trained_at` | stage `train` |
| Git sha (short) | `models/version.json` | `git_sha_short` | stage `train` |
| Data sha256 (first 12) | `models/version.json` | `data_sha256` (declared in `params.yaml` `data.sha256`, verified by stage `ingest`) | stage `train` |
| Train and test rows | `reports/metrics.json` | `n_train`, `n_test` (also `models/version.json` `n_train`) | stage `evaluate` |

## README, prose

| Number | Value | File | Key | Command |
|---|---|---|---|---|
| Source CSV bytes | 3,768,210 | `data/source/tweet_emotions.csv` (committed unchanged); echoed in `data/raw/fetch_manifest.json` | `bytes` | `python -c "import os; print(os.path.getsize('data/source/tweet_emotions.csv'))"` |
| Where `ingest` took the CSV from | `repository copy` or `download` | `data/raw/fetch_manifest.json` (rebuilt by `dvc repro`, not committed; copied into `data/DATA_CARD.md`) | `fetched_from` | `python -m tweet_emotion.ingest` (stage `ingest`) |
| Dataset rows | 40,000 | `data/raw/fetch_manifest.json` | `n_rows_total` | stage `ingest` |
| Labels in the full file | 13 | `data/raw/fetch_manifest.json` | `label_counts_total` (13 keys) | stage `ingest` |
| Happiness and sadness rows in the full file | 5,209 and 5,165 | `data/raw/fetch_manifest.json` | `label_counts_total.happiness`, `.sadness` | stage `ingest` |
| Rows kept after the duplicate rule | 10,337 kept, 37 dropped | `data/raw/fetch_manifest.json` | `n_kept`, `n_duplicates_dropped` | stage `ingest` |
| Duplicate texts in the full file (after HTML unescaping, case folding, and whitespace collapsing) | 224 | `data/raw/fetch_manifest.json` | `n_duplicate_texts_total` | stage `ingest` |
| Held-out fraction and seed | 0.2, 42 | `params.yaml` | `data.test_size`, `data.seed` (echoed in the manifest as `test_size`, `seed`) | declared |
| DVC stages | 6 | `dvc.yaml` | stage count | `python -m dvc dag` |
| Stop words in the frozen list | 198 | `configs/stopwords_en.txt` | line count | `python -c "print(sum(1 for _ in open('configs/stopwords_en.txt', encoding='utf-8')))"` |
| Stop words in effect (the list minus the negation tokens in `preprocess.keep_words`) | 174 | `configs/stopwords_en.txt`, `params.yaml` | `preprocess.keep_words`, subtracted by `TextNormalizer.from_params` | `python -c "from tweet_emotion import settings; from tweet_emotion.preprocess import TextNormalizer; print(len(TextNormalizer.from_params(settings.load_params()).stopwords))"` |
| Vectoriser settings | 1-2 grams, 5,000 features, `min_df` 2, sublinear tf | `params.yaml` | `features.ngram_range`, `.max_features`, `.min_df`, `.sublinear_tf` (echoed in `data/features/feature_manifest.json`) | declared |
| Fitted vocabulary size | 5,000 | `data/features/feature_manifest.json` (not committed) and `models/version.json` | `vocabulary_size`; `n_features` | stage `features`, stage `train` |
| Candidates and folds | 3 (`logreg`, `nb`, `xgboost`), 5 | `params.yaml` | `train.candidates`, `train.cv_folds` | declared |
| Decision threshold | 0.5 | `params.yaml` | `evaluate.threshold` | declared |
| Parity check tolerance between raw-text and pre-vectorised paths | 1e-6 | `src/tweet_emotion/evaluate.py` | the assertion in the evaluate stage | code |
| Request text limit | 1,000 characters | `params.yaml` | `api.max_chars` (enforced by `src/tweet_emotion/api/schemas.py`) | declared, checked by `tests/test_api.py` |
| Batch cap | 100 | `params.yaml` | `api.batch_max` | declared, checked by `tests/test_api.py` |
| Body limit | 1,250,000 bytes (1.25 MB, a full batch of 100 texts of 1,000 JSON-escaped characters) | `src/tweet_emotion/api/app.py` | the body-size middleware | code, checked by `tests/test_api.py` |
| Terms returned per prediction | up to 8 | `src/tweet_emotion/api/model_loader.py` | `explain` | code |
| CI tolerance | 0.005, absolute for values at or below 1 and relative for larger values; the four confusion counts within round(0.005 x n_test) samples | `.github/workflows/ci.yml`, `scripts/compare_metrics.py` | `--tolerance 0.005` on `scripts/compare_metrics.py` | CI job `reproduce` |
| Ports | 8000 local, 7860 on the Space | `Dockerfile` (`PORT=8000`), `scripts/deploy_space.py` (the Space port) | | build and deploy |

## Demo page tiles (`src/tweet_emotion/api/static/index.html`)

The page never hard-codes a number; each tile reads an endpoint, and each endpoint reads a file.

| Tile | Endpoint | File | Key |
|---|---|---|---|
| Model version in the eyebrow | `GET /version` | `models/version.json` | `model_version` |
| Disclaimer | `GET /version` | `src/tweet_emotion/settings.py` | `DISCLAIMER` |
| Test split n | `GET /version` | `reports/metrics.json` | `metrics.n_test` |
| Accuracy | `GET /version` | `reports/metrics.json` | `metrics.accuracy` |
| ROC-AUC | `GET /version` | `reports/metrics.json` | `metrics.roc_auc` |
| F1 | `GET /version` | `reports/metrics.json` | `metrics.f1` |
| Trained date | `GET /version` | `reports/metrics.json` | `metrics.trained_at` |
| p95 latency and host | `GET /version` | `reports/loadtest.json` | `loadtest.p95_ms`, `loadtest.host` (chip reads `[todo]` when the file is absent from the build) |
| Decision threshold in the result panel | `GET /version` | `reports/metrics.json` | `metrics.threshold` |
| Preset chips, their descriptions, and the textarea text | `GET /presets` | `configs/presets.json` | `presets[*].title`, `.description`, `.text` |
| Probability, label, threshold, normalised text, version after submit | `POST /predict` | `models/model.joblib`, `reports/metrics.json` | response fields `probability`, `probability_happiness`, `label`, `threshold`, `normalized_text`, `model_version` |
| "Why" bars under the result | `POST /predict` | `models/model.joblib` | response field `terms[*].term`, `.contribution` |
| Top terms, two columns | `GET /terms` | `reports/top_terms.json` | `happiness[*]`, `sadness[*]` (or `terms[*]` with `available` false) |
| Curl payload in "Run it yourself" | `GET /presets` | `configs/presets.json` | `presets[0].text` |

## Model card (`models/MODEL_CARD.md`)

| Section | File | Keys | Command |
|---|---|---|---|
| Provenance table | `models/version.json` | `model`, `model_version`, `package_version`, `git_sha`, `git_dirty`, `trained_at`, `data_sha256`, `n_train`, `n_features`, `selection_metric`, `cv_folds`, `winner_params`, `libraries` | stage `train` |
| Test rows in the provenance table | `reports/metrics.json` | `n_test` | stage `evaluate` |
| Data paragraph | `data/raw/fetch_manifest.json` | `n_kept`, `n_train`, `n_test` | stage `ingest` |
| Vocabulary size and density | `data/features/feature_manifest.json` | `vocabulary_size`, `density_train` | stage `features` |
| Hyperparameters | `params.yaml` | `train.logreg`, `train.nb`, `train.xgboost`, `train.seed` | declared |
| Cross-validation table | `models/version.json` | `candidates.<name>.*`, `model` | stage `train` |
| Metrics table and confusion matrix | `reports/metrics.json` | `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`, `brier`, `log_loss`, `majority_baseline_accuracy`, `confusion.tn`, `.fp`, `.fn`, `.tp`, `n_test`, `positive_rate_test`, `threshold` | stage `evaluate` |
| Latency paragraph | `reports/loadtest.json` | `p95_ms`, `requests`, `concurrency`, `rps`, `error_rate`, `host` | `python -m tweet_emotion.loadtest` |
| Top terms table | `reports/top_terms.json` | `method`, `available`, `happiness[0:5].term`, `sadness[0:5].term` | stage `evaluate` |

## Data card (`data/DATA_CARD.md`)

| Section | Source | Command |
|---|---|---|
| Hash and byte size | `params.yaml` `data.sha256`; `data/source/tweet_emotions.csv` (the committed file); `data/raw/fetch_manifest.json` `sha256`, `bytes` | stage `ingest` |
| Where the CSV came from | `data/raw/fetch_manifest.json` `fetched_from` | stage `ingest` |
| Row count and the 13 label counts | `data/raw/fetch_manifest.json` `n_rows_total`, `label_counts_total` | stage `ingest` |
| Kept rows, duplicates dropped, duplicates in the full file | `data/raw/fetch_manifest.json` `n_kept_before_dedup`, `n_duplicates_dropped`, `n_kept`, `n_duplicate_texts_total` | stage `ingest` |
| Split table | `data/raw/fetch_manifest.json` `n_train`, `n_test`, `positive_rate_train`, `positive_rate_test` | stage `ingest` |
| Columns | `src/tweet_emotion/settings.py` `CSV_COLUMNS`; `params.yaml` `data.text_column`, `.label_column`, `.id_column` | code, checked by `tests/test_ingest.py` |
| Share of rows that became empty after normalisation | the INFO line logged by `python -m tweet_emotion.preprocess` | stage `preprocess` |

## How CI reproduces these numbers

The `reproduce` job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on every
push to main and every pull request:

1. Check out the commit and install `requirements-pipeline.txt` plus the package.
2. Save the committed `reports/metrics.json` aside with `git show HEAD:reports/metrics.json`.
3. Run `python -m tweet_emotion.setup_nltk`, then `python -m dvc repro -f`, which copies the
   committed CSV from `data/source/` into `data/raw` (downloading from `data.url` only if that
   copy were missing), verifies its sha256, and rebuilds every stage from scratch on the runner.
4. Run `python scripts/compare_metrics.py --committed <saved> --fresh reports/metrics.json
   --tolerance 0.005`. Every numeric leaf must agree within the tolerance, which is absolute
   (0.005) for values at or below 1 and relative for larger values; the four confusion counts use
   an absolute tolerance of round(0.005 x n_test) samples. The nested cv record is compared the
   same way except `fit_seconds`; `n_train`, `n_test`, `data_sha256`, `model`, and
   `positive_label` must be identical; `trained_at`, `git_sha`, and `model_version` are ignored
   because they change on every run. Any other difference fails the job.
5. Upload the reproduced `reports/`, `models/version.json`, and `configs/presets.json` as a
   workflow artifact, so the fresh numbers can be read next to the committed ones.

Four more jobs run alongside it: `lint` (ruff), `test`, `gitleaks`, and `docker` (build the
image, start it, score the first preset through `scripts/smoke_live.py`). The `test` job first
runs `python -m tweet_emotion.setup_nltk` and `python -m tweet_emotion.sync_readme --check`, then
rebuilds the data through the `features` stage with `python -m dvc repro features` (the data files
are not committed), then runs pytest with coverage; with `data/features/test.npz` present, the
contract tests in `tests/test_model_contract.py` re-score the shipped model on the test split and
require accuracy and ROC-AUC to be within 1e-4 of `reports/metrics.json` instead of skipping.

Badge: [![CI](https://github.com/Zulqarnain-10/tweet-emotion-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Zulqarnain-10/tweet-emotion-pipeline/actions/workflows/ci.yml)

CI run: [todo: link the first green Actions run]

The README block itself is checked by `python -m tweet_emotion.sync_readme --check`, which exits
1 when the rendered block differs from the file. Run it locally before committing; CI's `test`
job runs it before pytest.

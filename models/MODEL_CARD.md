# Model card: `tweet-emotion-pipeline`

Demonstration system trained on public tweets labelled happiness or sadness by crowd workers in
2016. It scores short English text only and is not a mental-health tool. The same sentence is
returned by the API in every prediction response.

Every number below is copied by hand from `reports/metrics.json`, `models/version.json`,
`reports/top_terms.json`, and `reports/loadtest.json`, the first three written by `dvc repro`
and the last by `python -m tweet_emotion.loadtest`. This card is not rewritten by
`sync_readme`; after a retrain, refresh each `[todo: ...]` chip and each value from the file named
next to it. Metrics are rounded to four decimal places as stored in those files. Nothing is typed
from memory.

## Provenance (`models/version.json`)

| Field | Value |
|---|---|
| Shipped model | `logreg` (the winning candidate, inside a scikit-learn `Pipeline`) |
| Model version | 0.1.0+5d310c0 |
| Package version | 0.1.0 (`package_version`, from `src/tweet_emotion/__init__.py`) |
| Git sha at training time | `5d310c033a1cf8a3021574bf3914405923a27bd4` (`git_dirty`: false) |
| Trained at | 2026-09-08T21:53:52Z |
| Data sha256 | `cbceef78546853a516edb24689bd9a78cc2a2d6bcc5d5cc9a9e3dee6f923c199` (declared in `params.yaml`, verified by `ingest`, copied by `train`) |
| Rows used | 8,269 train, 2,068 test |
| Features seen by the classifier | 5,000 (the fitted vocabulary, at most 5,000 by `params.yaml`) |
| Selection metric and folds | `roc_auc`, 5 (`selection_metric`, `cv_folds`) |
| Winner hyperparameters | `C` 1.0, `max_iter` 1000 |
| Libraries | python 3.12.2, scikit-learn 1.9.0, numpy 2.5.3, pandas 2.3.3, scipy 1.18.1, nltk 3.10.3, xgboost 3.4.1, joblib 1.6.0 |

## Intended use

- Purpose: show a complete, reproducible path from a public CSV to a tested prediction endpoint,
  with every stage tracked by DVC and every number traceable to a file. The prediction is the
  probability that a short English text expresses happiness rather than sadness, plus the label
  at a fixed threshold.
- Users: people reviewing the repository, the demo page, or the API.
- Out of scope: any use on real people's messages for triage, moderation, wellbeing, or
  clinical purposes, in any language. The labels are crowd judgements about 2016 tweets, the
  task is binary by construction, and no review of any kind beyond the tests in this repository
  has been done.

## Data

CrowdFlower "Sentiment Analysis: Emotion in Text", 2016: 40,000 tweets with 13 labels. The
binary task keeps happiness (5,209 rows in the full file, class 1) and sadness (5,165 rows,
class 0), drops duplicate texts (judged after HTML unescaping, case folding, and whitespace
collapsing; first occurrence kept), and splits 80 / 20 stratified on the label with seed 42. The
CSV is committed at `data/source/tweet_emotions.csv` and verified by hash by `ingest`.
Rows after the duplicate rule and per split: 10,337 (8,269 train, 2,068 test). Source, hash,
label counts, quirks, and the license chip: `data/DATA_CARD.md`.

## Preprocessing

The first step of the shipped `Pipeline` is `TextNormalizer` (`src/tweet_emotion/preprocess.py`),
a picklable sklearn transformer configured from the `preprocess` block of `params.yaml`. On one
string, in order: cast to text (NaN becomes the empty string); lower-case; strip URLs; strip
`@mentions`; unescape HTML entities; strip digits; replace punctuation, curly quotes, and the
ellipsis with spaces; collapse whitespace; drop the stop words, which are the NLTK English list
of 198 words frozen in `configs/stopwords_en.txt` minus the negation tokens listed in
`params.yaml` `preprocess.keep_words` (174 in effect), because NLTK's list contains "not" and
dropping it would score "not happy" like "happy"; lemmatise each token with the WordNet lemmatiser (noun default);
join with single spaces. The empty string is a valid result. The stop-word set travels inside
the pickle, so `models/model.joblib` is self-contained; the WordNet corpus is fetched once by
`python -m tweet_emotion.setup_nltk`. The API runs the same object, so the text the model sees
in production is exactly the text it was trained on.

## Features

`TfidfVectorizer` fitted on the training split only and applied unchanged to the test split and
to every request (`params.yaml` `features`): unigrams and bigrams, at most 5,000 features,
`min_df` 2, sublinear term frequency, token pattern `(?u)\b\w+\b` so single-letter tokens
survive, no further lower-casing (the normaliser already did it), `float32` values. The fitted
vectoriser is `models/vectorizer.joblib`; its vocabulary size and the density of the training
matrix are in `data/features/feature_manifest.json` (rebuilt by `dvc repro`, not committed):
vocabulary 5,000, training density 0.0015.

## Candidates and selection rule (`params.yaml` `train`)

Three classifiers are cross-validated on the training matrix with `StratifiedKFold` (5 folds,
shuffled, seed 42), scored on ROC-AUC, accuracy, and F1. The test split plays no part.

| Candidate | Estimator | Hyperparameters |
|---|---|---|
| `logreg` | `LogisticRegression`, `liblinear` solver | `C` 1.0, `max_iter` 1000, seed 42 |
| `nb` | `MultinomialNB` | `alpha` 0.5 |
| `xgboost` | `XGBClassifier`, `hist` tree method, single thread | `n_estimators` 300, `max_depth` 6, `learning_rate` 0.1, `subsample` 0.9, `colsample_bytree` 0.6, `eval_metric` logloss, seed 42 |

Winner rule: the highest mean cv ROC-AUC ships as `models/model.joblib`; a tie goes to the
candidate listed first in `params.yaml` (`logreg`, then `nb`, then `xgboost`). The winner is then
refitted on the whole training split and wrapped with the normaliser and the fitted vectoriser;
the wrapping step does not refit anything.

Cross-validation record that decided the winner (`models/version.json` `candidates`, also
copied into `reports/metrics.json` `candidates_cv`):

| Candidate | cv ROC-AUC mean | cv ROC-AUC std | cv accuracy | cv F1 | Fit seconds | Shipped |
|---|---|---|---|---|---|---|
| `logreg` | 0.8868 | 0.0047 | 0.8037 | 0.8057 | 0.05 | yes |
| `nb` | 0.8685 | 0.004 | 0.782 | 0.7802 | 0.01 | no |
| `xgboost` | 0.8704 | 0.0069 | 0.7876 | 0.7982 | 19.9 | no |

## Metrics (`reports/metrics.json`)

Measured once on the held-out test split, n = 2,068, share labelled happiness 0.5015, at the
decision threshold 0.5 (`params.yaml` `evaluate.threshold`).
The test split was not used for any modelling or selection decision. `evaluate` also checks that
the full pipeline on the raw test texts reproduces the probabilities from the pre-vectorised
matrix to within 1e-6, so these numbers describe the object the API serves.

| Metric | Shipped model | Majority baseline |
|---|---|---|
| Accuracy | 0.8129 | 0.5015 |
| Precision | 0.8113 | |
| Recall | 0.8168 | |
| F1 | 0.814 | |
| ROC-AUC | 0.889 | |
| PR-AUC (average precision) | 0.8909 | |
| Brier score (lower is better) | 0.141 | |
| Log loss (lower is better) | 0.4431 | |

The majority baseline predicts the more common class for every tweet, so only its accuracy is
meaningful; the two kept labels are close in size, so it is near one half.

Confusion matrix at the threshold (`confusion`; happiness is the positive class):

| | Predicted sadness | Predicted happiness |
|---|---|---|
| Actual sadness | TN 834 | FP 197 |
| Actual happiness | FN 190 | TP 847 |

Figures: `reports/figures/roc.png`, `pr.png`, `confusion.png`, and `top_terms.png` (the last
one only when the winner exposes per-term weights).

## Latency (`reports/loadtest.json`)

p95 140.6 ms for `POST /predict`, 300 requests at concurrency 10, 105.71 requests per second,
error rate 0.0, on local uvicorn, Windows 11, Python 3.12, single process. Measured with
`python -m tweet_emotion.loadtest` against the URL recorded in the file; when the host is the
live Space the number includes network time from the client.

## Top terms (`reports/top_terms.json`)

The 20 terms (`params.yaml` `evaluate.top_terms`) that push the score hardest toward each label.
How they are computed depends on the winner: logistic-regression coefficients for `logreg`, the
log-probability ratio `feature_log_prob_[1] - feature_log_prob_[0]` for `nb`, and gain importance
(unsigned, so a single list rather than one per label) for `xgboost`. The file records which
(`method`, `available`). The demo page reads the same file through `GET /terms`, and the API's
`terms` field on each prediction lists the up-to-eight tokens in the request that moved that
particular score, when the winner is linear. Top five per label from the current file:

| Rank | Happiness | Sadness |
|---|---|---|
| 1 | thanks | sad |
| 2 | happy | miss |
| 3 | great | not |
| 4 | haha | suck |
| 5 | good | sorry |

## Limitations

- English only. The stop-word list, the lemmatiser, and the training text are English; other
  languages are scored as if they were English and the output means nothing.
- 2016 tweets. Slang, hashtags, and platform conventions have moved on; the model has no notion
  of anything written since, and the split is random rather than time-based, so nothing here
  measures decay.
- Crowd labels. Each tweet carries one worker-chosen label from a list of thirteen with blurry
  borders; label noise is part of the ceiling on every metric above.
- Binary. A neutral, worried, or angry text is still assigned happiness or sadness with a
  probability; the score says which way the text leans between two options, not whether either
  fits.
- Bag of n-grams. Word order beyond adjacent pairs, sarcasm, and long-range negation are
  invisible to the model.
- The decision threshold 0.5 is a fixed setting from `params.yaml`, not a tuned operating point.

## License

Code: MIT (`LICENSE`). Data: [todo: confirm the CrowdFlower Data for Everyone terms before reuse];
see `data/DATA_CARD.md`.

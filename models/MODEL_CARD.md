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
| Shipped model | [todo: `model`] (the winning candidate, inside a scikit-learn `Pipeline`) |
| Model version | [todo: `model_version`] |
| Package version | 0.1.0 (`package_version`, from `src/tweet_emotion/__init__.py`) |
| Git sha at training time | [todo: `git_sha`] (`git_dirty`: [todo]) |
| Trained at | [todo: `trained_at`] |
| Data sha256 | `cbceef78546853a516edb24689bd9a78cc2a2d6bcc5d5cc9a9e3dee6f923c199` (declared in `params.yaml`, verified by `ingest`, copied by `train`) |
| Rows used | [todo: `n_train`] train, [todo: `n_test` from `reports/metrics.json`] test |
| Features seen by the classifier | [todo: `n_features`] (the fitted vocabulary, at most 5,000 by `params.yaml`) |
| Selection metric and folds | `roc_auc`, 5 (`selection_metric`, `cv_folds`) |
| Winner hyperparameters | [todo: `winner_params`] |
| Libraries | [todo: `libraries`: python, scikit-learn, numpy, pandas, scipy, nltk, xgboost, joblib] |

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
Rows after the duplicate rule and per split: [todo: `n_kept`, `n_train`, `n_test` from
`data/raw/fetch_manifest.json`]. Source, hash, label counts, quirks, and the license chip:
`data/DATA_CARD.md`.

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
vocabulary [todo: `vocabulary_size`], training density [todo: `density_train`].

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
| `logreg` | [todo] | [todo] | [todo] | [todo] | [todo] | [todo: yes or no] |
| `nb` | [todo] | [todo] | [todo] | [todo] | [todo] | [todo: yes or no] |
| `xgboost` | [todo] | [todo] | [todo] | [todo] | [todo] | [todo: yes or no] |

## Metrics (`reports/metrics.json`)

Measured once on the held-out test split, n = [todo: `n_test`], share labelled happiness
[todo: `positive_rate_test`], at the decision threshold 0.5 (`params.yaml` `evaluate.threshold`).
The test split was not used for any modelling or selection decision. `evaluate` also checks that
the full pipeline on the raw test texts reproduces the probabilities from the pre-vectorised
matrix to within 1e-6, so these numbers describe the object the API serves.

| Metric | Shipped model | Majority baseline |
|---|---|---|
| Accuracy | [todo: `accuracy`] | [todo: `majority_baseline_accuracy`] |
| Precision | [todo: `precision`] | |
| Recall | [todo: `recall`] | |
| F1 | [todo: `f1`] | |
| ROC-AUC | [todo: `roc_auc`] | |
| PR-AUC (average precision) | [todo: `pr_auc`] | |
| Brier score (lower is better) | [todo: `brier`] | |
| Log loss (lower is better) | [todo: `log_loss`] | |

The majority baseline predicts the more common class for every tweet, so only its accuracy is
meaningful; the two kept labels are close in size, so it is near one half.

Confusion matrix at the threshold (`confusion`; happiness is the positive class):

| | Predicted sadness | Predicted happiness |
|---|---|---|
| Actual sadness | TN [todo: `confusion.tn`] | FP [todo: `confusion.fp`] |
| Actual happiness | FN [todo: `confusion.fn`] | TP [todo: `confusion.tp`] |

Figures: `reports/figures/roc.png`, `pr.png`, `confusion.png`, and `top_terms.png` (the last
one only when the winner exposes per-term weights).

## Latency (`reports/loadtest.json`)

p95 [todo: `p95_ms`] ms for `POST /predict`, [todo: `requests`] requests at concurrency
[todo: `concurrency`], [todo: `rps`] requests per second, error rate [todo: `error_rate`], on
[todo: `host`]. Measured with `python -m tweet_emotion.loadtest` against the URL recorded in the
file; when the host is the live Space the number includes network time from the client.

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
| 1 | [todo: `happiness[0].term`] | [todo: `sadness[0].term`] |
| 2 | [todo] | [todo] |
| 3 | [todo] | [todo] |
| 4 | [todo] | [todo] |
| 5 | [todo] | [todo] |

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

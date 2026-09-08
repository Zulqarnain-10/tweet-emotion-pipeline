# Data card: CrowdFlower "Sentiment Analysis: Emotion in Text" (2016)

Every number on this page is either declared in `params.yaml` (the source hash) or written by
the `ingest` stage to `data/raw/fetch_manifest.json` (the byte size, the 13 label counts, the
duplicate counts, the kept-row and split facts). Fields that the pipeline has not yet written
render as `[todo: ...]` chips; fill them from the manifest after `dvc repro`, never from memory.

## Source

- Publisher: CrowdFlower (later Figure Eight, now part of Appen), "Sentiment Analysis: Emotion in
  Text", published in the Data for Everyone library in 2016.
- Content: 40,000 English-language tweets, each with one emotion label chosen by crowd workers
  from a fixed list of 13.
- Where the file lives: the full CSV is redistributed unchanged in this repository at
  `data/source/tweet_emotions.csv` (`params.yaml` `data.local_source`), kept byte-identical by a
  `.gitattributes` `-text` rule so its sha256 keeps matching. `data.url` is the repository's own
  raw URL for that file; `ingest` downloads from it only when the committed copy is missing. The
  hash below is what makes either copy trustworthy: a file that does not match it stops the
  pipeline.
- License: the file is redistributed unchanged for reproducibility. The original terms are
  [todo: confirm the CrowdFlower Data for Everyone terms before any reuse beyond this pipeline].
  No license is asserted here until that is checked, and the file will be removed on request.

## Integrity

| File | sha256 | Bytes |
|---|---|---|
| `data/source/tweet_emotions.csv` (committed) and its copy `data/raw/tweet_emotions.csv` (`params.yaml` `data.sha256`) | `cbceef78546853a516edb24689bd9a78cc2a2d6bcc5d5cc9a9e3dee6f923c199` | 3,768,210 |

The `ingest` stage copies the committed file into `data/raw/tweet_emotions.csv` when its sha256
matches the declared value and downloads from `data.url` only when that copy is missing (a
committed copy whose hash does not match is skipped with a warning and the download runs
instead). Either way it then computes the sha256 of the file in `data/raw`, compares it with the
declared value, and raises with both hashes on a mismatch; a copy already in `data/raw` that
already matches is reused. `data/raw/fetch_manifest.json` records which path ran under
`fetched_from` (`repository copy` or `download`). Nothing downstream runs on an unverified file.

## Shape

- 40,000 rows, 3 columns: `tweet_id`, `sentiment`, `content`.
- 13 labels. Counts in the full file:

| Label | Rows |
|---|---|
| neutral | 8,638 |
| worry | 8,459 |
| happiness | 5,209 |
| sadness | 5,165 |
| love | 3,842 |
| surprise | 2,187 |
| fun | 1,776 |
| relief | 1,526 |
| hate | 1,323 |
| empty | 827 |
| enthusiasm | 759 |
| boredom | 179 |
| anger | 110 |

- Duplicate texts in the full file, judged the way the duplicate rule below judges them: 224.

## The binary task

`params.yaml` `data.keep_labels` keeps two of the thirteen labels and `data.positive_label` says
which one is the positive class:

| Label | Class | Rows in the full file |
|---|---|---|
| happiness | 1 | 5,209 |
| sadness | 0 | 5,165 |

That is 10,374 rows before the duplicate rule. From `data/raw/fetch_manifest.json`:

| Field | Value |
|---|---|
| `n_kept_before_dedup` | 10,374 |
| `n_duplicates_dropped` | 37 |
| `n_kept` | 10,337 |

## Duplicate rule

Within the kept rows, each text is compared with the earlier ones after three normalising steps:
HTML entities are unescaped with `html.unescape`, the text is case folded, and runs of whitespace
are collapsed to one space with leading and trailing whitespace dropped. A row whose normalised
text equals an earlier row's normalised text is dropped; the first occurrence is kept, and the
row keeps its original text, not the normalised key. Rows whose text is empty after that are
dropped too. This happens before the split, so the same tweet can never sit in both train and
test. The rule is switched on by `params.yaml` `data.drop_duplicates`; the manifest records how
many kept rows it removed (`n_duplicates_dropped`) and how many such duplicates the full
40,000-row file holds (`n_duplicate_texts_total`).

## Split

Stratified on `label` with `sklearn.model_selection.train_test_split`, `test_size` 0.2, seed 42,
then each split is sorted by `tweet_id` so the CSVs are byte-stable across runs. From
`data/raw/fetch_manifest.json`:

| Split | Rows | Positive rate (happiness) |
|---|---|---|
| train | 8,269 | 0.5014 |
| test | 2,068 | 0.5015 |

The test split is read by `evaluate` (once, for the headline metrics) and by `presets` (to pick
the four demo tweets). Nothing is fitted or tuned on it.

## Columns

Raw file, as published:

| Column | Type | Meaning |
|---|---|---|
| `tweet_id` | int | Identifier assigned by the publisher |
| `sentiment` | str | One of the 13 labels above |
| `content` | str | The tweet text |

Every CSV the pipeline writes (`data/raw/train.csv`, `data/raw/test.csv`,
`data/processed/train.csv`, `data/processed/test.csv`) has exactly the columns in
`settings.CSV_COLUMNS`:

| Column | Type | Meaning |
|---|---|---|
| `tweet_id` | int | Carried through unchanged |
| `text` | str | The tweet text; raw under `data/raw`, normalised under `data/processed` |
| `label` | int | 1 for happiness, 0 for sadness |

## Known quirks

- HTML entities. Texts contain `&quot;`, `&amp;`, `&lt;`, and `&gt;` where the original had
  quotes, ampersands, and angle brackets. The `preprocess` stage unescapes them with
  `html.unescape` before the punctuation rule, so they never survive as tokens.
- Mentions and URLs. Many tweets open with an `@handle` or carry a shortened link. Both are
  stripped by `preprocess` (`@\w+`, `https?://\S+`, `www\.\S+`); the API applies the same
  normaliser, and `configs/presets.json` prefers tweets without either so the demo reads cleanly.
- Crowd labels are noisy. One worker's "happiness" is another's "fun", "love", or
  "enthusiasm", and "sadness" borders "worry" and "empty". Keeping only the two clearest
  labels makes the task tractable, and it also means the model has never seen the other eleven.
- Empty results. Short tweets made only of a mention, a link, numbers, or stop words become the
  empty string after normalisation. They are kept as empty strings (never NaN) and the
  `preprocess` stage logs the share of rows that ended up empty.
- Duplicates. Some rows in the full file repeat an earlier row's text once entities, case, and
  whitespace are normalised (`n_duplicate_texts_total` in the manifest); the duplicate rule above
  removes the ones that fall inside the two kept labels before the split, and
  `n_duplicates_dropped` records how many that was.
- Identifiers. `tweet_id` is a numeric identifier carried in the source file; the model never
  sees it. The text can contain other users' handles, which the normaliser strips before
  anything is modelled or logged.

## Handling and provenance

- The source CSV is committed at `data/source/tweet_emotions.csv` and is a declared dependency
  of the `ingest` stage; derived data is never committed. `data/raw/`, `data/processed/`, and
  `data/features/` are gitignored and managed by DVC; `dvc repro` copies the committed CSV into
  `data/raw` (downloading it only when the copy is missing), verifies it, and rebuilds every
  derived file. The hash above makes a silent change to either copy impossible to miss.
- `data/raw/fetch_manifest.json` records the source URL, where the file came from
  (`fetched_from`: `repository copy` or `download`), the hash, the byte count, the fetch time,
  the total row count, the label counts, the kept labels, the duplicate counts (in the full file
  and among the kept rows), the split sizes, and the positive rate of each split. It is rebuilt
  with the stage and not committed; the tables above are copied from it.
- Downstream files: `data/processed/{train,test}.csv` (normalised text),
  `data/features/{train,test}.npz` (sparse TF-IDF matrices with the label vector),
  `data/features/feature_manifest.json` (vocabulary size, row counts, density),
  `models/vectorizer.joblib` (the fitted vectoriser, committed).
- The API logs one JSON line per prediction with the label, the probability, the character
  count, and the model version. It never logs the text.

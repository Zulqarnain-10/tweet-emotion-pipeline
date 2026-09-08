"""Shared fixtures: a TestClient with the lifespan run, receipts read from the settings paths,
and small synthetic inputs for the pure-function tests.

PREDICTION_LOG_PATH is set before the app is created so no test appends to the machine's
temp directory. Every fixture that needs a pipeline artifact skips, with a reason, when the
file is absent, so the suite runs before and after dvc repro.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from tweet_emotion import settings

WORDNET_HINT = "the WordNet corpus is absent; run python -m tweet_emotion.setup_nltk"

# Vocabulary for the synthetic corpus. The two class lists are disjoint, so a small linear
# model separates them; label noise and a few mixed tweets keep the scores imperfect.
HAPPY_WORDS = (
    "happy",
    "joy",
    "smile",
    "great",
    "wonderful",
    "excited",
    "yay",
    "awesome",
    "laugh",
    "sunshine",
    "celebrate",
    "delighted",
    "cheerful",
    "grin",
    "thrilled",
)
SAD_WORDS = (
    "sad",
    "cry",
    "tears",
    "lonely",
    "hurt",
    "lost",
    "sorry",
    "gloomy",
    "broken",
    "alone",
    "grief",
    "pain",
    "sigh",
    "miserable",
    "heartbroken",
)
NEUTRAL_WORDS = (
    "today",
    "morning",
    "weekend",
    "work",
    "school",
    "coffee",
    "home",
    "friend",
    "night",
    "going",
    "really",
    "time",
    "new",
    "week",
    "house",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def require(path: Path, hint: str) -> None:
    """Skip the calling test when a pipeline artifact is not on disk."""
    if not path.exists():
        pytest.skip(f"{path} is absent; {hint}")


def wordnet_present() -> bool:
    from tweet_emotion.preprocess import find_corpus

    try:
        find_corpus("wordnet")
    except LookupError:
        return False
    return True


def make_raw_frame() -> pd.DataFrame:
    """A CrowdFlower-shaped frame: tweet_id, sentiment, content.

    It covers all thirteen labels, two duplicate texts (one of them only after stripping
    whitespace), a whitespace-only text, URLs, mentions, HTML entities, and numbers.
    """
    rows = [
        (1, "happiness", "I am so happy today!!! :)"),
        (2, "sadness", "I feel so sad and empty today..."),
        (3, "happiness", "  I am so happy today!!! :)  "),
        (4, "sadness", "Lost my keys AGAIN @jenny http://bit.ly/abc123"),
        (5, "neutral", "Just posting a tweet"),
        (6, "worry", "worried about the exam tomorrow"),
        (7, "love", "love you all &lt;3"),
        (8, "surprise", "wow did not expect that"),
        (9, "fun", "this is fun"),
        (10, "relief", "phew, finally done"),
        (11, "hate", "i hate mondays"),
        (12, "empty", ""),
        (13, "enthusiasm", "can't wait for the weekend!"),
        (14, "boredom", "so bored"),
        (15, "anger", "this makes me angry"),
        (16, "happiness", "&quot;Best day ever&quot; said the 3 kids &amp; me"),
        (17, "sadness", "   "),
        (18, "happiness", "Happy birthday @mike!! www.example.com/party 2009"),
        (19, "sadness", "I feel so sad and empty today..."),
        (20, "sadness", "Missing my dog. He was 12 years old."),
        (21, "happiness", "Sunshine and coffee, what a morning"),
        (22, "sadness", "Rainy day, cancelled plans, feeling low"),
        (23, "happiness", "we won the game"),
        (24, "sadness", "cannot stop crying"),
    ]
    return pd.DataFrame(rows, columns=["tweet_id", "sentiment", "content"])


# tweet_ids that survive select_binary on make_raw_frame(): two labels, no duplicates, no
# empty texts. 3 and 19 are duplicates, 17 is whitespace only.
KEPT_IDS = (1, 2, 4, 16, 18, 20, 21, 22, 23, 24)
KEPT_LABELS = {1: 1, 2: 0, 4: 0, 16: 1, 18: 1, 20: 0, 21: 1, 22: 0, 23: 1, 24: 0}


def make_corpus(n: int = 160, seed: int = 0, noise: float = 0.08) -> pd.DataFrame:
    """Synthetic tweets with settings.CSV_COLUMNS and a learnable, imperfect signal.

    Every text is between 6 and 140 characters, carries no @mention and no URL, so all of
    them qualify for the presets stage's readability preference.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = i % 2
        pool = HAPPY_WORDS if label else SAD_WORDS
        other = SAD_WORDS if label else HAPPY_WORDS
        n_class = int(rng.integers(2, 5))
        n_neutral = int(rng.integers(1, 4))
        words = [pool[k] for k in rng.choice(len(pool), size=n_class, replace=False)]
        words += [NEUTRAL_WORDS[k] for k in rng.choice(len(NEUTRAL_WORDS), size=n_neutral)]
        if rng.random() < 0.10:
            words.append(other[int(rng.integers(len(other)))])
        rng.shuffle(words)
        text = " ".join(words)
        if rng.random() < 0.3:
            text = text.capitalize()
        text += ["", "!", "!!", "...", "."][int(rng.integers(5))]
        if rng.random() < noise:
            label = 1 - label
        rows.append({"tweet_id": 1000 + i, "text": text, "label": int(label)})
    return pd.DataFrame(rows, columns=list(settings.CSV_COLUMNS))


def make_toy_params(base: dict) -> dict:
    """The real params with the knobs turned down for tests that run a whole stage."""
    params = copy.deepcopy(base)
    params["preprocess"]["lemmatize"] = False
    params["features"]["max_features"] = 500
    params["features"]["min_df"] = 1
    params["train"]["cv_folds"] = 2
    params["train"]["candidates"] = ["logreg", "nb"]
    params["train"]["xgboost"]["n_estimators"] = 20
    params["evaluate"]["top_terms"] = 5
    params["evaluate"]["figures"] = False
    return params


def fit_toy_pipeline(frame: pd.DataFrame, classifier: str = "logreg"):
    """normalize -> vectorize -> classify, every step fitted, nothing refitted."""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    from tweet_emotion.preprocess import TextNormalizer

    normalizer = TextNormalizer(
        lowercase=True,
        strip_urls=True,
        strip_mentions=True,
        strip_numbers=True,
        strip_punctuation=True,
        remove_stopwords=True,
        lemmatize=False,
    )
    common = {"token_pattern": r"(?u)\b\w+\b", "lowercase": False, "min_df": 1}
    if classifier == "nb":
        vectorizer = CountVectorizer(dtype=np.float32, **common)
        clf = MultinomialNB(alpha=0.5)
    elif classifier == "xgboost":
        from xgboost import XGBClassifier

        vectorizer = TfidfVectorizer(dtype=np.float32, **common)
        clf = XGBClassifier(
            n_estimators=20,
            max_depth=3,
            learning_rate=0.3,
            random_state=0,
            n_jobs=1,
            tree_method="hist",
            eval_metric="logloss",
        )
    else:
        vectorizer = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, **common)
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=0)
    normalized = normalizer.fit(frame["text"]).transform(frame["text"])
    X = vectorizer.fit_transform(normalized)
    clf.fit(X, frame["label"].to_numpy().astype(int))
    return Pipeline([("normalize", normalizer), ("vectorize", vectorizer), ("classify", clf)])


def features_of(pipeline, texts) -> csr_matrix:
    """What the features stage would have stored: the vectoriser output for raw texts."""
    normalized = pipeline.named_steps["normalize"].transform(list(texts))
    return csr_matrix(pipeline.named_steps["vectorize"].transform(normalized))


def write_npz(path: Path, X: csr_matrix, y: np.ndarray) -> None:
    """The npz layout from settings.NPZ_KEYS, written without the features module."""
    X = csr_matrix(X)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": X.data,
        "indices": X.indices,
        "indptr": X.indptr,
        "shape": np.asarray(X.shape),
        "y": np.asarray(y),
    }
    np.savez(path, **{key: payload[key] for key in settings.NPZ_KEYS})


def write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Parameters and synthetic inputs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def params() -> dict:
    return settings.load_params()


@pytest.fixture
def toy_params(params: dict) -> dict:
    return make_toy_params(params)


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture
def tiny_csr() -> tuple[csr_matrix, np.ndarray]:
    dense = np.array(
        [
            [1, 0, 2, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 3, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 4, 0, 0],
        ],
        dtype=np.float32,
    )
    return csr_matrix(dense), np.array([1, 0, 1, 0, 1, 0])


@pytest.fixture(scope="session")
def corpus() -> pd.DataFrame:
    return make_corpus()


@pytest.fixture(scope="session")
def corpus_split(corpus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """120 training rows and 40 held-out rows, both balanced by construction."""
    train = corpus.iloc[:120].reset_index(drop=True)
    test = corpus.iloc[120:].reset_index(drop=True)
    return train, test


@pytest.fixture(scope="session")
def toy_pipeline(corpus_split):
    train, _ = corpus_split
    return fit_toy_pipeline(train, "logreg")


@pytest.fixture(scope="session")
def helpers():
    """Module-level helpers, reachable from any test file without importing conftest."""
    return {
        "make_raw_frame": make_raw_frame,
        "make_corpus": make_corpus,
        "make_toy_params": make_toy_params,
        "fit_toy_pipeline": fit_toy_pipeline,
        "features_of": features_of,
        "write_npz": write_npz,
        "write_json": write_json,
        "write_csv": write_csv,
        "read_json": read_json,
        "require": require,
        "wordnet_present": wordnet_present,
        "KEPT_IDS": KEPT_IDS,
        "KEPT_LABELS": KEPT_LABELS,
        "WORDNET_HINT": WORDNET_HINT,
    }


@pytest.fixture(scope="session")
def wordnet() -> None:
    """Skip when the WordNet corpus is not installed. Never downloads it."""
    if not wordnet_present():
        pytest.skip(WORDNET_HINT)


# ---------------------------------------------------------------------------
# The API and the shipped receipts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def prediction_log_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("prediction_log") / "predictions.jsonl"
    os.environ["PREDICTION_LOG_PATH"] = str(path)
    return path


@pytest.fixture(scope="session")
def app(prediction_log_path: Path, params: dict):
    require(settings.MODEL_PATH, "run dvc repro (train stage) first")
    require(settings.VERSION_PATH, "run dvc repro (train stage) first")
    require(settings.METRICS_PATH, "run dvc repro (evaluate stage) first")
    if params["preprocess"]["lemmatize"] and not wordnet_present():
        pytest.skip(WORDNET_HINT)
    from tweet_emotion.api.app import create_app

    return create_app()


@pytest.fixture(scope="session")
def client(app) -> Iterator:
    """A client whose lifespan has run, so the model is loaded once per session."""
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def version_info() -> dict:
    require(settings.VERSION_PATH, "run dvc repro (train stage) first")
    return read_json(settings.VERSION_PATH)


@pytest.fixture(scope="session")
def metrics() -> dict:
    require(settings.METRICS_PATH, "run dvc repro (evaluate stage) first")
    return read_json(settings.METRICS_PATH)


@pytest.fixture(scope="session")
def top_terms() -> dict:
    require(settings.TOP_TERMS_PATH, "run dvc repro (evaluate stage) first")
    return read_json(settings.TOP_TERMS_PATH)


@pytest.fixture(scope="session")
def presets_doc() -> dict:
    require(settings.PRESETS_PATH, "run dvc repro (presets stage) first")
    return read_json(settings.PRESETS_PATH)


@pytest.fixture(scope="session")
def presets(presets_doc: dict) -> list[dict]:
    return presets_doc["presets"]


@pytest.fixture
def preset_text(presets: list[dict]) -> str:
    """The first preset's raw tweet, the text the demo page submits first."""
    return str(presets[0]["text"])

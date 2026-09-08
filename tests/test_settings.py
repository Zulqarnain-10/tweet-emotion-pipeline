"""Contract tests for tweet_emotion.settings: paths, column and label contracts, params.yaml,
the frozen stop-word list, and the runtime helpers api_port, log_level, prediction_log_path.
"""

from __future__ import annotations

import logging
import re
import tempfile
import tomllib
from pathlib import Path

import pytest

from tweet_emotion import __version__, settings

PARAM_SECTIONS = ("data", "preprocess", "features", "train", "evaluate", "presets", "api")
PREPROCESS_FLAGS = (
    "lowercase",
    "strip_urls",
    "strip_mentions",
    "strip_numbers",
    "strip_punctuation",
    "remove_stopwords",
    "lemmatize",
)
PATHS_UNDER_ROOT = {
    "PARAMS_PATH": "params.yaml",
    "DATA_DIR": "data",
    "RAW_DIR": "data/raw",
    "PROCESSED_DIR": "data/processed",
    "FEATURES_DIR": "data/features",
    "MODELS_DIR": "models",
    "REPORTS_DIR": "reports",
    "FIGURES_DIR": "reports/figures",
    "CONFIGS_DIR": "configs",
    "RAW_SOURCE_PATH": "data/raw/tweet_emotions.csv",
    "FETCH_MANIFEST_PATH": "data/raw/fetch_manifest.json",
    "RAW_TRAIN_PATH": "data/raw/train.csv",
    "RAW_TEST_PATH": "data/raw/test.csv",
    "PROCESSED_TRAIN_PATH": "data/processed/train.csv",
    "PROCESSED_TEST_PATH": "data/processed/test.csv",
    "FEATURES_TRAIN_PATH": "data/features/train.npz",
    "FEATURES_TEST_PATH": "data/features/test.npz",
    "FEATURE_MANIFEST_PATH": "data/features/feature_manifest.json",
    "VECTORIZER_PATH": "models/vectorizer.joblib",
    "MODEL_PATH": "models/model.joblib",
    "VERSION_PATH": "models/version.json",
    "METRICS_PATH": "reports/metrics.json",
    "TOP_TERMS_PATH": "reports/top_terms.json",
    "LOADTEST_PATH": "reports/loadtest.json",
    "PRESETS_PATH": "configs/presets.json",
    "STOPWORDS_PATH": "configs/stopwords_en.txt",
}
EMOJI = re.compile("[\U0001f300-\U0001faff\u2600-\u27bf]")
EM_DASH = "\u2014"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def test_repo_root_holds_params_yaml():
    assert settings.REPO_ROOT.is_absolute()
    assert (settings.REPO_ROOT / "params.yaml").is_file()
    assert settings.PARAMS_PATH.is_file()


@pytest.mark.parametrize("name", sorted(PATHS_UNDER_ROOT))
def test_path_constants_sit_under_the_repo_root(name):
    value = getattr(settings, name)
    assert isinstance(value, Path)
    assert value == settings.REPO_ROOT / PATHS_UNDER_ROOT[name]


def test_find_repo_root_honours_the_home_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("TWEET_EMOTION_HOME", str(tmp_path))
    assert settings._find_repo_root() == tmp_path.resolve()


def test_find_repo_root_walks_up_to_params_yaml(monkeypatch):
    monkeypatch.delenv("TWEET_EMOTION_HOME", raising=False)
    root = settings._find_repo_root()
    assert (root / "params.yaml").is_file()
    assert (root / "src" / "tweet_emotion" / "settings.py").is_file()


def test_stopwords_file_is_frozen_and_clean():
    lines = settings.STOPWORDS_PATH.read_bytes().decode("utf-8").split("\n")
    words = [line for line in lines if line.strip()]
    assert len(words) == 198
    assert words == [w.strip() for w in words]
    assert all(w == w.lower() for w in words)
    assert len(set(words)) == len(words)
    assert "the" in words
    assert "happy" not in words
    assert "sad" not in words
    assert b"\r\n" not in settings.STOPWORDS_PATH.read_bytes()


# ---------------------------------------------------------------------------
# Column, label, and npz contracts
# ---------------------------------------------------------------------------


def test_csv_columns_contract():
    assert settings.CSV_COLUMNS == ("tweet_id", "text", "label")
    assert settings.ID_COLUMN == "tweet_id"
    assert settings.TEXT_COLUMN == "text"
    assert settings.LABEL_COLUMN == "label"


def test_labels_contract():
    assert settings.LABELS == ("sadness", "happiness")
    assert settings.NEGATIVE_LABEL == "sadness"
    assert settings.POSITIVE_LABEL == "happiness"


def test_labels_agree_with_params(params):
    data = params["data"]
    assert settings.LABELS[1] == data["positive_label"]
    assert set(settings.LABELS) == set(data["keep_labels"])
    assert len(data["keep_labels"]) == 2


def test_npz_keys_contract():
    assert settings.NPZ_KEYS == ("data", "indices", "indptr", "shape", "y")


def test_disclaimer_follows_the_copy_rules():
    text = settings.DISCLAIMER
    assert text.strip() == text
    assert text.endswith(".")
    assert "not a mental-health tool" in text
    assert EM_DASH not in text
    assert not EMOJI.search(text)
    assert text[0].isupper()


# ---------------------------------------------------------------------------
# params.yaml
# ---------------------------------------------------------------------------


def test_load_params_has_every_section(params):
    assert isinstance(params, dict)
    assert set(PARAM_SECTIONS) <= set(params)


def test_load_params_reads_an_explicit_path(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text("a:\n  b: 2\nc: [1, 2]\n", encoding="utf-8")
    assert settings.load_params(path) == {"a": {"b": 2}, "c": [1, 2]}


def test_load_params_default_is_the_repo_params(params):
    assert settings.load_params(settings.PARAMS_PATH) == params


def test_data_params_are_well_formed(params):
    data = params["data"]
    assert data["url"].startswith("https://")
    assert re.fullmatch(r"[0-9a-f]{64}", data["sha256"])
    assert data["text_column"] == "content"
    assert data["label_column"] == "sentiment"
    assert data["id_column"] == "tweet_id"
    assert data["positive_label"] in data["keep_labels"]
    assert isinstance(data["drop_duplicates"], bool)
    assert 0.0 < data["test_size"] < 1.0
    assert isinstance(data["seed"], int)


def test_preprocess_params_are_booleans_plus_the_stopwords_file(params):
    block = params["preprocess"]
    for flag in PREPROCESS_FLAGS:
        assert isinstance(block[flag], bool), flag
    assert block["stopwords_file"] == "configs/stopwords_en.txt"
    assert settings.REPO_ROOT / block["stopwords_file"] == settings.STOPWORDS_PATH


def test_keep_words_are_lowercase_and_unique(params):
    keep = params["preprocess"].get("keep_words", [])
    assert isinstance(keep, list)
    assert all(isinstance(word, str) and word == word.lower() for word in keep)
    assert len(set(keep)) == len(keep)


def test_features_params_are_well_formed(params):
    block = params["features"]
    assert block["method"] in {"tfidf", "bow"}
    assert isinstance(block["max_features"], int) and block["max_features"] > 0
    low, high = block["ngram_range"]
    assert 1 <= low <= high
    assert isinstance(block["min_df"], int) and block["min_df"] >= 1
    assert isinstance(block["sublinear_tf"], bool)


def test_train_params_name_known_candidates(params):
    block = params["train"]
    assert set(block["candidates"]) <= {"logreg", "nb", "xgboost"}
    assert len(block["candidates"]) == len(set(block["candidates"]))
    assert block["selection_metric"] in {"roc_auc", "accuracy", "f1"}
    assert block["cv_folds"] >= 2
    for name in block["candidates"]:
        assert isinstance(block[name], dict), name
    assert set(block["logreg"]) == {"C", "max_iter"}
    assert set(block["nb"]) == {"alpha"}
    assert set(block["xgboost"]) == {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
    }


def test_evaluate_presets_and_api_params(params):
    assert 0.0 < params["evaluate"]["threshold"] < 1.0
    assert params["evaluate"]["top_terms"] >= 1
    assert isinstance(params["evaluate"]["figures"], bool)
    assert params["presets"]["n"] == 4
    assert isinstance(params["api"]["max_chars"], int) and params["api"]["max_chars"] > 0
    assert isinstance(params["api"]["batch_max"], int) and params["api"]["batch_max"] > 0


def test_package_version_matches_pyproject():
    assert __version__ == "0.1.0"
    with open(settings.REPO_ROOT / "pyproject.toml", "rb") as fh:
        pyproject = tomllib.load(fh)
    assert pyproject["project"]["version"] == __version__
    assert pyproject["project"]["name"] == "tweet-emotion-pipeline"


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def test_defaults():
    assert settings.DEFAULT_PORT == 8000
    assert settings.LOG_LEVEL_NAMES == ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


@pytest.mark.parametrize(
    ("raw", "expected", "warns"),
    [
        (None, 8000, False),
        ("", 8000, True),
        ("   ", 8000, True),
        ("abc", 8000, True),
        ("8000.0", 8000, True),
        ("8127", 8127, False),
        (" 7860 ", 7860, False),
    ],
    ids=["unset", "empty", "blank", "letters", "decimal", "custom", "padded"],
)
def test_api_port_falls_back_to_default(monkeypatch, caplog, raw, expected, warns):
    if raw is None:
        monkeypatch.delenv("PORT", raising=False)
    else:
        monkeypatch.setenv("PORT", raw)
    with caplog.at_level(logging.WARNING, logger="tweet_emotion.settings"):
        assert settings.api_port() == expected
    warnings = [r for r in caplog.records if r.name == "tweet_emotion.settings"]
    assert bool(warnings) is warns
    if warns:
        assert "8000" in warnings[0].getMessage()


@pytest.mark.parametrize("raw", ["0", "65536", "70000", "-1"])
def test_api_port_rejects_impossible_values(monkeypatch, raw):
    monkeypatch.setenv("PORT", raw)
    with pytest.raises(ValueError, match="between 1 and 65535"):
        settings.api_port()


@pytest.mark.parametrize(
    ("raw", "expected", "warns"),
    [
        (None, "INFO", False),
        ("", "INFO", False),
        ("debug", "DEBUG", False),
        (" Warning ", "WARNING", False),
        ("ERROR", "ERROR", False),
        ("critical", "CRITICAL", False),
        ("verbose", "INFO", True),
    ],
    ids=["unset", "empty", "lower", "padded", "upper", "critical", "unknown"],
)
def test_log_level_helper(monkeypatch, caplog, raw, expected, warns):
    if raw is None:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("LOG_LEVEL", raw)
    with caplog.at_level(logging.WARNING, logger="tweet_emotion.settings"):
        assert settings.log_level() == expected
    warnings = [r for r in caplog.records if r.name == "tweet_emotion.settings"]
    assert bool(warnings) is warns


def test_prediction_log_path_honours_the_environment(monkeypatch, tmp_path):
    target = tmp_path / "logs" / "p.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(target))
    assert settings.prediction_log_path() == target


def test_prediction_log_path_defaults_to_the_temp_directory(monkeypatch):
    monkeypatch.delenv("PREDICTION_LOG_PATH", raising=False)
    path = settings.prediction_log_path()
    assert path.parent == Path(tempfile.gettempdir())
    assert path.name == "tweet_emotion_predictions.jsonl"


def test_prediction_log_path_ignores_an_empty_variable(monkeypatch):
    monkeypatch.setenv("PREDICTION_LOG_PATH", "")
    assert settings.prediction_log_path().name == "tweet_emotion_predictions.jsonl"

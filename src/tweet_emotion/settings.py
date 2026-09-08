"""Paths, label contracts, and runtime settings shared by every module.

Nothing here reads a secret. The service needs none at runtime.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path:
    """Repo root: TWEET_EMOTION_HOME if set, else the first ancestor with params.yaml, else cwd."""
    env = os.environ.get("TWEET_EMOTION_HOME")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "params.yaml").exists():
            return parent
    return Path.cwd().resolve()


REPO_ROOT: Path = _find_repo_root()
PARAMS_PATH = REPO_ROOT / "params.yaml"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIGS_DIR = REPO_ROOT / "configs"

# Stage outputs, in pipeline order.
RAW_SOURCE_PATH = RAW_DIR / "tweet_emotions.csv"
FETCH_MANIFEST_PATH = RAW_DIR / "fetch_manifest.json"
RAW_TRAIN_PATH = RAW_DIR / "train.csv"
RAW_TEST_PATH = RAW_DIR / "test.csv"
PROCESSED_TRAIN_PATH = PROCESSED_DIR / "train.csv"
PROCESSED_TEST_PATH = PROCESSED_DIR / "test.csv"
FEATURES_TRAIN_PATH = FEATURES_DIR / "train.npz"
FEATURES_TEST_PATH = FEATURES_DIR / "test.npz"
FEATURE_MANIFEST_PATH = FEATURES_DIR / "feature_manifest.json"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
MODEL_PATH = MODELS_DIR / "model.joblib"
VERSION_PATH = MODELS_DIR / "version.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
TOP_TERMS_PATH = REPORTS_DIR / "top_terms.json"
LOADTEST_PATH = REPORTS_DIR / "loadtest.json"
PRESETS_PATH = CONFIGS_DIR / "presets.json"
STOPWORDS_PATH = CONFIGS_DIR / "stopwords_en.txt"

# ---------------------------------------------------------------------------
# Column and label contracts
# ---------------------------------------------------------------------------

# Columns of every CSV the pipeline writes (data/raw/*.csv and data/processed/*.csv).
ID_COLUMN = "tweet_id"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
CSV_COLUMNS: tuple[str, ...] = (ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN)

# label 0 and label 1, in that order. LABELS[1] is params data.positive_label.
LABELS: tuple[str, str] = ("sadness", "happiness")
NEGATIVE_LABEL, POSITIVE_LABEL = LABELS

# Keys of the numpy .npz files written by the features stage. X is a CSR matrix stored as
# data, indices, indptr, shape; y is the label vector.
NPZ_KEYS: tuple[str, ...] = ("data", "indices", "indptr", "shape", "y")

DISCLAIMER = (
    "Demonstration system trained on public tweets labelled happiness or sadness by "
    "crowd workers in 2016. It scores short English text only and is not a mental-health tool."
)

# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


def load_params(path: Path | None = None) -> dict:
    """Load params.yaml as a plain dict."""
    with open(path or PARAMS_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Runtime settings for the API
# ---------------------------------------------------------------------------

DEFAULT_PORT = 8000
LOG_LEVEL_NAMES: tuple[str, ...] = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def api_port() -> int:
    """PORT from the environment, default 8000.

    A blank or non-numeric value falls back to the default with a warning. A number outside
    1..65535 raises ValueError with a one-line message.
    """
    raw = os.environ.get("PORT", "").strip()
    if raw == "":
        if "PORT" in os.environ:
            log.warning("PORT is empty, using %d", DEFAULT_PORT)
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError:
        log.warning("PORT=%r is not an integer, using %d", raw, DEFAULT_PORT)
        return DEFAULT_PORT
    if not 1 <= port <= 65535:
        raise ValueError(f"PORT must be an integer between 1 and 65535, got {port}")
    return port


def log_level() -> str:
    """LOG_LEVEL from the environment as a logging level name, default INFO."""
    raw = os.environ.get("LOG_LEVEL", "").strip().upper()
    if raw == "":
        return "INFO"
    if raw in LOG_LEVEL_NAMES:
        return raw
    log.warning("LOG_LEVEL=%r is not a logging level, using INFO", raw)
    return "INFO"


def prediction_log_path() -> Path:
    """Where the API appends one JSON line per prediction. Ephemeral on free hosting."""
    env = os.environ.get("PREDICTION_LOG_PATH")
    if env:
        return Path(env)
    return Path(tempfile.gettempdir()) / "tweet_emotion_predictions.jsonl"

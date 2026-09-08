"""Fit the vectoriser on train only, transform both splits, save sparse matrices and the vectoriser.

Usage:
    python -m tweet_emotion.features

The vectoriser (tf-idf or bag of words, per params.features) sees the training split only. Its
vocabulary is chosen here rather than by scikit-learn's max_features cut, so the same training
texts give the same terms on every machine (see select_vocabulary). Both splits are written as
numpy .npz files holding the CSR matrix parts plus the label vector, with the keys in
settings.NPZ_KEYS. The fitted vectoriser is saved with joblib and becomes the second step of
the serving Pipeline built by the train stage.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from tweet_emotion import settings

log = logging.getLogger(__name__)

# Single-letter tokens survive (the default pattern needs two characters); case was handled
# by the preprocess stage.
TOKEN_PATTERN = r"(?u)\b\w+\b"
VECTORIZER_METHODS: tuple[str, ...] = ("tfidf", "bow")
# Fixed zip member metadata so an unchanged matrix writes an identical file on every platform:
# the 1980 epoch, Unix as the creating system (Windows would stamp 0), and a plain rw-r--r--
# regular-file mode in the external attributes.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_ZIP_CREATE_SYSTEM = 3
_ZIP_EXTERNAL_ATTR = 0o100644 << 16


# ---------------------------------------------------------------------------
# Vectoriser
# ---------------------------------------------------------------------------


def _features_block(params: dict) -> dict:
    """Accept either the whole params dict or its features block."""
    if isinstance(params.get("features"), dict):
        return params["features"]
    return params


def make_vectorizer(
    params_features: dict, vocabulary: dict[str, int] | None = None
) -> TfidfVectorizer | CountVectorizer:
    """Unfitted vectoriser from the params.yaml features block.

    With vocabulary (the dict select_vocabulary returns) the term set is fixed up front and
    max_features and min_df are implied by it; they stay on the object so the manifest and the
    model card can still read them.
    """
    cfg = _features_block(params_features)
    method = str(cfg.get("method", "tfidf")).lower()
    if method not in VECTORIZER_METHODS:
        raise ValueError(f"features.method must be one of {VECTORIZER_METHODS}, got {method!r}")
    common = {
        "max_features": _max_features(cfg),
        "ngram_range": _ngram_range(cfg),
        "min_df": _min_df(cfg.get("min_df", 1)),
        "token_pattern": TOKEN_PATTERN,
        "lowercase": False,
        "dtype": np.float32,
    }
    if vocabulary is not None:
        common["vocabulary"] = dict(vocabulary)
    if method == "tfidf":
        return TfidfVectorizer(sublinear_tf=bool(cfg.get("sublinear_tf", False)), **common)
    return CountVectorizer(**common)


def select_vocabulary(params_features: dict, texts) -> dict[str, int]:
    """The training vocabulary, chosen so it is the same on every machine.

    scikit-learn's max_features cut ranks terms with numpy's argsort, whose order among ties
    depends on the sort kernel, so two machines can keep different terms. Here a CountVectorizer
    with the same token pattern, n-gram range, and min_df counts the training texts; terms are
    ranked by document frequency, most first, ties broken alphabetically; the top max_features
    survive; and the survivors get alphabetical indices.
    """
    cfg = _features_block(params_features)
    counter = CountVectorizer(
        ngram_range=_ngram_range(cfg),
        min_df=_min_df(cfg.get("min_df", 1)),
        token_pattern=TOKEN_PATTERN,
        lowercase=False,
        binary=True,
    )
    presence = counter.fit_transform(list(texts))
    document_frequency = np.asarray(presence.sum(axis=0)).ravel()
    terms = [str(term) for term in counter.get_feature_names_out()]
    ranked = sorted(range(len(terms)), key=lambda i: (-int(document_frequency[i]), terms[i]))
    chosen = sorted(terms[i] for i in ranked[: _max_features(cfg)])
    return {term: index for index, term in enumerate(chosen)}


def fit_vectorizer(params_features: dict, train_text) -> TfidfVectorizer | CountVectorizer:
    """Vectoriser fitted on the training texts over the vocabulary select_vocabulary chose."""
    cfg = _features_block(params_features)
    vectorizer = make_vectorizer(cfg, vocabulary=select_vocabulary(cfg, train_text))
    vectorizer.fit(list(train_text))
    return vectorizer


def _max_features(cfg: dict) -> int | None:
    return int(cfg["max_features"]) if cfg.get("max_features") else None


def _ngram_range(cfg: dict) -> tuple[int, int]:
    ngram = cfg.get("ngram_range", [1, 1])
    if len(ngram) != 2:
        raise ValueError(f"features.ngram_range must have two entries, got {ngram!r}")
    return (int(ngram[0]), int(ngram[1]))


def _min_df(value) -> int | float:
    """min_df is a document count when integral, a fraction otherwise."""
    number = float(value)
    return int(number) if number.is_integer() else number


def feature_names(vectorizer) -> list[str]:
    return [str(name) for name in vectorizer.get_feature_names_out()]


# ---------------------------------------------------------------------------
# Sparse matrix files
# ---------------------------------------------------------------------------


def save_npz(path: Path, X: csr_matrix, y: np.ndarray) -> None:
    """Write a CSR matrix and its labels as a numpy .npz with keys settings.NPZ_KEYS."""
    X = csr_matrix(X)
    X.sort_indices()
    arrays = {
        "data": X.data,
        "indices": X.indices,
        "indptr": X.indptr,
        "shape": np.asarray(X.shape, dtype=np.int64),
        "y": np.asarray(y),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key in settings.NPZ_KEYS:
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.ascontiguousarray(arrays[key]))
            info = zipfile.ZipInfo(f"{key}.npy", date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = _ZIP_CREATE_SYSTEM
            info.external_attr = _ZIP_EXTERNAL_ATTR
            archive.writestr(info, buffer.getvalue())
    os.replace(tmp, path)


def load_npz(path: Path) -> tuple[csr_matrix, np.ndarray]:
    """Read a file written by save_npz. Returns (X as CSR, y)."""
    with np.load(path) as archive:
        missing = [key for key in settings.NPZ_KEYS if key not in archive.files]
        if missing:
            raise ValueError(f"{path} is missing keys {missing}")
        shape = tuple(int(n) for n in archive["shape"])
        X = csr_matrix((archive["data"], archive["indices"], archive["indptr"]), shape=shape)
        y = np.asarray(archive["y"])
    return X, y


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def read_processed(path: Path) -> pd.DataFrame:
    """Read a processed split; empty strings stay strings, never NaN."""
    df = pd.read_csv(
        path, dtype={settings.TEXT_COLUMN: str}, keep_default_na=False, encoding="utf-8"
    )
    missing = [c for c in settings.CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    return df


def build_manifest(vectorizer, cfg: dict, X_train: csr_matrix, n_test: int) -> dict:
    method = str(cfg.get("method", "tfidf")).lower()
    n_train, vocabulary_size = X_train.shape
    cells = n_train * vocabulary_size
    return {
        "method": method,
        "max_features": int(cfg["max_features"]) if cfg.get("max_features") else None,
        "ngram_range": [int(n) for n in vectorizer.ngram_range],
        "min_df": _min_df(cfg.get("min_df", 1)),
        "sublinear_tf": bool(cfg.get("sublinear_tf", False)) if method == "tfidf" else False,
        "vocabulary_size": int(vocabulary_size),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "nnz_train": int(X_train.nnz),
        "density_train": round(float(X_train.nnz) / cells, 4) if cells else 0.0,
    }


def transform_splits(vectorizer, train_text, test_text) -> tuple[csr_matrix, csr_matrix]:
    """Both splits through the fitted vectoriser, as CSR matrices."""
    X_train = csr_matrix(vectorizer.transform(list(train_text)))
    X_test = csr_matrix(vectorizer.transform(list(test_text)))
    return X_train, X_test


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        params = settings.load_params()
        cfg = params["features"]
        train = read_processed(settings.PROCESSED_TRAIN_PATH)
        test = read_processed(settings.PROCESSED_TEST_PATH)
        vectorizer = fit_vectorizer(cfg, train[settings.TEXT_COLUMN])
        X_train, X_test = transform_splits(
            vectorizer, train[settings.TEXT_COLUMN], test[settings.TEXT_COLUMN]
        )
        y_train = train[settings.LABEL_COLUMN].to_numpy().astype(np.int64)
        y_test = test[settings.LABEL_COLUMN].to_numpy().astype(np.int64)
        log.info(
            "%s fitted on %d rows: vocabulary %d, train nnz %d",
            cfg.get("method", "tfidf"),
            X_train.shape[0],
            X_train.shape[1],
            X_train.nnz,
        )

        save_npz(settings.FEATURES_TRAIN_PATH, X_train, y_train)
        save_npz(settings.FEATURES_TEST_PATH, X_test, y_test)
        log.info("wrote %s and %s", settings.FEATURES_TRAIN_PATH, settings.FEATURES_TEST_PATH)

        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = settings.VECTORIZER_PATH.with_name(settings.VECTORIZER_PATH.name + ".tmp")
        joblib.dump(vectorizer, tmp, compress=3)
        os.replace(tmp, settings.VECTORIZER_PATH)
        log.info("wrote %s", settings.VECTORIZER_PATH)

        manifest = build_manifest(vectorizer, cfg, X_train, X_test.shape[0])
        tmp = settings.FEATURE_MANIFEST_PATH.with_name(settings.FEATURE_MANIFEST_PATH.name + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, settings.FEATURE_MANIFEST_PATH)
        log.info("wrote %s", settings.FEATURE_MANIFEST_PATH)
    except Exception as exc:
        log.error("features failed: %s: %s", type(exc).__name__, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

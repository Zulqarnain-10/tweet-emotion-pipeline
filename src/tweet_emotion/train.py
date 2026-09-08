"""Cross-validate the candidates on train, fit the winner, save the text-to-probability pipeline.

Usage:
    python -m tweet_emotion.train

Each candidate in params.train.candidates is scored by stratified k-fold cross-validation on the
training matrix only. The highest mean of the selection metric wins (ties go to the earlier
candidate), is refitted on all of train, and is saved as models/model.joblib inside a Pipeline
of TextNormalizer -> fitted vectoriser -> classifier, so the API scores raw text directly.
models/version.json records provenance. The test split is not touched here.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from tweet_emotion import __version__, settings
from tweet_emotion.features import load_npz
from tweet_emotion.preprocess import TextNormalizer

log = logging.getLogger(__name__)

CV_SCORING: tuple[str, ...] = ("roc_auc", "accuracy", "f1")
SMOKE_TEXT = "i love this"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


# Files that dvc repro rewrites; a diff there does not make the source tree dirty.
PIPELINE_OUTPUTS = ("dvc.lock", "models/", "reports/", "configs/presets.json", "data/")


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=settings.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_info() -> dict:
    """Current commit sha, its short form, and whether tracked source files are modified."""
    sha = _git("rev-parse", "HEAD")
    if not sha:
        return {"git_sha": "unknown", "git_sha_short": "unknown", "git_dirty": False}
    status = _git("status", "--porcelain", "--untracked-files=no") or ""
    # Pipeline outputs change while dvc repro runs; only modified source counts as dirty.
    changed = [path for path in _status_paths(status) if not path.startswith(PIPELINE_OUTPUTS)]
    return {"git_sha": sha, "git_sha_short": sha[:7], "git_dirty": bool(changed)}


def _status_paths(porcelain: str) -> list[str]:
    """Paths from git status --porcelain, tolerant of stripped leading columns and renames."""
    paths = []
    for raw in porcelain.splitlines():
        line = raw.strip()
        if not line:
            continue
        path = line.split(maxsplit=1)[1] if " " in line else line
        if " -> " in path:
            path = path.split(" -> ")[-1]
        paths.append(path.strip().strip('"').replace("\\", "/"))
    return paths


def _xgboost_version() -> str:
    """xgboost ships as the xgboost-cpu distribution here, so ask the module, not the metadata."""
    try:
        import xgboost
    except ImportError:
        return "not installed"
    return str(xgboost.__version__)


def library_versions() -> dict:
    def _version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "not installed"

    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "nltk": _version("nltk"),
        "xgboost": _xgboost_version(),
        "joblib": joblib.__version__,
    }


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def _train_block(params: dict) -> dict:
    """Accept either the whole params dict or its train block."""
    if isinstance(params.get("train"), dict):
        return params["train"]
    return params


def make_logreg(params: dict) -> LogisticRegression:
    train = _train_block(params)
    cfg = train["logreg"]
    return LogisticRegression(
        C=float(cfg["C"]),
        max_iter=int(cfg["max_iter"]),
        solver="liblinear",
        random_state=int(train["seed"]),
    )


def make_nb(params: dict) -> MultinomialNB:
    train = _train_block(params)
    return MultinomialNB(alpha=float(train["nb"]["alpha"]))


def make_xgboost(params: dict):
    """XGBClassifier on the sparse matrix; the lazy import lets the module load without xgboost."""
    from xgboost import XGBClassifier

    train = _train_block(params)
    cfg = train["xgboost"]
    return XGBClassifier(
        n_estimators=int(cfg["n_estimators"]),
        max_depth=int(cfg["max_depth"]),
        learning_rate=float(cfg["learning_rate"]),
        subsample=float(cfg["subsample"]),
        colsample_bytree=float(cfg["colsample_bytree"]),
        random_state=int(train["seed"]),
        n_jobs=1,
        tree_method="hist",
        eval_metric="logloss",
    )


CANDIDATE_FACTORIES = {"logreg": make_logreg, "nb": make_nb, "xgboost": make_xgboost}


def make_candidate(name: str, params: dict):
    """Unfitted classifier for a candidate name from params.train.candidates."""
    try:
        factory = CANDIDATE_FACTORIES[name]
    except KeyError:
        raise ValueError(
            f"unknown candidate {name!r}; known: {sorted(CANDIDATE_FACTORIES)}"
        ) from None
    return factory(params)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def cross_validate_candidate(
    name: str, clf, X: csr_matrix, y: np.ndarray, folds: int, seed: int
) -> dict:
    """Stratified k-fold scores for one candidate; fit_seconds is the total across folds."""
    y = np.asarray(y).astype(int)
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    result = cross_validate(clf, X, y, cv=splitter, scoring=list(CV_SCORING), n_jobs=1)
    out = {
        "cv_roc_auc_mean": round(float(np.mean(result["test_roc_auc"])), 4),
        "cv_roc_auc_std": round(float(np.std(result["test_roc_auc"])), 4),
        "cv_accuracy_mean": round(float(np.mean(result["test_accuracy"])), 4),
        "cv_f1_mean": round(float(np.mean(result["test_f1"])), 4),
        "fit_seconds": round(float(np.sum(result["fit_time"])), 2),
    }
    log.info("%s cv: %s", name, out)
    return out


def select_winner(cv_results: dict[str, dict], selection_metric: str) -> str:
    """Highest cv_<metric>_mean wins; ties go to the candidate listed first."""
    key = f"cv_{selection_metric}_mean"
    if not cv_results:
        raise ValueError("no candidates were cross-validated")
    winner, best = None, None
    for name, scores in cv_results.items():
        if key not in scores:
            raise ValueError(f"{name} has no {key}; selection_metric must be one of {CV_SCORING}")
        if best is None or scores[key] > best:
            winner, best = name, scores[key]
    return str(winner)


def build_pipeline(params: dict, vectorizer, clf) -> Pipeline:
    """Serving object: raw text -> normalise -> vectorise -> classify. Steps are already fitted."""
    pipeline = Pipeline(
        [
            ("normalize", TextNormalizer.from_params(params["preprocess"])),
            ("vectorize", vectorizer),
            ("classify", clf),
        ]
    )
    probabilities = pipeline.predict_proba([SMOKE_TEXT])
    if tuple(probabilities.shape) != (1, 2):
        raise RuntimeError(f"pipeline predict_proba shape {probabilities.shape}, expected (1, 2)")
    log.info("pipeline check: p(happiness | %r) = %.4f", SMOKE_TEXT, float(probabilities[0, 1]))
    return pipeline


def build_version(
    winner: str, cv_results: dict[str, dict], params: dict, n_train: int, n_features: int
) -> dict:
    train = params["train"]
    git = git_info()
    return {
        "model": winner,
        "package_version": __version__,
        "model_version": f"{__version__}+{git['git_sha_short']}",
        **git,
        "data_sha256": params["data"]["sha256"],
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "n_train": int(n_train),
        "n_features": int(n_features),
        "selection_metric": str(train["selection_metric"]),
        "cv_folds": int(train["cv_folds"]),
        "candidates": cv_results,
        "winner_params": dict(train[winner]),
        "libraries": library_versions(),
    }


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    started = time.perf_counter()
    try:
        params = settings.load_params()
        train = params["train"]
        X, y = load_npz(settings.FEATURES_TRAIN_PATH)
        vectorizer = joblib.load(settings.VECTORIZER_PATH)
        n_features = len(vectorizer.vocabulary_)
        if X.shape[1] != n_features:
            raise RuntimeError(
                f"train matrix has {X.shape[1]} columns but the vectoriser has {n_features} terms"
            )
        log.info("train matrix %s, positive rate %.4f", X.shape, float(np.mean(y)))

        cv_results: dict[str, dict] = {}
        for name in train["candidates"]:
            cv_results[name] = cross_validate_candidate(
                name, make_candidate(name, params), X, y, train["cv_folds"], train["seed"]
            )
        winner = select_winner(cv_results, train["selection_metric"])
        log.info("winner by cv %s: %s", train["selection_metric"], winner)

        clf = make_candidate(winner, params)
        fit_started = time.perf_counter()
        clf.fit(X, np.asarray(y).astype(int))
        log.info("%s fitted on all of train in %.2fs", winner, time.perf_counter() - fit_started)

        pipeline = build_pipeline(params, vectorizer, clf)
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = settings.MODEL_PATH.with_name(settings.MODEL_PATH.name + ".tmp")
        joblib.dump(pipeline, tmp, compress=3)
        os.replace(tmp, settings.MODEL_PATH)
        log.info("wrote %s", settings.MODEL_PATH)

        version = build_version(winner, cv_results, params, X.shape[0], n_features)
        tmp = settings.VERSION_PATH.with_name(settings.VERSION_PATH.name + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(version, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, settings.VERSION_PATH)
        log.info("wrote %s", settings.VERSION_PATH)
    except Exception as exc:
        log.error("train failed: %s: %s", type(exc).__name__, exc)
        return 1
    print(
        f"winner={winner} cv={cv_results[winner]} model_version={version['model_version']} "
        f"total={time.perf_counter() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

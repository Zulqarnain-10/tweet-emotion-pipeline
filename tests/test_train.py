"""Unit tests for tweet_emotion.train: candidate factories, cross-validation on a tiny sparse
matrix, the winner rule, git provenance helpers, and the stage that writes the serving
pipeline plus models/version.json.

The contract names cross_validate_candidate, _git, git_info, _status_paths, and
library_versions. The candidate factory and the winner rule are looked up under the names
the module is most likely to use; a test skips with a reason when none is found.
"""

from __future__ import annotations

import platform
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
import sklearn
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from tweet_emotion import __version__, settings, train
from tweet_emotion.preprocess import TextNormalizer, load_stopwords

CV_KEYS = {"cv_roc_auc_mean", "cv_roc_auc_std", "cv_accuracy_mean", "cv_f1_mean", "fit_seconds"}
PREPROCESS_FLAGS = (
    "lowercase",
    "strip_urls",
    "strip_mentions",
    "strip_numbers",
    "strip_punctuation",
    "remove_stopwords",
    "lemmatize",
)
VERSION_KEYS = {
    "model",
    "package_version",
    "model_version",
    "git_sha",
    "git_sha_short",
    "git_dirty",
    "data_sha256",
    "trained_at",
    "n_train",
    "n_features",
    "selection_metric",
    "cv_folds",
    "candidates",
    "winner_params",
    "libraries",
}
LIBRARY_KEYS = {"python", "scikit-learn", "numpy", "pandas", "scipy", "nltk", "xgboost", "joblib"}
ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
FACTORY_NAMES = (
    "make_candidate",
    "make_classifier",
    "build_candidate",
    "build_classifier",
    "make_model",
)
WINNER_NAMES = ("pick_winner", "select_winner", "choose_winner", "best_candidate")


def build(name: str, params: dict):
    """The classifier for a candidate name, whichever factory shape the module exposes."""
    for attr in FACTORY_NAMES:
        fn = getattr(train, attr, None)
        if fn is None:
            continue
        try:
            return fn(name, params)
        except TypeError:
            return fn(name, params["train"])
    table = getattr(train, "CANDIDATE_FACTORIES", None) or getattr(train, "MODEL_FACTORIES", None)
    if table is not None:
        return table[name](params)
    fn = getattr(train, f"make_{name}", None)
    if fn is not None:
        return fn(params)
    pytest.skip("tweet_emotion.train exposes no candidate factory under a known name")


def winner(results: dict, metric: str) -> str:
    for attr in WINNER_NAMES:
        fn = getattr(train, attr, None)
        if fn is None:
            continue
        try:
            return fn(results, metric)
        except (TypeError, KeyError):
            return fn(results, {"train": {"selection_metric": metric}})
    pytest.skip("tweet_emotion.train exposes no winner rule under a known name")


def sparse_problem(n: int = 60, seed: int = 0) -> tuple[csr_matrix, np.ndarray]:
    """Two classes that light up disjoint halves of ten features, with a little noise."""
    rng = np.random.default_rng(seed)
    y = np.array([i % 2 for i in range(n)])
    X = np.zeros((n, 10), dtype=np.float32)
    for i, label in enumerate(y):
        start = 0 if label else 5
        cols = rng.choice(np.arange(start, start + 5), size=3, replace=False)
        X[i, cols] = rng.uniform(0.5, 1.5, size=3)
        if rng.random() < 0.15:
            X[i, rng.integers(10)] = 1.0
    return csr_matrix(X), y


def cv(name: str, clf, X, y, folds: int = 3, seed: int = 0) -> dict:
    return train.cross_validate_candidate(name, clf, X, y, folds, seed)


# ---------------------------------------------------------------------------
# Candidate factories
# ---------------------------------------------------------------------------


def test_logreg_factory(params):
    clf = build("logreg", params)
    cfg = params["train"]["logreg"]
    assert isinstance(clf, LogisticRegression)
    assert cfg["C"] == clf.C
    assert cfg["max_iter"] == clf.max_iter
    assert clf.solver == "liblinear"
    assert clf.random_state == params["train"]["seed"]


def test_nb_factory(params):
    clf = build("nb", params)
    assert isinstance(clf, MultinomialNB)
    assert clf.alpha == params["train"]["nb"]["alpha"]


def test_xgboost_factory(params):
    xgboost = pytest.importorskip("xgboost")
    clf = build("xgboost", params)
    cfg = params["train"]["xgboost"]
    assert isinstance(clf, xgboost.XGBClassifier)
    got = clf.get_params()
    for key in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"):
        assert got[key] == cfg[key], key
    assert got["random_state"] == params["train"]["seed"]
    assert got["n_jobs"] == 1
    assert got["tree_method"] == "hist"
    assert got["eval_metric"] == "logloss"


def test_unknown_candidate_is_refused(params):
    with pytest.raises((ValueError, KeyError)):
        build("svm", params)


def test_factory_returns_a_fresh_object_each_time(params):
    assert build("logreg", params) is not build("logreg", params)


def test_factory_reads_the_params_block(params):
    tweaked = {**params, "train": {**params["train"], "logreg": {"C": 0.25, "max_iter": 77}}}
    clf = build("logreg", tweaked)
    assert clf.C == 0.25
    assert clf.max_iter == 77


# ---------------------------------------------------------------------------
# cross_validate_candidate
# ---------------------------------------------------------------------------


def test_cross_validate_returns_the_contract_keys():
    X, y = sparse_problem()
    out = cv("logreg", LogisticRegression(solver="liblinear", random_state=0), X, y)
    assert set(out) == CV_KEYS
    for key in CV_KEYS - {"fit_seconds"}:
        assert 0.0 <= out[key] <= 1.0, key
        assert out[key] == round(out[key], 4), key
    assert out["fit_seconds"] >= 0.0
    assert out["fit_seconds"] == round(out["fit_seconds"], 2)


def test_cross_validate_finds_the_signal():
    X, y = sparse_problem()
    out = cv("logreg", LogisticRegression(solver="liblinear", random_state=0), X, y)
    assert out["cv_roc_auc_mean"] > 0.8
    assert out["cv_accuracy_mean"] > 0.7
    assert out["cv_f1_mean"] > 0.7


def test_cross_validate_is_deterministic_for_a_seed():
    X, y = sparse_problem()
    a = cv("logreg", LogisticRegression(solver="liblinear", random_state=0), X, y, 3, 1)
    b = cv("logreg", LogisticRegression(solver="liblinear", random_state=0), X, y, 3, 1)
    for key in CV_KEYS - {"fit_seconds"}:
        assert a[key] == b[key], key


def test_cross_validate_folds_change_the_estimate_not_the_keys():
    X, y = sparse_problem()
    a = cv("logreg", LogisticRegression(solver="liblinear", random_state=0), X, y, 2, 0)
    b = cv("logreg", LogisticRegression(solver="liblinear", random_state=0), X, y, 5, 0)
    assert set(a) == set(b) == CV_KEYS


def test_cross_validate_works_for_naive_bayes():
    X, y = sparse_problem()
    out = cv("nb", MultinomialNB(alpha=0.5), X, y)
    assert set(out) == CV_KEYS
    assert out["cv_roc_auc_mean"] > 0.8


def test_cross_validate_does_not_fit_the_passed_estimator():
    X, y = sparse_problem()
    clf = LogisticRegression(solver="liblinear", random_state=0)
    cv("logreg", clf, X, y)
    assert not hasattr(clf, "coef_")


# ---------------------------------------------------------------------------
# Winner rule
# ---------------------------------------------------------------------------


def _cv(roc: float, acc: float = 0.5, f1: float = 0.5) -> dict:
    return {
        "cv_roc_auc_mean": roc,
        "cv_roc_auc_std": 0.01,
        "cv_accuracy_mean": acc,
        "cv_f1_mean": f1,
        "fit_seconds": 0.1,
    }


def test_winner_is_the_highest_mean_of_the_selection_metric():
    results = {"logreg": _cv(0.90), "nb": _cv(0.95), "xgboost": _cv(0.93)}
    assert winner(results, "roc_auc") == "nb"


def test_winner_ties_go_to_candidate_order():
    results = {"logreg": _cv(0.90), "nb": _cv(0.90), "xgboost": _cv(0.85)}
    assert winner(results, "roc_auc") == "logreg"
    reordered = {"nb": _cv(0.90), "logreg": _cv(0.90)}
    assert winner(reordered, "roc_auc") == "nb"


def test_winner_follows_the_selection_metric():
    results = {"logreg": _cv(0.95, acc=0.70, f1=0.60), "nb": _cv(0.90, acc=0.80, f1=0.75)}
    assert winner(results, "roc_auc") == "logreg"
    assert winner(results, "accuracy") == "nb"
    assert winner(results, "f1") == "nb"


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def test_status_paths_parses_porcelain_output():
    porcelain = (
        " M src/tweet_emotion/train.py\n"
        "?? notes.txt\n"
        "R  old.py -> new.py\n"
        '?? "with space.txt"\n'
        " M models\\version.json\n"
        "\n"
    )
    assert train._status_paths(porcelain) == [
        "src/tweet_emotion/train.py",
        "notes.txt",
        "new.py",
        "with space.txt",
        "models/version.json",
    ]


def test_status_paths_of_nothing():
    assert train._status_paths("") == []
    assert train._status_paths("\n\n") == []


def test_git_info_ignores_pipeline_outputs(monkeypatch):
    def fake_git(*args):
        if args[:2] == ("rev-parse", "HEAD"):
            return "a" * 40
        return (
            " M models/version.json\n M models/model.joblib\n M reports/metrics.json\n"
            " M reports/figures/roc.png\n M configs/presets.json\n M dvc.lock\n"
        )

    monkeypatch.setattr(train, "_git", fake_git)
    info = train.git_info()
    assert info == {"git_sha": "a" * 40, "git_sha_short": "a" * 7, "git_dirty": False}


def test_git_info_marks_modified_source_as_dirty(monkeypatch):
    def fake_git(*args):
        if args[:2] == ("rev-parse", "HEAD"):
            return "b" * 40
        return " M models/version.json\n M src/tweet_emotion/train.py\n"

    monkeypatch.setattr(train, "_git", fake_git)
    info = train.git_info()
    assert info["git_sha_short"] == "b" * 7
    assert info["git_dirty"] is True


def test_git_info_without_git(monkeypatch):
    monkeypatch.setattr(train, "_git", lambda *args: None)
    assert train.git_info() == {
        "git_sha": "unknown",
        "git_sha_short": "unknown",
        "git_dirty": False,
    }


def test_git_helper_returns_text_or_none():
    out = train._git("--version")
    assert out is None or out.startswith("git version")


def test_git_helper_swallows_failures(monkeypatch):
    import subprocess

    def boom(*args, **kwargs):
        raise OSError("git is not installed")

    monkeypatch.setattr(subprocess, "run", boom)
    assert train._git("rev-parse", "HEAD") is None


def test_library_versions_keys():
    libs = train.library_versions()
    assert set(libs) == LIBRARY_KEYS
    assert all(isinstance(v, str) and v for v in libs.values())
    assert libs["python"] == platform.python_version()
    assert libs["scikit-learn"] == sklearn.__version__
    assert libs["numpy"] == np.__version__


def test_library_versions_reports_a_missing_xgboost(monkeypatch):
    from importlib import metadata

    real = metadata.version

    def fake(name: str) -> str:
        if name.startswith("xgboost"):
            raise metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(metadata, "version", fake)
    monkeypatch.setitem(sys.modules, "xgboost", None)
    assert train.library_versions()["xgboost"] == "not installed"


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


@pytest.fixture
def staged(tmp_path: Path, monkeypatch, toy_params, corpus_split, helpers) -> dict:
    """train.npz and vectorizer.joblib built by hand, settings pointed at tmp."""
    train_df, _ = corpus_split
    normalizer = TextNormalizer.from_params(toy_params["preprocess"])
    normalized = normalizer.transform(train_df["text"])
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        dtype=np.float32,
    )
    X = csr_matrix(vectorizer.fit_transform(normalized))
    y = train_df["label"].to_numpy().astype(int)
    features_dir = tmp_path / "features"
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    helpers["write_npz"](features_dir / "train.npz", X, y)
    joblib.dump(vectorizer, models_dir / "vectorizer.joblib", compress=3)
    monkeypatch.setattr(settings, "load_params", lambda path=None: toy_params)
    monkeypatch.setattr(settings, "FEATURES_DIR", features_dir)
    monkeypatch.setattr(settings, "FEATURES_TRAIN_PATH", features_dir / "train.npz")
    monkeypatch.setattr(settings, "MODELS_DIR", models_dir)
    monkeypatch.setattr(settings, "VECTORIZER_PATH", models_dir / "vectorizer.joblib")
    monkeypatch.setattr(settings, "MODEL_PATH", models_dir / "model.joblib")
    monkeypatch.setattr(settings, "VERSION_PATH", models_dir / "version.json")
    return {
        "params": toy_params,
        "models_dir": models_dir,
        "train": train_df,
        "X": X,
        "y": y,
        "vectorizer": vectorizer,
    }


def test_main_writes_a_serving_pipeline(staged):
    assert train.main([]) == 0
    model = joblib.load(staged["models_dir"] / "model.joblib")
    assert isinstance(model, Pipeline)
    assert [name for name, _ in model.steps] == ["normalize", "vectorize", "classify"]
    assert isinstance(model.named_steps["normalize"], TextNormalizer)
    assert isinstance(model.named_steps["classify"], LogisticRegression | MultinomialNB)
    proba = model.predict_proba(["i love this"])
    assert proba.shape == (1, 2)
    assert np.isfinite(proba).all()


def test_main_keeps_the_fitted_vectorizer(staged):
    assert train.main([]) == 0
    model = joblib.load(staged["models_dir"] / "model.joblib")
    assert model.named_steps["vectorize"].vocabulary_ == staged["vectorizer"].vocabulary_


def test_main_normalizer_mirrors_the_params(staged):
    assert train.main([]) == 0
    model = joblib.load(staged["models_dir"] / "model.joblib")
    normalizer = model.named_steps["normalize"]
    block = staged["params"]["preprocess"]
    for flag in PREPROCESS_FLAGS:
        assert getattr(normalizer, flag) == block[flag], flag
    keep = {str(word).lower() for word in block.get("keep_words", [])}
    assert normalizer.stopwords == load_stopwords() - keep
    assert normalizer.keep_words == tuple(sorted(keep))


def test_main_pipeline_agrees_with_the_feature_matrix(staged):
    assert train.main([]) == 0
    model = joblib.load(staged["models_dir"] / "model.joblib")
    through_pipeline = model.predict_proba(staged["train"]["text"].tolist())[:, 1]
    through_matrix = model.named_steps["classify"].predict_proba(staged["X"])[:, 1]
    assert np.max(np.abs(through_pipeline - through_matrix)) < 1e-6


def test_version_json_has_the_contract_keys(staged, helpers):
    assert train.main([]) == 0
    version = helpers["read_json"](staged["models_dir"] / "version.json")
    assert set(version) == VERSION_KEYS
    cfg = staged["params"]["train"]
    assert version["model"] in cfg["candidates"]
    assert version["package_version"] == __version__
    assert version["model_version"] == f"{__version__}+{version['git_sha_short']}"
    assert isinstance(version["git_sha"], str)
    assert isinstance(version["git_dirty"], bool)
    assert version["data_sha256"] == staged["params"]["data"]["sha256"]
    assert ISO_Z.match(version["trained_at"])
    assert version["n_train"] == len(staged["train"])
    assert version["n_features"] == len(staged["vectorizer"].vocabulary_)
    assert version["selection_metric"] == cfg["selection_metric"]
    assert version["cv_folds"] == cfg["cv_folds"]
    assert set(version["candidates"]) == set(cfg["candidates"])
    for name, block in version["candidates"].items():
        assert set(block) == CV_KEYS, name
    assert version["winner_params"] == cfg[version["model"]]
    assert set(version["libraries"]) == LIBRARY_KEYS


def test_version_json_winner_has_the_best_cv_score(staged, helpers):
    assert train.main([]) == 0
    version = helpers["read_json"](staged["models_dir"] / "version.json")
    key = f"cv_{version['selection_metric']}_mean"
    best = max(block[key] for block in version["candidates"].values())
    assert version["candidates"][version["model"]][key] == best


def test_version_json_uses_lf(staged):
    assert train.main([]) == 0
    raw = (staged["models_dir"] / "version.json").read_bytes()
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")


def test_main_with_a_single_naive_bayes_candidate(staged, monkeypatch, helpers):
    only_nb = {**staged["params"], "train": {**staged["params"]["train"], "candidates": ["nb"]}}
    monkeypatch.setattr(settings, "load_params", lambda path=None: only_nb)
    assert train.main([]) == 0
    model = joblib.load(staged["models_dir"] / "model.joblib")
    assert isinstance(model.named_steps["classify"], MultinomialNB)
    version = helpers["read_json"](staged["models_dir"] / "version.json")
    assert version["model"] == "nb"
    assert list(version["candidates"]) == ["nb"]
    assert version["winner_params"] == {"alpha": staged["params"]["train"]["nb"]["alpha"]}


def test_main_fails_without_features(tmp_path: Path, monkeypatch, toy_params):
    monkeypatch.setattr(settings, "load_params", lambda path=None: toy_params)
    monkeypatch.setattr(settings, "FEATURES_TRAIN_PATH", tmp_path / "nope" / "train.npz")
    monkeypatch.setattr(settings, "VECTORIZER_PATH", tmp_path / "nope" / "vectorizer.joblib")
    monkeypatch.setattr(settings, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(settings, "VERSION_PATH", tmp_path / "version.json")
    try:
        code = train.main([])
    except (OSError, ValueError):
        return
    assert code != 0

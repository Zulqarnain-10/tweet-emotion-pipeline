"""Unit tests for tweet_emotion.evaluate: the metrics file, the top-terms file, the figures,
and the train/serve parity check, all on a tiny pipeline fitted inside the tests.

Every number in reports/metrics.json is recomputed here with scikit-learn on the same
probabilities, so the stage cannot drift from the library it cites.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tweet_emotion import evaluate, settings

METRICS_KEYS = {
    "model",
    "model_version",
    "trained_at",
    "git_sha",
    "data_sha256",
    "n_train",
    "n_test",
    "n_features",
    "positive_label",
    "positive_rate_test",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier",
    "log_loss",
    "confusion",
    "majority_baseline_accuracy",
    "candidates_cv",
    "selection_metric",
}
RATE_KEYS = (
    "positive_rate_test",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier",
    "majority_baseline_accuracy",
)
CV_BLOCK = {
    "cv_roc_auc_mean": 0.9,
    "cv_roc_auc_std": 0.01,
    "cv_accuracy_mean": 0.8,
    "cv_f1_mean": 0.8,
    "fit_seconds": 0.1,
}
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
METRIC_HELPER_NAMES = ("compute_metrics", "classification_metrics", "score_metrics", "metrics_at")


def version_doc(params: dict, model: str, n_train: int, n_features: int) -> dict:
    return {
        "model": model,
        "package_version": "0.1.0",
        "model_version": "0.1.0+abc1234",
        "git_sha": "abc1234" * 5 + "abcde",
        "git_sha_short": "abc1234",
        "git_dirty": False,
        "data_sha256": params["data"]["sha256"],
        "trained_at": "2026-01-01T00:00:00Z",
        "n_train": n_train,
        "n_features": n_features,
        "selection_metric": "roc_auc",
        "cv_folds": 2,
        "candidates": {model: dict(CV_BLOCK)},
        "winner_params": dict(params["train"][model]),
        "libraries": {"python": "3.12.0", "scikit-learn": "1.0"},
    }


@pytest.fixture
def stage(tmp_path: Path, monkeypatch, toy_params, corpus_split, helpers):
    """Build every input of the evaluate stage for one classifier kind, point settings at it."""

    def _stage(classifier: str = "logreg", figures: bool = False, top_n: int = 5) -> dict:
        train_df, test_df = corpus_split
        pipeline = helpers["fit_toy_pipeline"](train_df, classifier)
        X_test = helpers["features_of"](pipeline, test_df["text"])
        y_test = test_df["label"].to_numpy().astype(int)
        models = tmp_path / "models"
        reports = tmp_path / "reports"
        features = tmp_path / "features"
        raw = tmp_path / "raw"
        models.mkdir(exist_ok=True)
        joblib.dump(pipeline, models / "model.joblib", compress=3)
        n_features = len(pipeline.named_steps["vectorize"].vocabulary_)
        version = version_doc(toy_params, classifier, len(train_df), n_features)
        helpers["write_json"](models / "version.json", version)
        helpers["write_npz"](features / "test.npz", X_test, y_test)
        helpers["write_csv"](raw / "test.csv", test_df)
        params = {
            **toy_params,
            "evaluate": {"threshold": 0.5, "top_terms": top_n, "figures": figures},
        }
        monkeypatch.setattr(settings, "load_params", lambda path=None: params)
        monkeypatch.setattr(settings, "MODEL_PATH", models / "model.joblib")
        monkeypatch.setattr(settings, "VERSION_PATH", models / "version.json")
        monkeypatch.setattr(settings, "FEATURES_DIR", features)
        monkeypatch.setattr(settings, "FEATURES_TEST_PATH", features / "test.npz")
        monkeypatch.setattr(settings, "RAW_TEST_PATH", raw / "test.csv")
        monkeypatch.setattr(settings, "REPORTS_DIR", reports)
        monkeypatch.setattr(settings, "FIGURES_DIR", reports / "figures")
        monkeypatch.setattr(settings, "METRICS_PATH", reports / "metrics.json")
        monkeypatch.setattr(settings, "TOP_TERMS_PATH", reports / "top_terms.json")
        p = pipeline.predict_proba(test_df["text"].tolist())[:, 1]
        return {
            "pipeline": pipeline,
            "test": test_df,
            "X_test": X_test,
            "y": y_test,
            "p": p,
            "version": version,
            "params": params,
            "reports": reports,
            "features": features,
            "metrics_path": reports / "metrics.json",
            "top_terms_path": reports / "top_terms.json",
        }

    return _stage


def run(stage_dict: dict, helpers) -> dict:
    assert evaluate.main([]) == 0
    return helpers["read_json"](stage_dict["metrics_path"])


# ---------------------------------------------------------------------------
# reports/metrics.json
# ---------------------------------------------------------------------------


def test_metrics_json_has_exactly_the_contract_keys(stage, helpers):
    metrics = run(stage(), helpers)
    assert set(metrics) == METRICS_KEYS
    assert set(metrics["confusion"]) == {"tn", "fp", "fn", "tp"}


def test_metrics_json_reproduces_scikit_learn(stage, helpers):
    s = stage()
    metrics = run(s, helpers)
    y, p = s["y"], s["p"]
    pred = (p >= 0.5).astype(int)
    expected = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "positive_rate_test": y.mean(),
        "majority_baseline_accuracy": max(y.mean(), 1 - y.mean()),
    }
    for key, value in expected.items():
        assert metrics[key] == pytest.approx(round(float(value), 4), abs=1e-4), key


def test_metrics_json_confusion_counts(stage, helpers):
    s = stage()
    metrics = run(s, helpers)
    y, p = s["y"], s["p"]
    pred = p >= 0.5
    confusion = metrics["confusion"]
    assert confusion["tp"] == int((pred & (y == 1)).sum())
    assert confusion["fp"] == int((pred & (y == 0)).sum())
    assert confusion["fn"] == int((~pred & (y == 1)).sum())
    assert confusion["tn"] == int((~pred & (y == 0)).sum())
    assert sum(confusion.values()) == metrics["n_test"] == len(y)
    assert all(isinstance(v, int) for v in confusion.values())


def test_metrics_json_copies_provenance_from_version_json(stage, helpers):
    s = stage()
    metrics = run(s, helpers)
    version = s["version"]
    for key in ("model", "model_version", "trained_at", "git_sha", "data_sha256"):
        assert metrics[key] == version[key], key
    assert metrics["n_train"] == version["n_train"]
    assert metrics["n_features"] == version["n_features"]
    assert metrics["candidates_cv"] == version["candidates"]
    assert metrics["selection_metric"] == version["selection_metric"]
    assert metrics["positive_label"] == "happiness"
    assert metrics["threshold"] == 0.5
    assert metrics["n_test"] == len(s["test"])


def test_metrics_json_rounds_to_four_decimals(stage, helpers):
    metrics = run(stage(), helpers)
    for key in RATE_KEYS:
        assert 0.0 <= metrics[key] <= 1.0, key
        assert metrics[key] == round(metrics[key], 4), key
    assert metrics["log_loss"] >= 0.0
    assert metrics["log_loss"] == round(metrics["log_loss"], 4)


def test_metrics_json_uses_lf(stage, helpers):
    s = stage()
    run(s, helpers)
    raw = s["metrics_path"].read_bytes()
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")


def test_stage_is_deterministic(stage, helpers):
    s = stage()
    first = run(s, helpers)
    second = run(s, helpers)
    assert first == second


# ---------------------------------------------------------------------------
# reports/top_terms.json
# ---------------------------------------------------------------------------


def _check_linear_terms(doc: dict, pipeline, scores: np.ndarray, n: int) -> None:
    """Both lists hold the n most extreme scores, each term carrying its own score.

    Ties are compared by value, not by position, so equal scores may come in any order.
    """
    names = list(pipeline.named_steps["vectorize"].get_feature_names_out())
    by_term = {str(name): float(score) for name, score in zip(names, scores, strict=True)}
    ordered = np.sort(scores)
    expected = {
        "happiness": [round(float(v), 4) for v in ordered[::-1][:n]],
        "sadness": [round(float(-v), 4) for v in ordered[:n]],
    }
    sign = {"happiness": 1.0, "sadness": -1.0}
    for side in ("happiness", "sadness"):
        assert len(doc[side]) == n, side
        assert [t["weight"] for t in doc[side]] == expected[side], side
        for item in doc[side]:
            assert set(item) == {"term", "weight"}
            assert item["weight"] > 0
            assert item["weight"] == round(item["weight"], 4)
            own = round(sign[side] * by_term[item["term"]], 4)
            assert item["weight"] == pytest.approx(own, abs=1e-4), item["term"]


def test_top_terms_for_logistic_regression(stage, helpers):
    s = stage("logreg", top_n=5)
    run(s, helpers)
    doc = helpers["read_json"](s["top_terms_path"])
    assert doc["available"] is True
    assert doc["method"] == "logistic regression coefficients"
    assert set(doc) == {"available", "method", "happiness", "sadness"}
    coef = s["pipeline"].named_steps["classify"].coef_[0]
    _check_linear_terms(doc, s["pipeline"], coef, 5)
    top = doc["happiness"][0]
    assert top["weight"] == pytest.approx(round(float(coef.max()), 4), abs=1e-4)


def test_top_terms_for_naive_bayes(stage, helpers):
    s = stage("nb", top_n=4)
    run(s, helpers)
    doc = helpers["read_json"](s["top_terms_path"])
    assert doc["available"] is True
    assert doc["method"] == "log-probability ratio"
    clf = s["pipeline"].named_steps["classify"]
    ratio = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    _check_linear_terms(doc, s["pipeline"], ratio, 4)
    assert doc["happiness"][0]["weight"] == pytest.approx(round(float(ratio.max()), 4), abs=1e-4)


def test_top_terms_for_xgboost_are_importances(stage, helpers):
    pytest.importorskip("xgboost")
    s = stage("xgboost", top_n=5)
    run(s, helpers)
    doc = helpers["read_json"](s["top_terms_path"])
    assert doc["available"] is False
    assert doc["method"] == "gain importance"
    assert set(doc) == {"available", "method", "terms"}
    assert 1 <= len(doc["terms"]) <= 5
    vocabulary = set(s["pipeline"].named_steps["vectorize"].get_feature_names_out())
    weights = [t["weight"] for t in doc["terms"]]
    assert weights == sorted(weights, reverse=True)
    for item in doc["terms"]:
        assert set(item) == {"term", "weight"}
        assert item["term"] in vocabulary
        assert item["weight"] >= 0


def test_top_terms_count_follows_params(stage, helpers):
    s = stage("logreg", top_n=3)
    run(s, helpers)
    doc = helpers["read_json"](s["top_terms_path"])
    assert len(doc["happiness"]) == 3
    assert len(doc["sadness"]) == 3


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def test_no_figures_when_disabled(stage, helpers):
    s = stage(figures=False)
    run(s, helpers)
    figures = s["reports"] / "figures"
    assert not figures.exists() or not list(figures.glob("*.png"))


def test_stale_figures_are_removed_when_disabled(stage, helpers):
    s = stage(figures=False)
    figures = s["reports"] / "figures"
    figures.mkdir(parents=True)
    for name in ("roc.png", "old_model.png"):
        (figures / name).write_bytes(PNG_MAGIC)
    (figures / "notes.txt").write_text("keep me", encoding="utf-8")
    run(s, helpers)
    assert list(figures.glob("*.png")) == []
    assert (figures / "notes.txt").is_file()


def test_stale_figures_are_removed_when_enabled(stage, helpers):
    pytest.importorskip("matplotlib")
    s = stage("logreg", figures=True)
    figures = s["reports"] / "figures"
    figures.mkdir(parents=True)
    (figures / "old_model.png").write_bytes(PNG_MAGIC)
    run(s, helpers)
    assert not (figures / "old_model.png").exists()
    assert {path.name for path in figures.glob("*.png")} == {
        "roc.png",
        "pr.png",
        "confusion.png",
        "top_terms.png",
    }


def test_clear_figures_creates_the_directory_and_removes_only_pngs(tmp_path: Path):
    figures = tmp_path / "reports" / "figures"
    assert evaluate.clear_figures(figures) == []
    assert figures.is_dir()
    (figures / "b.png").write_bytes(PNG_MAGIC)
    (figures / "a.png").write_bytes(PNG_MAGIC)
    (figures / "notes.txt").write_text("keep me", encoding="utf-8")
    removed = evaluate.clear_figures(figures)
    assert [path.name for path in removed] == ["a.png", "b.png"]
    assert sorted(path.name for path in figures.iterdir()) == ["notes.txt"]
    assert evaluate.clear_figures(figures) == []


def test_figures_are_written_when_enabled(stage, helpers):
    pytest.importorskip("matplotlib")
    s = stage("logreg", figures=True)
    run(s, helpers)
    figures = s["reports"] / "figures"
    for name in ("roc", "pr", "confusion", "top_terms"):
        path = figures / f"{name}.png"
        assert path.is_file(), name
        raw = path.read_bytes()
        assert raw.startswith(PNG_MAGIC), name
        width = int.from_bytes(raw[16:20], "big")
        height = int.from_bytes(raw[20:24], "big")
        assert width >= 600 and height >= 400, name


def test_no_top_terms_figure_without_coefficients(stage, helpers):
    pytest.importorskip("xgboost")
    pytest.importorskip("matplotlib")
    s = stage("xgboost", figures=True)
    run(s, helpers)
    figures = s["reports"] / "figures"
    assert (figures / "roc.png").is_file()
    assert (figures / "confusion.png").is_file()
    assert not (figures / "top_terms.png").exists()


# ---------------------------------------------------------------------------
# Train/serve parity
# ---------------------------------------------------------------------------


def test_parity_check_rejects_a_matrix_that_disagrees_with_the_pipeline(stage, helpers):
    s = stage()
    X_bad = s["X_test"].copy()
    X_bad.data = X_bad.data * 3.0 + 0.5
    helpers["write_npz"](s["features"] / "test.npz", X_bad, s["y"])
    try:
        code = evaluate.main([])
    except (AssertionError, ValueError, RuntimeError):
        return
    assert code != 0
    assert not s["metrics_path"].exists()


def test_parity_check_rejects_labels_that_disagree(stage, helpers):
    s = stage()
    helpers["write_npz"](s["features"] / "test.npz", s["X_test"], 1 - s["y"])
    try:
        code = evaluate.main([])
    except (AssertionError, ValueError, RuntimeError):
        return
    assert code != 0


# ---------------------------------------------------------------------------
# Pure helpers on hand cases
# ---------------------------------------------------------------------------

# Sorted by score the labels read 0, 1, 0, 1, 0, 1 from the top: two of each mistake at 0.5.
Y = np.array([0, 0, 1, 1, 1, 0])
P = np.array([0.1, 0.4, 0.6, 0.9, 0.3, 0.7])


class _Linear:
    coef_ = np.array([[0.5, -1.0, 2.0, 0.0, -0.25]])


class _NaiveBayes:
    feature_log_prob_ = np.log(np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]]))


class _Tree:
    feature_importances_ = np.array([0.1, 0.7, 0.2])


class _Opaque:
    pass


def test_confusion_at_counts_every_row():
    out = evaluate.confusion_at(Y, P, 0.5)
    assert out == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
    assert sum(out.values()) == len(Y)


def test_confusion_at_threshold_is_inclusive():
    assert evaluate.confusion_at([1, 0], [0.5, 0.5], 0.5) == {"tn": 0, "fp": 1, "fn": 0, "tp": 1}
    assert evaluate.confusion_at([1, 0], [0.49, 0.49], 0.5) == {"tn": 1, "fp": 0, "fn": 1, "tp": 0}


@pytest.mark.parametrize(
    ("y", "expected"),
    [([1, 1, 1, 0], 0.75), ([0, 0, 1, 1], 0.5), ([0, 0, 0, 0], 1.0), ([1], 1.0), ([], 0.0)],
    ids=["mostly_positive", "balanced", "all_negative", "one", "empty"],
)
def test_majority_baseline_accuracy(y, expected):
    assert evaluate.majority_baseline_accuracy(np.array(y, dtype=int)) == expected


def test_compute_metrics_hand_case():
    out = evaluate.compute_metrics(Y, P, 0.5)
    assert out["threshold"] == 0.5
    assert out["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert out["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["roc_auc"] == pytest.approx(roc_auc_score(Y, P), abs=1e-4)
    assert out["pr_auc"] == pytest.approx(average_precision_score(Y, P), abs=1e-4)
    assert out["brier"] == pytest.approx(brier_score_loss(Y, P), abs=1e-4)
    assert out["log_loss"] == pytest.approx(log_loss(Y, P), abs=1e-4)
    assert out["confusion"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
    assert out["majority_baseline_accuracy"] == 0.5
    for key, value in out.items():
        if key != "confusion":
            assert value == round(value, 4), key


def test_compute_metrics_without_a_positive_prediction():
    out = evaluate.compute_metrics([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4], 0.5)
    assert out["precision"] == 0.0
    assert out["recall"] == 0.0
    assert out["f1"] == 0.0
    assert out["confusion"] == {"tn": 2, "fp": 0, "fn": 2, "tp": 0}


def test_top_terms_for_a_linear_classifier():
    out = evaluate.top_terms(_Linear(), ["a", "b", "c", "d", "e"], 2)
    assert out["available"] is True
    assert out["method"] == "logistic regression coefficients"
    assert out["happiness"] == [{"term": "c", "weight": 2.0}, {"term": "a", "weight": 0.5}]
    assert out["sadness"] == [{"term": "b", "weight": 1.0}, {"term": "e", "weight": 0.25}]


def test_top_terms_for_naive_bayes_uses_the_log_ratio():
    out = evaluate.top_terms(_NaiveBayes(), ["a", "b", "c"], 3)
    assert out["available"] is True
    assert out["method"] == "log-probability ratio"
    assert out["happiness"][0] == {"term": "a", "weight": round(float(np.log(3.0)), 4)}
    assert out["sadness"][0] == {"term": "b", "weight": round(float(np.log(5.0)), 4)}
    assert all(item["weight"] > 0 for item in out["happiness"] + out["sadness"])


def test_top_terms_for_a_tree_model_are_importances():
    out = evaluate.top_terms(_Tree(), ["a", "b", "c"], 2)
    assert out == {
        "available": False,
        "method": "gain importance",
        "terms": [{"term": "b", "weight": 0.7}, {"term": "c", "weight": 0.2}],
    }


def test_top_terms_for_an_opaque_model_is_unavailable():
    out = evaluate.top_terms(_Opaque(), ["a"], 3)
    assert out["available"] is False
    assert out.get("terms") == []


def test_top_terms_refuses_a_vocabulary_of_the_wrong_size():
    with pytest.raises((RuntimeError, ValueError)):
        evaluate.top_terms(_Linear(), ["a", "b"], 2)


def test_parity_check_accepts_the_pipelines_own_scores(toy_pipeline, corpus_split):
    _, test = corpus_split
    texts = test["text"].tolist()
    p = toy_pipeline.predict_proba(texts)[:, 1]
    assert evaluate.parity_check(toy_pipeline, texts, p) < 1e-6


def test_parity_check_rejects_shifted_scores(toy_pipeline, corpus_split):
    _, test = corpus_split
    texts = test["text"].tolist()
    p = toy_pipeline.predict_proba(texts)[:, 1]
    with pytest.raises((RuntimeError, AssertionError, ValueError)):
        evaluate.parity_check(toy_pipeline, texts, np.clip(p + 0.01, 0.0, 1.0))
    with pytest.raises((RuntimeError, AssertionError, ValueError)):
        evaluate.parity_check(toy_pipeline, texts, p[:-1])


# ---------------------------------------------------------------------------
# Metric helper, when the module exposes one
# ---------------------------------------------------------------------------


def test_metric_helper_on_a_hand_case():
    fn = next((getattr(evaluate, n) for n in METRIC_HELPER_NAMES if hasattr(evaluate, n)), None)
    if fn is None:
        pytest.skip("tweet_emotion.evaluate exposes no metric helper under a known name")
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.4, 0.6, 0.9, 0.3, 0.7])
    out = fn(y, p, 0.5)
    assert out["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert out["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["roc_auc"] == pytest.approx(roc_auc_score(y, p), abs=1e-4)

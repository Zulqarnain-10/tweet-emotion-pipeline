"""Contract tests for the shipped model and its receipts.

Loads models/model.joblib, models/version.json, reports/metrics.json, reports/top_terms.json,
configs/presets.json, and, when the DVC-managed data is present, data/raw/test.csv and
data/features/test.npz. The scoring tests re-measure the headline numbers on the held-out
split and skip when those files are absent (for example in CI before dvc repro).
"""

from __future__ import annotations

import re

import joblib
import numpy as np
import pandas as pd
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
from sklearn.pipeline import Pipeline

from tweet_emotion import __version__, settings
from tweet_emotion.features import load_npz
from tweet_emotion.preprocess import TextNormalizer, load_stopwords

METRIC_TOLERANCE = 1e-4
PARITY_TOLERANCE = 1e-6
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
CV_KEYS = {"cv_roc_auc_mean", "cv_roc_auc_std", "cv_accuracy_mean", "cv_f1_mean", "fit_seconds"}
LIBRARY_KEYS = {"python", "scikit-learn", "numpy", "pandas", "scipy", "nltk", "xgboost", "joblib"}
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
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier",
    "majority_baseline_accuracy",
)
FULL_LABEL_COUNTS = {
    "neutral": 8638,
    "worry": 8459,
    "happiness": 5209,
    "sadness": 5165,
    "love": 3842,
    "surprise": 2187,
    "fun": 1776,
    "relief": 1526,
    "hate": 1323,
    "empty": 827,
    "enthusiasm": 759,
    "boredom": 179,
    "anger": 110,
}
PRESET_IDS = ["clear_happiness", "clear_sadness", "close_call", "model_miss"]
PRESET_KEYS = {"id", "title", "text", "label", "description", "probability_happiness", "tweet_id"}
ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
URL = re.compile(r"https?://\S+|www\.\S+")
COMMITTED_JSON = ("VERSION_PATH", "METRICS_PATH", "TOP_TERMS_PATH", "PRESETS_PATH")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model(helpers, params) -> Pipeline:
    helpers["require"](settings.MODEL_PATH, "run dvc repro (train stage) first")
    if params["preprocess"]["lemmatize"] and not helpers["wordnet_present"]():
        pytest.skip(helpers["WORDNET_HINT"])
    return joblib.load(settings.MODEL_PATH)


@pytest.fixture(scope="module")
def vectorizer(helpers):
    helpers["require"](settings.VECTORIZER_PATH, "run dvc repro (features stage) first")
    return joblib.load(settings.VECTORIZER_PATH)


@pytest.fixture(scope="module")
def held_out(helpers) -> pd.DataFrame:
    helpers["require"](settings.RAW_TEST_PATH, "the split is DVC-managed, run dvc repro first")
    return pd.read_csv(settings.RAW_TEST_PATH, keep_default_na=False)


@pytest.fixture(scope="module")
def test_features(helpers):
    helpers["require"](settings.FEATURES_TEST_PATH, "run dvc repro (features stage) first")
    return load_npz(settings.FEATURES_TEST_PATH)


@pytest.fixture(scope="module")
def scored(model, held_out) -> tuple[np.ndarray, np.ndarray]:
    y = held_out[settings.LABEL_COLUMN].to_numpy().astype(int)
    p = model.predict_proba(held_out[settings.TEXT_COLUMN].astype(str).tolist())[:, 1]
    return y, p


@pytest.fixture(scope="module")
def feature_manifest(helpers) -> dict:
    helpers["require"](settings.FEATURE_MANIFEST_PATH, "run dvc repro (features stage) first")
    return helpers["read_json"](settings.FEATURE_MANIFEST_PATH)


@pytest.fixture(scope="module")
def fetch_manifest(helpers) -> dict:
    helpers["require"](settings.FETCH_MANIFEST_PATH, "run dvc repro (ingest stage) first")
    return helpers["read_json"](settings.FETCH_MANIFEST_PATH)


def qualifies(text: str) -> bool:
    return 6 <= len(text) <= 140 and "@" not in text and not URL.search(text)


# ---------------------------------------------------------------------------
# The pipeline object
# ---------------------------------------------------------------------------


def test_model_is_the_three_step_pipeline(model, params):
    assert isinstance(model, Pipeline)
    assert [name for name, _ in model.steps] == ["normalize", "vectorize", "classify"]
    normalizer = model.named_steps["normalize"]
    assert isinstance(normalizer, TextNormalizer)
    block = params["preprocess"]
    for flag in PREPROCESS_FLAGS:
        assert getattr(normalizer, flag) == block[flag], flag
    if block["remove_stopwords"]:
        assert isinstance(normalizer.stopwords, frozenset)
        keep = {str(word).lower() for word in block.get("keep_words", [])}
        assert normalizer.stopwords == load_stopwords() - keep
    assert hasattr(model.named_steps["vectorize"], "vocabulary_")
    assert hasattr(model.named_steps["classify"], "predict_proba")


def test_pipeline_vectorizer_is_the_saved_one(model, vectorizer):
    assert model.named_steps["vectorize"].vocabulary_ == vectorizer.vocabulary_
    assert type(model.named_steps["vectorize"]) is type(vectorizer)


def test_vectorizer_settings_follow_params(vectorizer, params):
    cfg = params["features"]
    assert vectorizer.lowercase is False
    assert vectorizer.token_pattern == r"(?u)\b\w+\b"
    assert tuple(vectorizer.ngram_range) == tuple(cfg["ngram_range"])
    assert vectorizer.min_df == cfg["min_df"]
    assert vectorizer.max_features == cfg["max_features"]
    assert len(vectorizer.vocabulary_) <= cfg["max_features"]


def test_probabilities_are_well_formed(model, presets):
    texts = [preset["text"] for preset in presets]
    proba = model.predict_proba(texts)
    assert proba.shape == (len(texts), 2)
    assert np.isfinite(proba).all()
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_single_text_equals_batch_prediction(model, presets):
    texts = [preset["text"] for preset in presets]
    batch = model.predict_proba(texts)[:, 1]
    for i, text in enumerate(texts):
        single = model.predict_proba([text])[:, 1]
        assert single.shape == (1,)
        assert single[0] == pytest.approx(batch[i], abs=1e-9)


def test_empty_and_noise_texts_still_score(model):
    proba = model.predict_proba(["", "   ", "@bob 123 !!!", "http://t.co/x"])
    assert proba.shape == (4, 2)
    assert np.isfinite(proba).all()
    assert np.allclose(proba[0], proba[1])
    assert np.allclose(proba[0], proba[2])


def test_prediction_is_deterministic(model, presets):
    texts = [preset["text"] for preset in presets]
    np.testing.assert_array_equal(model.predict_proba(texts), model.predict_proba(texts))


def test_pipeline_survives_a_pickle_round_trip(model, presets, tmp_path):
    joblib.dump(model, tmp_path / "copy.joblib")
    restored = joblib.load(tmp_path / "copy.joblib")
    texts = [preset["text"] for preset in presets]
    np.testing.assert_array_equal(restored.predict_proba(texts), model.predict_proba(texts))


# ---------------------------------------------------------------------------
# Held-out receipts
# ---------------------------------------------------------------------------


def test_held_out_split_is_well_formed(held_out, metrics):
    assert list(held_out.columns) == list(settings.CSV_COLUMNS)
    assert len(held_out) == metrics["n_test"]
    assert held_out[settings.ID_COLUMN].is_unique
    assert held_out[settings.ID_COLUMN].is_monotonic_increasing
    assert set(held_out[settings.LABEL_COLUMN].unique()) == {0, 1}
    assert (held_out[settings.TEXT_COLUMN].astype(str).str.strip() != "").all()
    assert held_out[settings.TEXT_COLUMN].astype(str).str.strip().is_unique


def test_held_out_accuracy_and_roc_auc_match_metrics_json(metrics, scored):
    y, p = scored
    pred = (p >= metrics["threshold"]).astype(int)
    measured = {
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
    }
    for key, value in measured.items():
        assert abs(value - metrics[key]) <= METRIC_TOLERANCE, (
            f"{key}: measured {value:.6f}, reports/metrics.json says {metrics[key]}"
        )


def test_held_out_secondary_metrics_match_metrics_json(metrics, scored):
    y, p = scored
    pred = (p >= metrics["threshold"]).astype(int)
    measured = {
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "positive_rate_test": float(y.mean()),
        "majority_baseline_accuracy": float(max(y.mean(), 1 - y.mean())),
    }
    for key, value in measured.items():
        assert abs(value - metrics[key]) <= METRIC_TOLERANCE, (
            f"{key}: measured {value:.6f}, reports/metrics.json says {metrics[key]}"
        )


def test_confusion_at_threshold_reproduces(metrics, scored):
    y, p = scored
    pred = p >= metrics["threshold"]
    confusion = metrics["confusion"]
    assert confusion["tp"] == int((pred & (y == 1)).sum())
    assert confusion["fp"] == int((pred & (y == 0)).sum())
    assert confusion["fn"] == int((~pred & (y == 1)).sum())
    assert confusion["tn"] == int((~pred & (y == 0)).sum())
    assert sum(confusion.values()) == metrics["n_test"] == len(y)


def test_train_serve_parity_on_the_feature_matrix(model, held_out, test_features):
    X_test, y_npz = test_features
    assert X_test.shape[0] == len(held_out)
    np.testing.assert_array_equal(y_npz, held_out[settings.LABEL_COLUMN].to_numpy())
    from_matrix = model.named_steps["classify"].predict_proba(X_test)[:, 1]
    from_text = model.predict_proba(held_out[settings.TEXT_COLUMN].astype(str).tolist())[:, 1]
    assert float(np.max(np.abs(from_matrix - from_text))) < PARITY_TOLERANCE


def test_feature_matrix_matches_the_vectorizer(model, held_out, test_features):
    X_test, _ = test_features
    normalized = model.named_steps["normalize"].transform(
        held_out[settings.TEXT_COLUMN].astype(str).tolist()
    )
    again = model.named_steps["vectorize"].transform(normalized)
    assert again.shape == X_test.shape
    assert abs(again - X_test).max() < 1e-6


# ---------------------------------------------------------------------------
# models/version.json
# ---------------------------------------------------------------------------


def test_version_json_has_contract_keys(version_info, params, model):
    missing = [key for key in VERSION_KEYS if key not in version_info]
    assert not missing, f"models/version.json lacks {missing}"
    cfg = params["train"]
    assert version_info["model"] in cfg["candidates"]
    assert version_info["package_version"] == __version__
    short = version_info["git_sha_short"]
    assert version_info["model_version"] == f"{__version__}+{short}"
    assert isinstance(version_info["git_dirty"], bool)
    sha = version_info["git_sha"]
    assert sha == "unknown" or re.fullmatch(r"[0-9a-f]{40}", sha)
    assert version_info["data_sha256"] == params["data"]["sha256"]
    assert ISO_Z.match(version_info["trained_at"])
    assert version_info["n_train"] > 0
    assert version_info["n_features"] == len(model.named_steps["vectorize"].vocabulary_)
    assert version_info["selection_metric"] == cfg["selection_metric"]
    assert version_info["cv_folds"] == cfg["cv_folds"]
    assert set(version_info["candidates"]) == set(cfg["candidates"])
    for name, block in version_info["candidates"].items():
        assert set(block) == CV_KEYS, name
    assert version_info["winner_params"] == cfg[version_info["model"]]
    assert set(version_info["libraries"]) == LIBRARY_KEYS


def test_version_json_winner_has_the_best_cv_score(version_info):
    key = f"cv_{version_info['selection_metric']}_mean"
    scores = {name: block[key] for name, block in version_info["candidates"].items()}
    best = max(scores.values())
    tied = [name for name, score in scores.items() if score == best]
    assert version_info["model"] == tied[0]


def test_version_json_classifier_matches_the_winner(version_info, model):
    clf = model.named_steps["classify"]
    expected = {
        "logreg": "LogisticRegression",
        "nb": "MultinomialNB",
        "xgboost": "XGBClassifier",
    }[version_info["model"]]
    assert type(clf).__name__ == expected


# ---------------------------------------------------------------------------
# reports/metrics.json
# ---------------------------------------------------------------------------


def test_metrics_json_has_contract_keys(metrics, params):
    missing = [key for key in METRICS_KEYS if key not in metrics]
    assert not missing, f"reports/metrics.json lacks {missing}"
    assert set(metrics["confusion"]) == {"tn", "fp", "fn", "tp"}
    assert metrics["positive_label"] == "happiness"
    assert metrics["threshold"] == params["evaluate"]["threshold"]
    assert metrics["selection_metric"] == params["train"]["selection_metric"]
    assert set(metrics["candidates_cv"]) == set(params["train"]["candidates"])


def test_metrics_json_numbers_are_finite_and_in_range(metrics):
    for key in RATE_KEYS:
        value = metrics[key]
        assert isinstance(value, float | int) and not isinstance(value, bool), key
        assert np.isfinite(value), key
        assert 0.0 <= value <= 1.0, key
        assert value == round(value, 4), key
    assert np.isfinite(metrics["log_loss"]) and metrics["log_loss"] >= 0.0
    assert metrics["log_loss"] == round(metrics["log_loss"], 4)
    for key in ("n_train", "n_test", "n_features"):
        assert isinstance(metrics[key], int) and metrics[key] > 0, key
    for key, value in metrics["confusion"].items():
        assert isinstance(value, int) and value >= 0, key
    assert sum(metrics["confusion"].values()) == metrics["n_test"]
    assert metrics["majority_baseline_accuracy"] >= 0.5
    assert metrics["accuracy"] > metrics["majority_baseline_accuracy"]
    assert metrics["roc_auc"] > 0.5
    for name, block in metrics["candidates_cv"].items():
        assert set(block) == CV_KEYS, name
        for key in CV_KEYS - {"fit_seconds"}:
            assert 0.0 <= block[key] <= 1.0, (name, key)
        assert block["fit_seconds"] >= 0.0


def test_metrics_json_agrees_with_version_json(metrics, version_info):
    for key in ("model", "model_version", "trained_at", "git_sha", "data_sha256"):
        assert metrics[key] == version_info[key], key
    assert metrics["n_train"] == version_info["n_train"]
    assert metrics["n_features"] == version_info["n_features"]
    assert metrics["candidates_cv"] == version_info["candidates"]
    assert metrics["selection_metric"] == version_info["selection_metric"]


# ---------------------------------------------------------------------------
# reports/top_terms.json and figures
# ---------------------------------------------------------------------------


def test_top_terms_shape_matches_the_classifier(top_terms, model, params):
    n = params["evaluate"]["top_terms"]
    vocabulary = set(model.named_steps["vectorize"].get_feature_names_out())
    clf = model.named_steps["classify"]
    if top_terms["available"]:
        assert set(top_terms) == {"available", "method", "happiness", "sadness"}
        assert top_terms["method"] in {"logistic regression coefficients", "log-probability ratio"}
        for side in ("happiness", "sadness"):
            assert len(top_terms[side]) == n
            for item in top_terms[side]:
                assert set(item) == {"term", "weight"}
                assert item["term"] in vocabulary
                assert item["weight"] > 0
                assert item["weight"] == round(item["weight"], 4)
            weights = [item["weight"] for item in top_terms[side]]
            assert weights == sorted(weights, reverse=True)
        assert hasattr(clf, "coef_") or hasattr(clf, "feature_log_prob_")
    else:
        assert set(top_terms) == {"available", "method", "terms"}
        assert top_terms["method"] == "gain importance"
        assert 1 <= len(top_terms["terms"]) <= n
        assert all(item["term"] in vocabulary for item in top_terms["terms"])
        assert hasattr(clf, "feature_importances_")


def test_top_terms_are_the_extreme_coefficients(top_terms, model):
    clf = model.named_steps["classify"]
    if not hasattr(clf, "coef_"):
        pytest.skip("the shipped classifier is not linear")
    names = model.named_steps["vectorize"].get_feature_names_out()
    coef = clf.coef_[0]
    n = len(top_terms["happiness"])
    order = np.argsort(coef)
    assert [t["term"] for t in top_terms["happiness"]] == [str(names[i]) for i in order[::-1][:n]]
    assert [t["term"] for t in top_terms["sadness"]] == [str(names[i]) for i in order[:n]]
    top_happiness = top_terms["happiness"][0]["weight"]
    top_sadness = top_terms["sadness"][0]["weight"]
    assert top_happiness == pytest.approx(round(float(coef.max()), 4), abs=1e-4)
    assert top_sadness == pytest.approx(round(float(-coef.min()), 4), abs=1e-4)


def test_figures_exist_when_enabled(metrics, top_terms, params):
    if not params["evaluate"]["figures"]:
        pytest.skip("figures are disabled in params.yaml")
    for name in ("roc", "pr", "confusion"):
        path = settings.FIGURES_DIR / f"{name}.png"
        assert path.is_file(), name
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), name
    assert (settings.FIGURES_DIR / "top_terms.png").is_file() == bool(top_terms["available"])


# ---------------------------------------------------------------------------
# configs/presets.json
# ---------------------------------------------------------------------------


def test_presets_file_shape(presets_doc, version_info):
    assert set(presets_doc) == {"presets", "chosen_by", "model_version"}
    assert presets_doc["chosen_by"] == "model scores on the held-out test split"
    assert presets_doc["model_version"] == version_info["model_version"]
    assert [preset["id"] for preset in presets_doc["presets"]] == PRESET_IDS
    for preset in presets_doc["presets"]:
        assert set(preset) == PRESET_KEYS, preset["id"]
        assert preset["title"] == preset["id"].replace("_", " ").capitalize()
        assert preset["label"] in settings.LABELS
        assert preset["description"].endswith(".")
        assert preset["description"][0].isupper()
        assert "—" not in preset["description"]
        assert 0.0 <= preset["probability_happiness"] <= 1.0
        assert preset["probability_happiness"] == round(preset["probability_happiness"], 4)
        assert isinstance(preset["tweet_id"], int)


def test_presets_scores_match_the_model(presets, model):
    texts = [preset["text"] for preset in presets]
    p = model.predict_proba(texts)[:, 1]
    for value, preset in zip(p, presets, strict=True):
        assert abs(float(value) - preset["probability_happiness"]) < METRIC_TOLERANCE


def test_presets_are_held_out_rows(presets, held_out):
    by_id = held_out.set_index(settings.ID_COLUMN)
    for preset in presets:
        assert preset["tweet_id"] in by_id.index, preset["id"]
        row = by_id.loc[preset["tweet_id"]]
        assert preset["text"] == str(row[settings.TEXT_COLUMN])
        assert preset["label"] == settings.LABELS[int(row[settings.LABEL_COLUMN])]


def test_presets_follow_the_selection_rule(presets, held_out, scored, metrics):
    y, p = scored
    ids = held_out[settings.ID_COLUMN].to_numpy()
    texts = held_out[settings.TEXT_COLUMN].astype(str).tolist()
    ok = np.array([qualifies(t) for t in texts])
    if not ok.any():
        ok = np.ones(len(y), dtype=bool)
    pred = (p >= metrics["threshold"]).astype(int)
    wrong = (pred != y) & ok
    got = {preset["id"]: preset["tweet_id"] for preset in presets}

    def pick(mask, score, best) -> int:
        idx = np.flatnonzero(mask)
        return int(ids[idx[best(score[idx])]])

    assert got["clear_happiness"] == pick((y == 1) & ok, p, np.argmax)
    assert got["clear_sadness"] == pick((y == 0) & ok, p, np.argmin)
    assert got["close_call"] == pick(ok, np.abs(p - 0.5), np.argmin)
    if wrong.any():
        assert got["model_miss"] == pick(wrong, np.abs(p - 0.5), np.argmax)


def test_model_miss_is_wrong_and_the_clear_ones_are_right(presets, metrics):
    by_id = {preset["id"]: preset for preset in presets}
    threshold = metrics["threshold"]

    def predicted(preset: dict) -> str:
        return "happiness" if preset["probability_happiness"] >= threshold else "sadness"

    assert predicted(by_id["model_miss"]) != by_id["model_miss"]["label"]
    assert by_id["clear_happiness"]["label"] == "happiness"
    assert predicted(by_id["clear_happiness"]) == "happiness"
    assert by_id["clear_sadness"]["label"] == "sadness"
    assert predicted(by_id["clear_sadness"]) == "sadness"


# ---------------------------------------------------------------------------
# Manifests and hygiene
# ---------------------------------------------------------------------------


def test_feature_manifest_agrees_with_the_model(feature_manifest, version_info, metrics, params):
    assert feature_manifest["vocabulary_size"] == version_info["n_features"]
    assert feature_manifest["n_train"] == version_info["n_train"]
    assert feature_manifest["n_test"] == metrics["n_test"]
    assert feature_manifest["method"] == params["features"]["method"]
    assert feature_manifest["max_features"] == params["features"]["max_features"]
    assert 0.0 < feature_manifest["density_train"] < 1.0


def test_fetch_manifest_describes_the_public_dataset(fetch_manifest, params, version_info, metrics):
    assert fetch_manifest["sha256"] == params["data"]["sha256"]
    assert fetch_manifest["source_url"] == params["data"]["url"]
    assert fetch_manifest["n_rows_total"] == 40_000
    assert fetch_manifest["label_counts_total"] == FULL_LABEL_COUNTS
    assert fetch_manifest["keep_labels"] == list(params["data"]["keep_labels"])
    assert fetch_manifest["positive_label"] == params["data"]["positive_label"]
    assert fetch_manifest["n_train"] == version_info["n_train"]
    assert fetch_manifest["n_test"] == metrics["n_test"]
    assert fetch_manifest["n_train"] + fetch_manifest["n_test"] == fetch_manifest["n_kept"]
    assert fetch_manifest["n_kept"] <= FULL_LABEL_COUNTS["happiness"] + FULL_LABEL_COUNTS["sadness"]
    assert fetch_manifest["positive_rate_test"] == metrics["positive_rate_test"]


@pytest.mark.parametrize("name", COMMITTED_JSON)
def test_committed_receipts_use_lf(name, helpers):
    path = getattr(settings, name)
    helpers["require"](path, "run dvc repro first")
    raw = path.read_bytes()
    assert b"\r\n" not in raw, name
    assert raw.endswith(b"\n"), name

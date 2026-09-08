"""Contract tests for tweet_emotion.api.schemas: request limits from params.yaml, the
prediction shape, batch bounds, health, and the version payload.

Pure pydantic: no model, no app.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from tweet_emotion import settings
from tweet_emotion.api.schemas import (
    BatchRequest,
    BatchResponse,
    Health,
    Prediction,
    PredictRequest,
    TermContribution,
    VersionInfo,
)

PREDICTION_KEYS = {
    "label",
    "probability_happiness",
    "probability",
    "threshold",
    "normalized_text",
    "terms",
    "model",
    "model_version",
    "disclaimer",
}


def prediction(**overrides) -> dict:
    doc = {
        "label": "happiness",
        "probability_happiness": 0.9123,
        "probability": 0.9123,
        "threshold": 0.5,
        "normalized_text": "happy day",
        "terms": [{"term": "happy", "contribution": 1.2345}],
        "model": "logreg",
        "model_version": "0.1.0+abc1234",
        "disclaimer": settings.DISCLAIMER,
    }
    doc.update(overrides)
    return doc


@pytest.fixture(scope="module")
def max_chars(params) -> int:
    return int(params["api"]["max_chars"])


@pytest.fixture(scope="module")
def batch_max(params) -> int:
    return int(params["api"]["batch_max"])


# ---------------------------------------------------------------------------
# PredictRequest
# ---------------------------------------------------------------------------


def test_schemas_are_pydantic_v2_models():
    for model in (PredictRequest, Prediction, BatchRequest, BatchResponse, Health, VersionInfo):
        assert issubclass(model, BaseModel)
        assert hasattr(model, "model_validate")


def test_predict_request_accepts_text():
    request = PredictRequest(text="I am so happy today")
    assert request.text == "I am so happy today"


def test_predict_request_accepts_the_maximum_length(max_chars):
    assert len(PredictRequest(text="x" * max_chars).text) == max_chars


def test_predict_request_rejects_one_over_the_maximum(max_chars):
    with pytest.raises(ValidationError) as excinfo:
        PredictRequest(text="x" * (max_chars + 1))
    errors = excinfo.value.errors()
    assert any(e["loc"] == ("text",) and e["type"] == "string_too_long" for e in errors)


def test_predict_request_rejects_empty_text():
    with pytest.raises(ValidationError) as excinfo:
        PredictRequest(text="")
    assert any(e["loc"] == ("text",) for e in excinfo.value.errors())


@pytest.mark.parametrize(
    "text", [" ", "   ", "\n\t ", "  \r\n"], ids=["space", "spaces", "mixed", "crlf"]
)
def test_predict_request_rejects_whitespace_only(text):
    with pytest.raises(ValidationError) as excinfo:
        PredictRequest(text=text)
    assert any(e["loc"] == ("text",) for e in excinfo.value.errors())


@pytest.mark.parametrize(
    "value", [123, None, ["a"], {"a": 1}, 1.5], ids=["int", "none", "list", "dict", "float"]
)
def test_predict_request_rejects_non_strings(value):
    with pytest.raises(ValidationError):
        PredictRequest(text=value)


def test_predict_request_requires_text():
    with pytest.raises(ValidationError) as excinfo:
        PredictRequest()
    assert any(e["type"] == "missing" and e["loc"] == ("text",) for e in excinfo.value.errors())


def test_predict_request_json_schema_carries_the_limits(max_chars):
    schema = PredictRequest.model_json_schema()
    text = schema["properties"]["text"]
    assert text["maxLength"] == max_chars
    assert text["minLength"] == 1
    assert schema["required"] == ["text"]


def test_predict_request_limit_comes_from_params(max_chars):
    assert max_chars == settings.load_params()["api"]["max_chars"]


def test_predict_request_keeps_unicode():
    text = "Café au lait, quelle joie"
    assert PredictRequest(text=text).text == text


# ---------------------------------------------------------------------------
# Prediction and TermContribution
# ---------------------------------------------------------------------------


def test_prediction_has_exactly_the_contract_fields():
    assert set(Prediction.model_fields) == PREDICTION_KEYS


def test_prediction_round_trips():
    doc = prediction()
    out = Prediction(**doc)
    assert out.model_dump() == doc
    assert out.terms is not None
    assert isinstance(out.terms[0], TermContribution)


@pytest.mark.parametrize("label", ["happiness", "sadness"])
def test_prediction_accepts_both_labels(label):
    assert Prediction(**prediction(label=label)).label == label


@pytest.mark.parametrize("label", ["joy", "HAPPINESS", "", None, 1])
def test_prediction_rejects_other_labels(label):
    with pytest.raises(ValidationError):
        Prediction(**prediction(label=label))


def test_prediction_terms_may_be_none():
    assert Prediction(**prediction(terms=None)).terms is None


def test_prediction_terms_may_be_empty():
    assert Prediction(**prediction(terms=[])).terms == []


def test_prediction_terms_description_names_both_empty_shapes():
    """The page renders [] and null differently, so the schema must say which is which."""
    field = Prediction.model_fields["terms"]
    description = str(field.description).lower()
    assert "empty list" in description
    assert "linear" in description
    assert "null" in description
    published = Prediction.model_json_schema()["properties"]["terms"]["description"]
    assert published == field.description


def test_term_contribution_fields():
    assert set(TermContribution.model_fields) == {"term", "contribution"}
    item = TermContribution(term="happy", contribution=-0.25)
    assert item.term == "happy"
    assert item.contribution == -0.25


def test_prediction_rejects_a_non_numeric_probability():
    with pytest.raises(ValidationError):
        Prediction(**prediction(probability_happiness="high"))


def test_prediction_requires_the_disclaimer():
    doc = prediction()
    doc.pop("disclaimer")
    with pytest.raises(ValidationError):
        Prediction(**doc)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


def test_batch_request_accepts_the_maximum(batch_max):
    request = BatchRequest(items=[{"text": "fine"}] * batch_max)
    assert len(request.items) == batch_max
    assert all(isinstance(item, PredictRequest) for item in request.items)


def test_batch_request_rejects_one_over_the_maximum(batch_max):
    with pytest.raises(ValidationError) as excinfo:
        BatchRequest(items=[{"text": "fine"}] * (batch_max + 1))
    errors = excinfo.value.errors()
    assert any(e["loc"][0] == "items" for e in errors)


def test_batch_request_validates_each_item():
    with pytest.raises(ValidationError) as excinfo:
        BatchRequest(items=[{"text": "fine"}, {"text": "   "}])
    assert any(e["loc"][:2] == ("items", 1) for e in excinfo.value.errors())


def test_batch_request_requires_items():
    with pytest.raises(ValidationError):
        BatchRequest()


def test_batch_limit_comes_from_params(batch_max):
    assert batch_max == settings.load_params()["api"]["batch_max"]


def test_batch_response_shape():
    out = BatchResponse(count=1, predictions=[Prediction(**prediction())])
    assert out.count == 1
    assert len(out.predictions) == 1
    assert set(BatchResponse.model_fields) == {"count", "predictions"}


# ---------------------------------------------------------------------------
# Health and VersionInfo
# ---------------------------------------------------------------------------


def test_health_fields():
    assert set(Health.model_fields) == {"status", "model_loaded", "model_version"}
    health = Health(status="ok", model_loaded=True, model_version="0.1.0+abc1234")
    assert health.model_dump() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "0.1.0+abc1234",
    }


def test_health_status_is_a_literal_ok():
    with pytest.raises(ValidationError):
        Health(status="down", model_loaded=False, model_version="x")


def test_version_info_carries_metrics_and_the_disclaimer():
    fields = VersionInfo.model_fields
    assert "disclaimer" in fields
    assert "metrics" in fields
    assert "loadtest" in fields
    assert not fields["loadtest"].is_required()


def test_version_info_accepts_a_version_payload():
    doc = {
        "model": "logreg",
        "package_version": "0.1.0",
        "model_version": "0.1.0+abc1234",
        "git_sha": "abc1234" * 5 + "abcde",
        "git_sha_short": "abc1234",
        "git_dirty": False,
        "data_sha256": "0" * 64,
        "trained_at": "2026-01-01T00:00:00Z",
        "n_train": 100,
        "n_features": 50,
        "selection_metric": "roc_auc",
        "cv_folds": 5,
        "candidates": {
            "logreg": {
                "cv_roc_auc_mean": 0.9,
                "cv_roc_auc_std": 0.01,
                "cv_accuracy_mean": 0.8,
                "cv_f1_mean": 0.8,
                "fit_seconds": 0.5,
            }
        },
        "winner_params": {"C": 1.0, "max_iter": 1000},
        "libraries": {"python": "3.12.0"},
        "metrics": {
            "accuracy": 0.8,
            "f1": 0.8,
            "roc_auc": 0.9,
            "pr_auc": 0.9,
            "n_test": 25,
            "trained_at": "2026-01-01T00:00:00Z",
            "threshold": 0.5,
        },
        "params": {"max_chars": 1000, "batch_max": 100},
        "disclaimer": settings.DISCLAIMER,
    }
    info = VersionInfo.model_validate(doc)
    dumped = info.model_dump()
    assert dumped["model_version"] == "0.1.0+abc1234"
    assert dumped["disclaimer"] == settings.DISCLAIMER
    assert dumped["metrics"]["roc_auc"] == 0.9
    assert dumped.get("loadtest") is None

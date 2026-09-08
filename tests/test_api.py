"""API tests against the real, shipped model artifacts.

Every test that needs the model goes through the session client, which skips when
models/model.joblib is absent. The entry point and the logging setup are tested with the
network call stubbed out.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from tweet_emotion import __version__, preprocess, settings
from tweet_emotion.api import __main__ as api_main
from tweet_emotion.api import app as app_module
from tweet_emotion.api import model_loader
from tweet_emotion.api.app import create_app

JSON_HEADERS = {"content-type": "application/json"}
TITLE = "Tweet emotion pipeline"
MAX_BODY_BYTES = int(app_module.MAX_BODY_BYTES)
# The widest one character gets under json.dumps: an astral character written as a
# surrogate pair, \uXXXX\uXXXX.
JSON_BYTES_PER_CHAR_WORST_CASE = 12
HEAD_PATHS = ["/", "/health", "/version", "/presets", "/terms"]
STATIC_DIR = Path(getattr(app_module, "STATIC_DIR", Path(app_module.__file__).parent / "static"))
VERSION_METRIC_KEYS = {"accuracy", "f1", "roc_auc", "pr_auc", "n_test", "trained_at", "threshold"}
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
LOG_KEYS = {"ts", "label", "probability_happiness", "n_chars", "model_version"}
PRESET_IDS = ["clear_happiness", "clear_sadness", "close_call", "model_miss"]


@pytest.fixture(scope="module")
def max_chars(params) -> int:
    return int(params["api"]["max_chars"])


@pytest.fixture(scope="module")
def batch_max(params) -> int:
    return int(params["api"]["batch_max"])


@pytest.fixture
def bundle(client):
    found = getattr(client.app.state, "bundle", None)
    return found if found is not None else model_loader.load_bundle()


def is_linear(bundle) -> bool:
    return hasattr(bundle.pipeline.named_steps["classify"], "coef_")


# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------


def test_body_cap_is_the_contract_value():
    assert MAX_BODY_BYTES == 1_250_000


def test_body_cap_covers_a_full_batch_at_the_worst_case_escaping(batch_max, max_chars):
    assert batch_max * max_chars * JSON_BYTES_PER_CHAR_WORST_CASE < MAX_BODY_BYTES


def test_openapi_description_states_the_cap_in_bytes(client):
    description = client.get("/openapi.json").json()["info"]["description"]
    assert f"{MAX_BODY_BYTES:,} bytes" in description
    assert "413" in description


def test_health(client, version_info):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": version_info["model_version"],
    }


def test_version_keys(client, version_info, metrics):
    body = client.get("/version").json()
    for key, value in version_info.items():
        assert body[key] == value, key
    assert body["disclaimer"] == settings.DISCLAIMER
    assert set(body["metrics"]) == VERSION_METRIC_KEYS
    for key in VERSION_METRIC_KEYS:
        assert body["metrics"][key] == metrics[key], key
    if settings.LOADTEST_PATH.is_file():
        assert {"p95_ms", "host"} <= set(body["loadtest"])
    else:
        assert body.get("loadtest") is None


def test_version_loadtest_block_when_report_exists(client, bundle, tmp_path, monkeypatch):
    doc = {"p95_ms": 12.5, "host": "test host", "rps": 80.0, "p50_ms": 4.0}
    report = tmp_path / "loadtest.json"
    report.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(settings, "LOADTEST_PATH", report)
    monkeypatch.setattr(bundle, "loadtest", doc, raising=False)
    body = client.get("/version").json()
    assert body["loadtest"]["p95_ms"] == 12.5
    assert body["loadtest"]["host"] == "test host"
    assert body["loadtest"]["rps"] == 80.0


def test_presets_endpoint(client, presets):
    body = client.get("/presets").json()
    assert body["presets"] == presets
    assert [preset["id"] for preset in body["presets"]] == PRESET_IDS


def test_terms_endpoint(client):
    response = client.get("/terms")
    if settings.TOP_TERMS_PATH.is_file():
        assert response.status_code == 200
        with open(settings.TOP_TERMS_PATH, encoding="utf-8") as fh:
            assert response.json() == json.load(fh)
    else:
        assert response.status_code == 404


def test_terms_404_when_not_loaded(client, bundle, monkeypatch):
    monkeypatch.setattr(bundle, "top_terms", None)
    assert client.get("/terms").status_code == 404


def test_unknown_route_404_and_wrong_method_405(client):
    assert client.get("/nope").status_code == 404
    assert client.get("/predict").status_code == 405
    assert client.head("/predict").status_code == 405


def test_head_health_and_root_return_200(client):
    assert client.head("/health").status_code == 200
    assert client.head("/").status_code == 200


@pytest.mark.parametrize("path", HEAD_PATHS)
def test_head_matches_get_without_a_body(client, path):
    get = client.get(path)
    head = client.head(path)
    assert head.status_code == get.status_code
    assert head.content == b""
    assert head.headers["content-type"] == get.headers["content-type"]
    assert head.headers.get("content-length") == get.headers.get("content-length")


def test_head_routes_stay_out_of_the_openapi_spec(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/" not in paths
    for path in HEAD_PATHS[1:]:
        assert set(paths[path]) == {"get"}, path


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------


def test_predict_happy_path(client, bundle, preset_text, version_info, metrics):
    response = client.post("/predict", json={"text": preset_text})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == PREDICTION_KEYS
    assert body["label"] in settings.LABELS
    p = body["probability_happiness"]
    assert 0.0 <= p <= 1.0
    assert p == round(p, 4)
    assert body["threshold"] == metrics["threshold"]
    expected_label = "happiness" if p >= body["threshold"] else "sadness"
    assert body["label"] == expected_label
    expected_probability = p if body["label"] == "happiness" else 1 - p
    assert body["probability"] == pytest.approx(expected_probability, abs=1e-4)
    assert body["normalized_text"] == bundle.normalized([preset_text])[0]
    assert body["model"] == version_info["model"]
    assert body["model_version"] == version_info["model_version"]
    assert body["disclaimer"] == settings.DISCLAIMER


@pytest.mark.parametrize("index", [0, 1, 2, 3], ids=PRESET_IDS)
def test_predict_matches_each_preset(client, presets, index):
    preset = presets[index]
    response = client.post("/predict", json={"text": preset["text"]})
    assert response.status_code == 200
    assert abs(response.json()["probability_happiness"] - preset["probability_happiness"]) < 1e-4


def test_predict_is_deterministic(client, preset_text):
    first = client.post("/predict", json={"text": preset_text}).json()
    second = client.post("/predict", json={"text": preset_text}).json()
    assert first == second


def test_predict_agrees_with_the_bundle(client, bundle, preset_text):
    body = client.post("/predict", json={"text": preset_text}).json()
    raw = float(bundle.predict_texts([preset_text])[0])
    assert abs(body["probability_happiness"] - round(raw, 4)) < 1e-6


@pytest.mark.parametrize(
    ("p", "label", "probability"),
    [(0.9, "happiness", 0.9), (0.1, "sadness", 0.9), (0.5, "happiness", 0.5)],
    ids=["happy", "sad", "at_threshold"],
)
def test_predict_label_follows_the_threshold(
    client, bundle, monkeypatch, preset_text, p, label, probability
):
    if bundle.threshold != 0.5:
        pytest.skip("this case assumes the params.yaml threshold of 0.5")
    clf = bundle.pipeline.named_steps["classify"]

    def fixed(X, *args, **kwargs) -> np.ndarray:
        return np.tile([1.0 - p, p], (X.shape[0], 1))

    # Patch the classifier itself so every scoring path in the bundle sees the same number.
    monkeypatch.setattr(clf, "predict_proba", fixed)
    body = client.post("/predict", json={"text": preset_text}).json()
    assert body["label"] == label
    assert body["probability_happiness"] == p
    assert body["probability"] == pytest.approx(probability, abs=1e-6)


def test_predict_terms_shape(client, bundle, preset_text):
    body = client.post("/predict", json={"text": preset_text}).json()
    terms = body["terms"]
    if not is_linear(bundle):
        assert terms is None
        return
    assert isinstance(terms, list)
    assert 0 < len(terms) <= 8
    for item in terms:
        assert set(item) == {"term", "contribution"}
        assert item["contribution"] == round(item["contribution"], 4)
    magnitudes = [abs(item["contribution"]) for item in terms]
    assert magnitudes == sorted(magnitudes, reverse=True)
    normalized = body["normalized_text"]
    for item in terms:
        assert all(token in normalized.split() for token in item["term"].split())


def test_predict_terms_none_when_explain_returns_none(client, bundle, monkeypatch, preset_text):
    name = "explain_matrix" if hasattr(bundle, "explain_matrix") else "explain"
    monkeypatch.setattr(bundle, name, lambda *args, **kwargs: None)
    body = client.post("/predict", json={"text": preset_text}).json()
    assert body["terms"] is None


def test_predict_text_of_only_noise(client, bundle):
    body = client.post("/predict", json={"text": "@bob 123 !!!"}).json()
    assert body["normalized_text"] == ""
    assert 0.0 <= body["probability_happiness"] <= 1.0
    # The page renders the two shapes differently: [] is a linear model with no
    # vocabulary hit, null is a model without per-term coefficients.
    if is_linear(bundle):
        assert body["terms"] == []
    else:
        assert body["terms"] is None


def test_predict_unicode_text(client):
    response = client.post("/predict", json={"text": "Café au lait, quelle joie ☀"})
    assert response.status_code == 200


def test_predict_at_the_maximum_length(client, max_chars):
    response = client.post("/predict", json={"text": "happy " * (max_chars // 6)})
    assert response.status_code == 200
    assert client.post("/predict", json={"text": "x" * max_chars}).status_code == 200


def test_predict_over_length_422(client, max_chars):
    response = client.post("/predict", json={"text": "x" * (max_chars + 1)})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(item["loc"][-1] == "text" and item["type"] == "string_too_long" for item in detail)


@pytest.mark.parametrize("text", ["", " ", "   \n\t  "], ids=["empty", "space", "mixed"])
def test_predict_blank_text_422(client, text):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 422
    assert any(item["loc"][-1] == "text" for item in response.json()["detail"])


def test_predict_missing_text_422(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(item["loc"][-1] == "text" and item["type"] == "missing" for item in detail)


def test_predict_non_string_text_422(client):
    response = client.post("/predict", json={"text": 12345})
    assert response.status_code == 422
    assert any(item["loc"][-1] == "text" for item in response.json()["detail"])


def test_predict_invalid_json_422(client):
    response = client.post("/predict", content=b"{not json", headers=JSON_HEADERS)
    assert response.status_code == 422


def test_predict_body_over_cap_413(client):
    raw = json.dumps({"text": "x" * 1_300_000})
    assert len(raw) > MAX_BODY_BYTES
    response = client.post("/predict", content=raw, headers=JSON_HEADERS)
    assert response.status_code == 413


def test_predict_chunked_body_over_cap_413(client):
    async def chunks():
        for _ in range(4):
            yield b"[" + b" " * (MAX_BODY_BYTES // 2)

    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
            return await http.post("/predict", content=chunks(), headers=JSON_HEADERS)

    response = asyncio.run(run())
    assert "content-length" not in response.request.headers
    assert response.status_code == 413


def test_predict_body_under_cap_is_validated_not_refused(client, max_chars):
    raw = json.dumps({"text": "x" * (max_chars + 4_000)})
    assert len(raw) < MAX_BODY_BYTES
    response = client.post("/predict", content=raw, headers=JSON_HEADERS)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------------------------


def test_predict_batch_of_presets(client, presets):
    items = [{"text": preset["text"]} for preset in presets]
    response = client.post("/predict/batch", json={"items": items})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"predictions", "count"}
    assert body["count"] == len(presets)
    assert len(body["predictions"]) == len(presets)
    for prediction, preset in zip(body["predictions"], presets, strict=True):
        assert set(prediction) == PREDICTION_KEYS
        assert abs(prediction["probability_happiness"] - preset["probability_happiness"]) < 1e-4
        assert prediction["disclaimer"] == settings.DISCLAIMER


def test_predict_batch_matches_single_calls(client, presets):
    items = [{"text": preset["text"]} for preset in presets]
    batch = client.post("/predict/batch", json={"items": items}).json()["predictions"]
    singles = [client.post("/predict", json=item).json() for item in items]
    assert batch == singles


def test_predict_batch_over_cap_422(client, batch_max):
    items = [{"text": "fine"}] * (batch_max + 1)
    response = client.post("/predict/batch", json={"items": items})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("items" in item["loc"] for item in detail)


def test_predict_batch_at_cap_200(client, batch_max):
    items = [{"text": "fine"}] * batch_max
    response = client.post("/predict/batch", json={"items": items})
    assert response.status_code == 200
    assert response.json()["count"] == batch_max


def test_predict_batch_of_full_length_ascii_texts_reaches_validation(client, batch_max, max_chars):
    text = ("so happy about today " * max_chars)[:max_chars]
    assert len(text) == max_chars and text.isascii()
    raw = json.dumps({"items": [{"text": text}] * batch_max})
    assert len(raw) < MAX_BODY_BYTES
    response = client.post("/predict/batch", content=raw, headers=JSON_HEADERS)
    assert response.status_code == 200
    assert response.json()["count"] == batch_max


def test_predict_batch_at_the_worst_case_escaping_fits_the_cap(client, batch_max, max_chars):
    # One astral character is one character to the validator and twelve bytes on the wire.
    text = "\U0001f600" * max_chars
    raw = json.dumps({"items": [{"text": text}] * batch_max})
    assert len(raw) < MAX_BODY_BYTES
    assert len(raw) > batch_max * max_chars * JSON_BYTES_PER_CHAR_WORST_CASE
    response = client.post("/predict/batch", content=raw, headers=JSON_HEADERS)
    assert response.status_code == 200
    assert response.json()["count"] == batch_max


def test_predict_batch_validates_each_item(client):
    response = client.post("/predict/batch", json={"items": [{"text": "fine"}, {"text": "   "}]})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(item["loc"][:3] == ["body", "items", 1] for item in detail)


# ---------------------------------------------------------------------------
# Prediction log and metrics
# ---------------------------------------------------------------------------


def test_prediction_log_line_never_carries_the_text(tmp_path, monkeypatch, preset_text):
    log_path = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_path))
    with TestClient(create_app()) as fresh:
        response = fresh.post("/predict", json={"text": preset_text})
        assert response.status_code == 200
    body = response.json()
    raw = log_path.read_bytes().decode("utf-8")
    assert b"\r\n" not in log_path.read_bytes()
    lines = raw.split("\n")
    assert lines[-1] == ""
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert set(record) == LOG_KEYS
    assert record["ts"].endswith("Z")
    assert record["label"] == body["label"]
    assert abs(record["probability_happiness"] - body["probability_happiness"]) < 1e-4
    assert record["n_chars"] == len(preset_text.strip())
    assert record["model_version"] == body["model_version"]
    assert preset_text not in raw
    assert body["normalized_text"] not in raw or body["normalized_text"] == ""


def test_prediction_log_one_line_per_batch_item(tmp_path, monkeypatch, presets):
    log_path = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_path))
    items = [{"text": preset["text"]} for preset in presets]
    with TestClient(create_app()) as fresh:
        assert fresh.post("/predict/batch", json={"items": items}).status_code == 200
    lines = [line for line in log_path.read_text(encoding="utf-8").split("\n") if line]
    assert len(lines) == len(presets)
    expected = [len(item["text"].strip()) for item in items]
    assert [json.loads(line)["n_chars"] for line in lines] == expected


def test_prediction_log_is_intact_under_concurrent_requests(tmp_path, monkeypatch, presets):
    """Eight threads, two hundred predictions, one log line each and every line parses."""
    log_path = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_path))
    texts = [preset["text"] for preset in presets]
    n_requests, n_threads = 200, 8

    with TestClient(create_app()) as fresh:

        def post(i: int) -> int:
            return fresh.post("/predict", json={"text": texts[i % len(texts)]}).status_code

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            statuses = list(pool.map(post, range(n_requests)))
    assert statuses == [200] * n_requests
    raw = log_path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    lines = raw.split("\n")[:-1]
    assert len(lines) == n_requests
    records = [json.loads(line) for line in lines]
    assert all(set(record) == LOG_KEYS for record in records)
    assert {record["n_chars"] for record in records} == {len(t.strip()) for t in texts}


def test_prediction_log_failure_never_fails_the_request(tmp_path, monkeypatch, preset_text):
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path))  # a directory cannot be appended to
    with TestClient(create_app()) as fresh:
        response = fresh.post("/predict", json={"text": preset_text})
    assert response.status_code == 200
    assert 0.0 <= response.json()["probability_happiness"] <= 1.0


def test_metrics_exposes_the_prediction_counter(client, preset_text):
    assert client.post("/predict", json={"text": preset_text}).status_code == 200
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "tweet_emotion_predictions_total" in text
    assert "http_request" in text
    pattern = r'tweet_emotion_predictions_total\{label="(happiness|sadness)"\} (\d+\.\d+)'
    counts = re.findall(pattern, text)
    assert counts
    assert {label for label, _ in counts} <= set(settings.LABELS)
    assert sum(float(value) for _, value in counts) >= 1


def test_metrics_counter_grows_with_predictions(client, preset_text):
    def total() -> float:
        text = client.get("/metrics").text
        found = re.findall(r'tweet_emotion_predictions_total\{label="[a-z]+"\} (\d+\.\d+)', text)
        return sum(float(v) for v in found)

    before = total()
    assert client.post("/predict", json={"text": preset_text}).status_code == 200
    assert client.post("/predict", json={"text": preset_text}).status_code == 200
    assert total() == before + 2


# ---------------------------------------------------------------------------
# Pages and docs
# ---------------------------------------------------------------------------


def test_root_serves_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    index = STATIC_DIR / "index.html"
    if index.is_file():
        assert response.text == index.read_text(encoding="utf-8")
    else:
        assert "/docs" in response.text
        assert settings.DISCLAIMER in response.text


def test_static_assets_when_bundled(client):
    if not (STATIC_DIR / "style.css").is_file():
        pytest.skip("the demo page assets are not in this checkout")
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/missing.css").status_code == 404


def test_docs_and_openapi(client, max_chars):
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    assert spec["info"]["title"] == TITLE
    assert spec["info"]["version"] == __version__
    assert settings.DISCLAIMER in spec["info"]["description"]
    assert {"/health", "/version", "/presets", "/terms", "/predict", "/predict/batch"} <= set(
        spec["paths"]
    )
    request = spec["components"]["schemas"]["PredictRequest"]
    assert request["properties"]["text"]["maxLength"] == max_chars
    assert request["properties"]["text"]["minLength"] == 1


# ---------------------------------------------------------------------------
# ModelBundle
# ---------------------------------------------------------------------------


def test_bundle_shape(bundle, version_info, metrics):
    assert isinstance(bundle, model_loader.ModelBundle)
    assert isinstance(bundle.pipeline, Pipeline)
    assert bundle.version == version_info
    assert bundle.metrics == metrics
    assert bundle.threshold == metrics["threshold"]
    assert bundle.model_name == version_info["model"]
    assert bundle.model_version == version_info["model_version"]
    assert bundle.presets is None or isinstance(bundle.presets, dict)
    assert bundle.top_terms is None or isinstance(bundle.top_terms, dict)
    assert bundle.loadtest is None or isinstance(bundle.loadtest, dict)


def test_bundle_predict_texts(bundle, presets):
    texts = [preset["text"] for preset in presets]
    p = bundle.predict_texts(texts)
    assert isinstance(p, np.ndarray)
    assert p.shape == (len(texts),)
    assert np.isfinite(p).all()
    assert ((p >= 0.0) & (p <= 1.0)).all()
    for value, preset in zip(p, presets, strict=True):
        assert abs(float(value) - preset["probability_happiness"]) < 1e-4


def test_bundle_normalized_runs_only_the_first_step(bundle, presets):
    texts = [preset["text"] for preset in presets]
    out = bundle.normalized(texts)
    assert out == bundle.pipeline.named_steps["normalize"].transform(texts)
    assert all(isinstance(item, str) for item in out)


def test_bundle_explain(bundle, preset_text):
    out = bundle.explain(preset_text)
    if not is_linear(bundle):
        assert out is None
        return
    assert isinstance(out, list)
    assert 0 < len(out) <= 8
    for item in out:
        assert set(item) == {"term", "contribution"}
        assert item["contribution"] == round(item["contribution"], 4)
    magnitudes = [abs(item["contribution"]) for item in out]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_bundle_explain_of_noise_is_empty(bundle):
    out = bundle.explain("@bob 123 !!!")
    if is_linear(bundle):
        assert out == []
    else:
        assert out is None


def test_bundle_explain_is_none_for_a_model_without_coefficients(helpers, corpus_split):
    train, _ = corpus_split
    pipeline = helpers["fit_toy_pipeline"](train, "nb")
    toy = model_loader.ModelBundle(
        pipeline=pipeline, version={"model": "nb"}, metrics={"threshold": 0.5}
    )
    assert not toy.is_linear
    assert toy.explain("happy day") is None
    assert toy.explain("@bob 123 !!!") is None
    assert toy.score_texts(["happy day", "@bob"]).terms == [None, None]


# ---------------------------------------------------------------------------
# WordNet at serving time
# ---------------------------------------------------------------------------


def _wordnet_missing(package: str):
    raise LookupError(f"no {package} on nltk.data.path")


def test_warm_up_refuses_to_serve_without_wordnet(bundle, monkeypatch):
    if not bundle.lemmatizes:
        pytest.skip("the shipped normaliser does not lemmatise")
    calls: list[str] = []

    def missing(package: str):
        calls.append(package)
        return _wordnet_missing(package)

    monkeypatch.setattr(preprocess, "find_corpus", missing)
    with pytest.raises(RuntimeError) as excinfo:
        bundle.warm_up()
    assert str(excinfo.value) == model_loader.WORDNET_MISSING
    assert "python -m tweet_emotion.setup_nltk" in str(excinfo.value)
    assert "NLTK_DATA" in str(excinfo.value)
    assert calls == ["wordnet"]
    assert isinstance(excinfo.value.__cause__, LookupError)


def test_warm_up_checks_wordnet_before_it_scores(bundle, monkeypatch):
    if not bundle.lemmatizes:
        pytest.skip("the shipped normaliser does not lemmatise")
    order: list[str] = []
    original = bundle.score_texts

    def found(package: str) -> str:
        order.append(f"find:{package}")
        return "present"

    def scored(texts):
        order.append("score")
        return original(texts)

    monkeypatch.setattr(preprocess, "find_corpus", found)
    monkeypatch.setattr(bundle, "score_texts", scored)
    bundle.warm_up()
    assert order[:2] == ["find:wordnet", "score"]


def test_warm_up_never_downloads_when_the_corpus_is_missing(bundle, monkeypatch):
    if not bundle.lemmatizes:
        pytest.skip("the shipped normaliser does not lemmatise")
    monkeypatch.setattr(preprocess, "find_corpus", _wordnet_missing)

    def no_download(*args, **kwargs):
        raise AssertionError("serving must never download a corpus")

    monkeypatch.setattr(preprocess.nltk, "download", no_download)
    with pytest.raises(RuntimeError):
        bundle.warm_up()


def test_warm_up_skips_the_wordnet_check_without_lemmatising(toy_pipeline, monkeypatch):
    def never(package: str):
        raise AssertionError("find_corpus must not run for a normaliser that does not lemmatise")

    monkeypatch.setattr(preprocess, "find_corpus", never)
    toy = model_loader.ModelBundle(
        pipeline=toy_pipeline, version={"model": "logreg"}, metrics={"threshold": 0.5}
    )
    assert toy.lemmatizes is False
    toy.warm_up()


def test_app_start_up_fails_fast_without_wordnet(monkeypatch, params, prediction_log_path):
    if not params["preprocess"]["lemmatize"]:
        pytest.skip("the shipped normaliser does not lemmatise")
    monkeypatch.setattr(preprocess, "find_corpus", _wordnet_missing)
    with (
        pytest.raises(RuntimeError, match=r"python -m tweet_emotion\.setup_nltk"),
        TestClient(create_app()),
    ):
        pass


def test_load_bundle_requires_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path)
    with pytest.raises((FileNotFoundError, OSError)):
        model_loader.load_bundle()


# ---------------------------------------------------------------------------
# Entry point and logging
# ---------------------------------------------------------------------------


def test_main_starts_uvicorn_on_the_default_port(monkeypatch):
    calls: dict = {}
    if hasattr(api_main, "configure_logging"):
        monkeypatch.setattr(api_main, "configure_logging", lambda *a, **k: None)

    def fake_run(*args, **kwargs) -> None:
        calls.update(kwargs, app=args[0])

    monkeypatch.setattr(api_main.uvicorn, "run", fake_run)
    monkeypatch.setenv("PORT", "")
    assert api_main.main() == 0
    assert calls["app"] == "tweet_emotion.api.app:app"
    assert calls["host"] == "0.0.0.0"
    assert calls["port"] == 8000
    assert calls["workers"] == 1
    assert calls["log_config"] is None


def test_main_honours_port(monkeypatch):
    calls: dict = {}
    if hasattr(api_main, "configure_logging"):
        monkeypatch.setattr(api_main, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(api_main.uvicorn, "run", lambda *args, **kwargs: calls.update(kwargs))
    monkeypatch.setenv("PORT", "7860")
    assert api_main.main() == 0
    assert calls["port"] == 7860


def test_main_exits_with_one_line_on_an_impossible_port(monkeypatch, capsys):
    calls: dict = {}
    if hasattr(api_main, "configure_logging"):
        monkeypatch.setattr(api_main, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(api_main.uvicorn, "run", lambda *args, **kwargs: calls.update(kwargs))
    monkeypatch.setenv("PORT", "70000")
    assert api_main.main() == 2
    assert calls == {}
    err = capsys.readouterr().err.strip()
    assert err == "PORT must be an integer between 1 and 65535, got 70000"


def test_configure_logging_honours_log_level(monkeypatch, tmp_path):
    configure_logging = getattr(app_module, "configure_logging", None)
    config_path = getattr(app_module, "LOGGING_CONFIG_PATH", settings.CONFIGS_DIR / "logging.yaml")
    if configure_logging is None:
        pytest.skip("the app module exposes no configure_logging")
    if not Path(config_path).is_file():
        pytest.skip("configs/logging.yaml is not in this checkout")
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        monkeypatch.setenv("LOG_LEVEL", "debug")
        configure_logging()
        assert root.level == logging.DEBUG

        root.handlers[:] = []
        monkeypatch.setenv("LOG_LEVEL", "warning")
        configure_logging(tmp_path / "missing.yaml")
        assert root.level == logging.WARNING
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)

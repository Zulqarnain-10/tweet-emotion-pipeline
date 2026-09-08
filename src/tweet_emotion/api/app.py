"""FastAPI application: demo page, prediction endpoints, receipts, and Prometheus metrics.

The model and its reports load once in the lifespan handler. Every prediction is appended
to a JSONL log that carries the label, the probability, the text length, and the model
version, never the text itself; a failure to write that log is a warning, never an error
for the caller.
"""

from __future__ import annotations

import json
import logging
import logging.config
import math
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from tweet_emotion import __version__, settings
from tweet_emotion.api import model_loader
from tweet_emotion.api.schemas import (
    BatchRequest,
    BatchResponse,
    Health,
    Prediction,
    PredictRequest,
    TermContribution,
    VersionInfo,
)

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_PATH = STATIC_DIR / "index.html"
LOGGING_CONFIG_PATH = settings.CONFIGS_DIR / "logging.yaml"

# Sized so the advertised batch limit is reachable: BATCH_MAX_ITEMS texts of MAX_CHARS
# characters, every character at its JSON-escaped worst case of 12 bytes (an astral
# character written as a \uXXXX surrogate pair), plus the framing around each item.
MAX_BODY_BYTES = 1_250_000

TITLE = "Tweet emotion pipeline"
DESCRIPTION = (
    "Scores short English text as happiness or sadness with a probability, using the model "
    "that the DVC pipeline in this repository trained, evaluated, and versioned. Request "
    f"bodies above {MAX_BODY_BYTES:,} bytes ({MAX_BODY_BYTES / 1_000_000:g} MB) are refused "
    "with status 413. " + settings.DISCLAIMER
)

PLACEHOLDER_HTML = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Tweet emotion pipeline</title></head>
<body>
<main>
<h1>Tweet emotion pipeline</h1>
<p>The demo page is not bundled with this build. The API is up: see <a href="/docs">/docs</a>,
<a href="/version">/version</a>, and <a href="/health">/health</a>.</p>
<p>{disclaimer}</p>
</main>
</body>
</html>
"""

PREDICTIONS_TOTAL = Counter(
    "tweet_emotion_predictions_total",
    "Predictions served, by label at the decision threshold.",
    ["label"],
)
for _label in settings.LABELS:
    PREDICTIONS_TOTAL.labels(label=_label)

# Endpoints run in the worker thread pool, so concurrent requests append to the prediction
# log at the same time. One lock per process serialises the mkdir, open, and write; on
# Windows an append is emulated as a seek to the end followed by a write, which is not
# atomic across handles.
_PREDICTION_LOG_LOCK = threading.Lock()


def configure_logging(config_path: Path | None = None) -> None:
    """Structured JSON logs from configs/logging.yaml when present, else basicConfig.

    The root level comes from LOG_LEVEL (default INFO) in both cases.
    """
    config_path = config_path or LOGGING_CONFIG_PATH
    level = settings.log_level()
    if config_path.is_file():
        try:
            with open(config_path, encoding="utf-8") as fh:
                logging.config.dictConfig(yaml.safe_load(fh))
            logging.getLogger().setLevel(level)
            return
        except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError) as exc:
            logging.basicConfig(level=level)
            log.warning("could not apply %s, using basicConfig: %s", config_path, exc)
            return
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


class BodySizeLimit:
    """ASGI middleware that refuses request bodies above max_bytes with status 413.

    A declared Content-Length above the cap is refused before the app runs. A chunked body
    is counted as it streams and refused once it passes the cap, before the JSON parser
    sees the whole payload.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.detail = f"request body exceeds {max_bytes} bytes"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length", "")
        if declared.isdigit() and int(declared) > self.max_bytes:
            response = JSONResponse({"detail": self.detail}, status_code=413)
            await response(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # FastAPI re-raises HTTPException from body parsing, so this becomes a 413.
                    raise HTTPException(status_code=413, detail=self.detail)
            return message

        await self.app(scope, limited_receive, send)


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats, which JSON cannot carry, with their text form."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


async def validation_error_422(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's default 422 body, made renderable when the request carried NaN or Infinity.

    Python's json module emits and accepts those tokens; the default handler echoes the
    offending input and the JSON encoder then refuses it, turning a 422 into a 500.
    """
    return JSONResponse(
        status_code=422, content={"detail": _json_safe(jsonable_encoder(exc.errors()))}
    )


def read_index_html() -> str:
    """The demo page when it ships with the package, else a small placeholder."""
    if INDEX_PATH.is_file():
        return INDEX_PATH.read_text(encoding="utf-8")
    return PLACEHOLDER_HTML.format(disclaimer=settings.DISCLAIMER)


def append_prediction_log(path: Path, records: list[dict]) -> None:
    """Append one JSON line per record under the process-wide lock.

    Never raises: a failed write is a warning.
    """
    try:
        with _PREDICTION_LOG_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8", newline="\n") as fh:
                for record in records:
                    fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception as exc:
        log.warning("prediction log write failed at %s: %s", path, exc)


def get_bundle(request: Request) -> model_loader.ModelBundle:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return bundle


def score(
    bundle: model_loader.ModelBundle, items: list[PredictRequest], log_path: Path
) -> list[Prediction]:
    """Score texts in one pipeline pass, count them, and log them.

    The label compares the four-decimal probability the response carries against the
    threshold, so a displayed probability equal to the threshold reads happiness.
    """
    texts = [item.text for item in items]
    scored = bundle.score_texts(texts)
    ts = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    predictions: list[Prediction] = []
    records: list[dict] = []
    for text, probability, normalized_text, terms in zip(
        texts, scored.probability_happiness, scored.normalized, scored.terms, strict=True
    ):
        p = float(probability)
        shown = round(p, 4)
        if shown >= bundle.threshold:
            label = settings.POSITIVE_LABEL
            label_probability = shown
        else:
            label = settings.NEGATIVE_LABEL
            label_probability = round(1.0 - shown, 4)
        PREDICTIONS_TOTAL.labels(label=label).inc()
        predictions.append(
            Prediction(
                label=label,
                probability_happiness=shown,
                probability=label_probability,
                threshold=bundle.threshold,
                normalized_text=normalized_text,
                terms=None if terms is None else [TermContribution(**term) for term in terms],
                model=bundle.model_name,
                model_version=bundle.model_version,
                disclaimer=settings.DISCLAIMER,
            )
        )
        records.append(
            {
                "ts": ts,
                "label": label,
                "probability_happiness": round(p, 6),
                "n_chars": len(text),
                "model_version": bundle.model_version,
            }
        )
    append_prediction_log(log_path, records)
    return predictions


def create_app(bundle: model_loader.ModelBundle | None = None) -> FastAPI:
    """Build the application.

    With no argument the lifespan loads the artifacts from disk. Passing a bundle serves that
    bundle instead, which lets tests run the whole app around an in-memory pipeline.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Load artifacts once, warm the model, resolve the prediction log path once."""
        if not logging.getLogger().handlers:
            configure_logging()
        app.state.bundle = bundle if bundle is not None else model_loader.load_bundle()
        app.state.bundle.warm_up()
        app.state.prediction_log_path = settings.prediction_log_path()
        log.info(
            "api ready",
            extra={
                "model_version": app.state.bundle.model_version,
                "prediction_log": str(app.state.prediction_log_path),
            },
        )
        yield

    app = FastAPI(
        title=TITLE,
        version=__version__,
        description=DESCRIPTION,
        license_info={"name": "MIT"},
        contact={"name": "Syed Zulqarnain Hassan", "url": "https://zulqarnainhassan.com"},
        lifespan=lifespan,
    )
    app.add_exception_handler(RequestValidationError, validation_error_422)
    app.add_middleware(BodySizeLimit, max_bytes=MAX_BODY_BYTES)

    # Every GET endpoint also answers HEAD, which uptime probes and link checkers send.
    # FastAPI does not add HEAD to a GET route the way Starlette does, and one route
    # declared with both methods lists HEAD in /docs under a duplicate operation id, so
    # HEAD is a second route on the same function, kept out of the schema. Starlette
    # drops the body and keeps the headers.
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.head("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse(read_index_html())

    @app.get("/health", response_model=Health, tags=["service"])
    @app.head("/health", response_model=Health, include_in_schema=False)
    def health(request: Request) -> Health:
        bundle = get_bundle(request)
        return Health(status="ok", model_loaded=True, model_version=bundle.model_version)

    # Documented through `responses` rather than response_model so the payload passes through
    # exactly as built: loadtest is present only when reports/loadtest.json exists, and an
    # unexpected type in version.json can never turn this receipts endpoint into a 500.
    @app.get("/version", tags=["service"], responses={200: {"model": VersionInfo}})
    @app.head("/version", include_in_schema=False)
    def version(request: Request) -> dict[str, Any]:
        """models/version.json plus headline metrics, request limits, and the disclaimer."""
        return model_loader.version_payload(get_bundle(request))

    @app.get("/presets", tags=["receipts"])
    @app.head("/presets", include_in_schema=False)
    def presets(request: Request) -> dict[str, Any]:
        """Real held-out tweets chosen by the model's own scores, from configs/presets.json."""
        bundle = get_bundle(request)
        if bundle.presets is None:
            raise HTTPException(status_code=404, detail="presets.json is not in this build")
        return bundle.presets

    @app.get("/terms", tags=["receipts"])
    @app.head("/terms", include_in_schema=False)
    def terms(request: Request) -> dict[str, Any]:
        """The terms that move the score most, from reports/top_terms.json."""
        bundle = get_bundle(request)
        if bundle.top_terms is None:
            raise HTTPException(status_code=404, detail="top_terms.json is not in this build")
        return bundle.top_terms

    @app.post("/predict", response_model=Prediction, tags=["predict"])
    def predict(item: PredictRequest, request: Request) -> Prediction:
        """Probability of happiness for one text and the label at the decision threshold."""
        bundle = get_bundle(request)
        return score(bundle, [item], request.app.state.prediction_log_path)[0]

    @app.post("/predict/batch", response_model=BatchResponse, tags=["predict"])
    def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
        """Score several texts in one call; predictions keep the request order."""
        bundle = get_bundle(request)
        predictions = score(bundle, batch.items, request.app.state.prediction_log_path)
        return BatchResponse(count=len(predictions), predictions=predictions)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # The handler label for the mounted StaticFiles app is the mount path "/static", so the
    # pattern anchors on the prefix rather than requiring a trailing slash.
    Instrumentator(
        should_group_status_codes=False,
        excluded_handlers=["^/metrics$", "^/static"],
    ).instrument(app).expose(app, tags=["service"])
    return app


app = create_app()

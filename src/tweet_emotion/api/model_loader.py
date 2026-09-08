"""Load the model and its receipts once, and score raw tweet text.

The loader reads under settings.MODELS_DIR, settings.REPORTS_DIR, and settings.CONFIGS_DIR.
The model, version.json, and metrics.json are required. presets.json, top_terms.json, and
loadtest.json are optional so a slim serving image still starts. A bundle can also be built
directly around an in-memory pipeline, which is how the tests exercise the API without artifacts.

The serving object is the sklearn Pipeline the train stage saved: a normalise step, a vectorise
step, and a classify step. Scoring runs those three steps once per request and reads the
normalised text and the per-term contributions from the same pass, so every field in a response
comes from one traversal of the same fitted objects.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

from tweet_emotion import preprocess, settings
from tweet_emotion.api.schemas import BATCH_MAX_ITEMS, MAX_CHARS

log = logging.getLogger(__name__)

VERSION_METRIC_KEYS: tuple[str, ...] = (
    "accuracy",
    "f1",
    "roc_auc",
    "pr_auc",
    "n_test",
    "trained_at",
    "threshold",
)
LOADTEST_KEYS: tuple[str, ...] = (
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "mean_ms",
    "rps",
    "error_rate",
    "requests",
    "concurrency",
    "host",
    "timestamp",
    "url",
)

# Step names the train stage uses. Position is the fallback when a pipeline names them otherwise.
STEP_NORMALIZE = "normalize"
STEP_VECTORIZE = "vectorize"
STEP_CLASSIFY = "classify"

TOP_TERMS_PER_TEXT = 8
WARMUP_TEXT = "warming up the model before the first request"
PARITY_TOLERANCE = 1e-9
WORDNET_MISSING = (
    "WordNet corpus not found on nltk.data.path; run python -m tweet_emotion.setup_nltk "
    "(NLTK_DATA is honoured)"
)


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_json_optional(path: Path) -> dict | None:
    """The file's contents, or None with a warning when it is absent or unreadable."""
    if not path.is_file():
        log.warning("optional artifact missing: %s", path)
        return None
    try:
        return read_json(path)
    except (OSError, ValueError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return None


def loadtest_block(data: dict | None) -> dict | None:
    """The latency receipt trimmed to the keys /version publishes, or None."""
    if not data:
        return None
    return {key: data[key] for key in LOADTEST_KEYS if key in data}


@dataclass
class Scored:
    """One pass of the pipeline over a batch of texts."""

    probability_happiness: np.ndarray
    normalized: list[str]
    terms: list[list[dict] | None]


@dataclass
class ModelBundle:
    """The fitted pipeline plus everything the endpoints answer from.

    threshold defaults to metrics["threshold"] when not given.
    """

    pipeline: object
    version: dict
    metrics: dict
    threshold: float | None = None
    presets: dict | None = None
    top_terms: dict | None = None
    loadtest: dict | None = None

    def __post_init__(self) -> None:
        if self.threshold is None:
            if "threshold" not in self.metrics:
                raise ValueError("threshold is required: pass threshold= or put it in metrics")
            self.threshold = float(self.metrics["threshold"])
        self.threshold = float(self.threshold)
        if not hasattr(self.pipeline, "steps") or len(self.pipeline.steps) < 2:
            raise ValueError("pipeline must be an sklearn Pipeline with at least two steps")

    @classmethod
    def from_paths(
        cls,
        models_dir: Path | None = None,
        reports_dir: Path | None = None,
        configs_dir: Path | None = None,
    ) -> ModelBundle:
        """Read the artifacts from disk. Call once per process."""
        models_dir = models_dir or settings.MODELS_DIR
        reports_dir = reports_dir or settings.REPORTS_DIR
        configs_dir = configs_dir or settings.CONFIGS_DIR

        model_path = models_dir / settings.MODEL_PATH.name
        if not model_path.is_file():
            raise FileNotFoundError(f"model artifact not found: {model_path}")
        pipeline = joblib.load(model_path)
        version = read_json(models_dir / settings.VERSION_PATH.name)
        metrics = read_json(reports_dir / settings.METRICS_PATH.name)
        bundle = cls(
            pipeline=pipeline,
            version=version,
            metrics=metrics,
            threshold=float(metrics["threshold"]),
            presets=read_json_optional(configs_dir / settings.PRESETS_PATH.name),
            top_terms=read_json_optional(reports_dir / settings.TOP_TERMS_PATH.name),
            loadtest=read_json_optional(reports_dir / settings.LOADTEST_PATH.name),
        )
        log.info(
            "model artifacts loaded",
            extra={
                "model": bundle.model_name,
                "model_version": bundle.model_version,
                "threshold": bundle.threshold,
                "models_dir": str(models_dir),
            },
        )
        return bundle

    # ------------------------------------------------------------------ identity

    @property
    def model_name(self) -> str:
        return str(self.version.get("model", "unknown"))

    @property
    def model_version(self) -> str:
        return str(self.version.get("model_version", "unknown"))

    # ------------------------------------------------------------------ steps

    def _step(self, name: str, position: int) -> object:
        named = getattr(self.pipeline, "named_steps", {})
        if name in named:
            return named[name]
        return self.pipeline.steps[position][1]

    @property
    def normalizer(self) -> object:
        """The normalise step: the first step of the pipeline."""
        return self._step(STEP_NORMALIZE, 0)

    @property
    def vectorizer(self) -> object:
        """The vectorise step: the step that owns the vocabulary."""
        return self._step(STEP_VECTORIZE, 1)

    @property
    def classifier(self) -> object:
        """The classify step: the last step of the pipeline."""
        return self._step(STEP_CLASSIFY, -1)

    @property
    def is_linear(self) -> bool:
        """True when the classifier exposes per-feature coefficients."""
        return hasattr(self.classifier, "coef_")

    @cached_property
    def feature_names(self) -> np.ndarray | None:
        """Vocabulary terms in column order, or None when the vectoriser cannot name them."""
        getter = getattr(self.vectorizer, "get_feature_names_out", None)
        if getter is None:
            return None
        return np.asarray(getter(), dtype=object)

    # ------------------------------------------------------------------ scoring

    def normalized(self, texts: Iterable[str]) -> list[str]:
        """Run only the normalise step, so callers can show what the model saw."""
        return [str(item) for item in self.normalizer.transform([str(t) for t in texts])]

    def vectorize(self, normalized_texts: list[str]) -> sparse.csr_matrix:
        """Run every step between the normaliser and the classifier."""
        X = normalized_texts
        for _, step in self.pipeline.steps[1:-1]:
            X = step.transform(X)
        return sparse.csr_matrix(X)

    def predict_texts(self, texts: Iterable[str]) -> np.ndarray:
        """p(happiness) for raw texts through the full pipeline, one number per text."""
        return np.asarray(self.pipeline.predict_proba([str(t) for t in texts])[:, 1], dtype=float)

    def explain_matrix(self, X: sparse.csr_matrix) -> list[list[dict]] | None:
        """Per-row term contributions for a linear classifier, else None.

        For each row, the columns present in the vectorised text contribute coef * value.
        The top TOP_TERMS_PER_TEXT by absolute contribution are returned, largest first, as
        [{"term", "contribution"}] with four-decimal contributions. A row that hits no
        vocabulary term yields an empty list.
        """
        if not self.is_linear:
            return None
        names = self.feature_names
        if names is None:
            return None
        coef = np.asarray(self.classifier.coef_, dtype=float)
        if coef.ndim == 2:
            coef = coef[0]
        if coef.shape[0] != X.shape[1] or names.shape[0] != X.shape[1]:
            log.warning(
                "coefficient, vocabulary, and feature widths disagree (%d, %d, %d)",
                coef.shape[0],
                names.shape[0],
                X.shape[1],
            )
            return None
        out: list[list[dict]] = []
        for row in range(X.shape[0]):
            start, end = X.indptr[row], X.indptr[row + 1]
            idx = X.indices[start:end]
            contributions = coef[idx] * X.data[start:end]
            order = np.argsort(-np.abs(contributions), kind="stable")[:TOP_TERMS_PER_TEXT]
            out.append(
                [
                    {"term": str(names[idx[j]]), "contribution": round(float(contributions[j]), 4)}
                    for j in order
                ]
            )
        return out

    def explain(self, text: str) -> list[dict] | None:
        """Term contributions for one raw text.

        An empty list when the classifier is linear but the normalised text hits no
        vocabulary term; None when the classifier has no per-feature coefficients.
        """
        explained = self.explain_matrix(self.vectorize(self.normalized([text])))
        return None if explained is None else explained[0]

    def score_texts(self, texts: Iterable[str]) -> Scored:
        """Normalise once, vectorise once, then read probabilities and contributions."""
        raw = [str(t) for t in texts]
        normalized = self.normalized(raw)
        X = self.vectorize(normalized)
        probabilities = np.asarray(self.classifier.predict_proba(X)[:, 1], dtype=float)
        explained = self.explain_matrix(X)
        terms: list[list[dict] | None] = [None] * len(raw) if explained is None else list(explained)
        return Scored(probability_happiness=probabilities, normalized=normalized, terms=terms)

    @property
    def lemmatizes(self) -> bool:
        """True when the normalise step lemmatises, which needs the WordNet corpus."""
        return bool(getattr(self.normalizer, "lemmatize", False))

    def require_wordnet(self) -> None:
        """Refuse to serve when the normaliser lemmatises and WordNet is not installed.

        The lemmatiser would otherwise download the corpus on its first use, so a serving
        process that skipped setup_nltk would block start-up on the network. Checking
        nltk.data.path first fails fast with the fix in the message and never downloads.
        """
        if not self.lemmatizes:
            return
        try:
            preprocess.find_corpus("wordnet")
        except LookupError as exc:
            raise RuntimeError(WORDNET_MISSING) from exc

    def warm_up(self) -> None:
        """Run one text through both scoring paths and confirm they agree.

        The WordNet check runs first so a missing corpus is a clear error rather than a
        download. The warm-up then pays any remaining lazy start-up cost (building the
        lemmatiser) before the first request and proves the step-by-step path equals
        pipeline.predict_proba.
        """
        self.require_wordnet()
        scored = self.score_texts([WARMUP_TEXT])
        direct = self.predict_texts([WARMUP_TEXT])
        gap = float(abs(scored.probability_happiness[0] - direct[0]))
        if gap > PARITY_TOLERANCE:
            raise RuntimeError(
                f"step-by-step scoring disagrees with the pipeline by {gap:.3g}; refusing to serve"
            )
        log.info(
            "model warm",
            extra={
                "model_version": self.model_version,
                "linear": self.is_linear,
                "n_features": None if self.feature_names is None else len(self.feature_names),
            },
        )


def load_bundle(
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
    configs_dir: Path | None = None,
) -> ModelBundle:
    """Read the artifacts from disk. Call once per process."""
    return ModelBundle.from_paths(models_dir, reports_dir, configs_dir)


def version_payload(bundle: ModelBundle) -> dict:
    """models/version.json plus headline metrics, the load test when present, the request
    limits, and the disclaimer."""
    payload = dict(bundle.version)
    payload["model"] = bundle.model_name
    payload["model_version"] = bundle.model_version
    metrics = {key: bundle.metrics.get(key) for key in VERSION_METRIC_KEYS}
    metrics["threshold"] = bundle.threshold
    payload["metrics"] = metrics
    loadtest = loadtest_block(bundle.loadtest)
    if loadtest is not None:
        payload["loadtest"] = loadtest
    payload["params"] = {"max_chars": MAX_CHARS, "batch_max": BATCH_MAX_ITEMS}
    payload["disclaimer"] = settings.DISCLAIMER
    return payload

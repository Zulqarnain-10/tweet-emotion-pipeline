"""Request and response models for the tweet emotion API.

The text limits come from params.yaml (api.max_chars and api.batch_max), read once at import
through settings.load_params(), so the validator, /version, and the demo page agree on the same
numbers. Unknown fields are rejected so a typo never silently drops an input. Types are strict:
a number or a boolean where a string is expected is refused rather than coerced.

Nothing here imports the preprocessing code. The normaliser travels inside the pickled pipeline
and is only touched by the model loader.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tweet_emotion import settings

DEFAULT_MAX_CHARS = 1000
DEFAULT_BATCH_MAX = 100


def _api_limits() -> tuple[int, int]:
    """(max_chars, batch_max) from params.yaml, with defaults when the api block is absent."""
    params = settings.load_params().get("api") or {}
    return (
        int(params.get("max_chars", DEFAULT_MAX_CHARS)),
        int(params.get("batch_max", DEFAULT_BATCH_MAX)),
    )


MAX_CHARS, BATCH_MAX_ITEMS = _api_limits()

Label = Literal["happiness", "sadness"]

BLANK_TEXT_MESSAGE = "text must contain at least one non-whitespace character"


class PredictRequest(BaseModel):
    """One tweet to score."""

    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_CHARS,
        description=(
            f"Tweet text, 1 to {MAX_CHARS} characters after surrounding whitespace is trimmed. "
            "Whitespace-only text is refused."
        ),
        examples=["so happy to see the sun again"],
    )

    @field_validator("text", mode="before")
    @classmethod
    def _trim_and_reject_blank(cls, value: Any) -> Any:
        """Trim surrounding whitespace; refuse text that is empty once trimmed.

        Non-string input passes through untouched so the strict type check reports it.
        """
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                raise ValueError(BLANK_TEXT_MESSAGE)
            return trimmed
        return value


class TermContribution(BaseModel):
    """One vocabulary term found in the normalised text and its pull on the score."""

    term: str = Field(description="Vocabulary term (a unigram or bigram) present in the text")
    contribution: float = Field(
        description=(
            "Coefficient times feature value, four decimals. Positive pushes towards happiness, "
            "negative towards sadness"
        )
    )


class Prediction(BaseModel):
    """Score and label for one tweet.

    terms has three shapes and the demo page renders each one differently: a list of
    contributions, an empty list when the model is linear but the normalised text hits no
    vocabulary term, and null when the shipped model has no per-term coefficients.
    """

    model_config = ConfigDict(protected_namespaces=())

    label: Label = Field(
        description=(
            "happiness when probability_happiness >= threshold, else sadness. The comparison "
            "uses the four-decimal probability in this response, so the fields never disagree"
        )
    )
    probability_happiness: float = Field(
        description="Probability the tweet expresses happiness, rounded to four decimals"
    )
    probability: float = Field(
        description="Probability of the returned label, rounded to four decimals"
    )
    threshold: float = Field(description="Decision threshold from params.yaml (evaluate.threshold)")
    normalized_text: str = Field(
        description="The text after the pipeline's normalise step, which is what the model saw"
    )
    terms: list[TermContribution] | None = Field(
        description=(
            "Up to eight terms with the largest absolute contribution to the score, for linear "
            "models. An empty list means the model is linear but the normalised text hit no "
            "vocabulary term, so nothing moved the score away from the intercept. null means "
            "the shipped model has no per-term coefficients"
        )
    )
    model: str = Field(description="Model family behind the score, for example logreg")
    model_version: str = Field(
        description="Package version plus the short git sha the model was trained at"
    )
    disclaimer: str = Field(description="Scope note that travels with every prediction")


class BatchRequest(BaseModel):
    """Up to BATCH_MAX_ITEMS tweets scored in one call."""

    model_config = ConfigDict(extra="forbid")

    items: list[PredictRequest] = Field(
        min_length=1,
        max_length=BATCH_MAX_ITEMS,
        description=f"Tweets to score, in order; 1 to {BATCH_MAX_ITEMS} per call",
    )

    @field_validator("items", mode="before")
    @classmethod
    def _check_batch_size(cls, value: Any) -> Any:
        if isinstance(value, list) and not 1 <= len(value) <= BATCH_MAX_ITEMS:
            raise ValueError(f"a batch holds 1 to {BATCH_MAX_ITEMS} items, got {len(value)}")
        return value


class BatchResponse(BaseModel):
    """Predictions in the same order as the request items."""

    count: int = Field(description="Number of predictions returned")
    predictions: list[Prediction]


class Health(BaseModel):
    """Liveness and readiness in one small object."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"]
    model_loaded: bool
    model_version: str


class VersionMetrics(BaseModel):
    """Headline numbers from reports/metrics.json, measured once on the held-out test split."""

    accuracy: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    pr_auc: float | None = None
    n_test: int | None = None
    trained_at: str | None = None
    threshold: float | None = None


class LoadtestSummary(BaseModel):
    """Latency receipt from reports/loadtest.json, present only when a load test has run."""

    model_config = ConfigDict(extra="allow")

    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    mean_ms: float | None = None
    rps: float | None = None
    error_rate: float | None = None
    requests: int | None = None
    concurrency: int | None = None
    host: str | None = None
    timestamp: str | None = None
    url: str | None = None


class ApiParams(BaseModel):
    """Request limits from params.yaml, so the demo page never hard-codes them."""

    max_chars: int = Field(description="Maximum characters accepted in one text")
    batch_max: int = Field(description="Maximum items accepted by /predict/batch")


class VersionInfo(BaseModel):
    """models/version.json plus the metrics subset, the load test, the limits, and the disclaimer.

    Fields written by the train stage are optional here so a build with a partial version.json
    still answers; extra keys pass through unchanged.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model: str = Field(description="Winning candidate name, for example logreg")
    model_version: str = Field(description="Package version plus the short git sha")
    package_version: str | None = None
    git_sha: str | None = None
    git_sha_short: str | None = None
    git_dirty: bool | str | None = None
    data_sha256: str | None = None
    trained_at: str | None = None
    n_train: int | None = None
    n_features: int | None = None
    selection_metric: str | None = None
    cv_folds: int | None = None
    candidates: dict[str, dict[str, Any]] | None = None
    winner_params: dict[str, Any] | None = None
    libraries: dict[str, Any] | None = None
    metrics: VersionMetrics
    loadtest: LoadtestSummary | None = None
    params: ApiParams
    disclaimer: str

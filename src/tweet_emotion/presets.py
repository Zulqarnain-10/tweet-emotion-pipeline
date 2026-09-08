"""Four real held-out tweets chosen by the shipped model's own scores, for the demo page.

Usage:
    python -m tweet_emotion.presets

Every tweet in data/raw/test.csv is scored through the full Pipeline. The presets are the tweet
the model is most sure is happiness, the one it is most sure is sadness, the one closest to the
decision threshold, and the confident miss, each chosen among the readable held-out tweets
(6 to 140 characters, no @mentions, no URLs) with a fallback to any row when none qualify.
Nothing is written by hand; each description says why the row was chosen and which of those
two pools it came from.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tweet_emotion import settings
from tweet_emotion.preprocess import MENTION_RE, URL_RE

log = logging.getLogger(__name__)

MIN_CHARS, MAX_CHARS = 6, 140
PROBABILITY_COLUMN = "probability_happiness"
CHOSEN_BY = "model scores on the held-out test split"

# Preset ids in output order, with their titles and the reason each row is chosen. The reason
# is the tail of the description; describe() prefixes the pool the row came from.
PRESET_SPECS: tuple[tuple[str, str, str], ...] = (
    ("clear_happiness", "Clear happiness", "the one the model is most sure is happiness"),
    ("clear_sadness", "Clear sadness", "the one the model is most sure is sadness"),
    ("close_call", "Close call", "the one whose score sits closest to the decision threshold"),
    ("model_miss", "Model miss", "the one the model gets wrong with the most confidence"),
)
PRESET_IDS: tuple[str, ...] = tuple(spec[0] for spec in PRESET_SPECS)
PRESET_REASONS: dict[str, str] = {spec[0]: spec[2] for spec in PRESET_SPECS}
READABLE_POOL = "Among readable held-out tweets"
FALLBACK_POOL = "Among all held-out tweets"


def describe(preset_id: str, readable: bool) -> str:
    """One plain sentence: which pool the row came from, then why it was chosen.

    The readability filter (is_readable) runs before the scores are ranked, so a preset is an
    extreme among readable rows, not across the whole split; the sentence says so. A row picked
    by the fallback, when no readable row was left, is described as chosen among all rows.
    """
    pool = READABLE_POOL if readable else FALLBACK_POOL
    return f"{pool}, {PRESET_REASONS[preset_id]}."


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def is_readable(text: str) -> bool:
    """Between 6 and 140 characters, no @mentions, no URLs."""
    text = str(text)
    if not MIN_CHARS <= len(text) <= MAX_CHARS:
        return False
    return not (MENTION_RE.search(text) or URL_RE.search(text))


def score_texts(pipeline, texts) -> np.ndarray:
    """p(happiness) for each raw text through the full Pipeline."""
    return np.asarray(pipeline.predict_proba(list(texts))[:, 1], dtype=float)


def _first_unused(candidates: pd.DataFrame, used: set[int]) -> int | None:
    """Positional index (into the scored frame) of the first row not already chosen."""
    for idx in candidates.index:
        if int(idx) not in used:
            return int(idx)
    return None


def _pick(ordered: pd.DataFrame, readable: pd.Series, used: set[int]) -> int | None:
    """Prefer readable rows in the given order; fall back to any row in that order."""
    idx = _first_unused(ordered.loc[readable.loc[ordered.index]], used)
    if idx is None:
        idx = _first_unused(ordered, used)
    return idx


def choose_presets(scored: pd.DataFrame, threshold: float) -> dict[str, int]:
    """Row index of each preset in the scored frame; rows are never reused.

    scored needs columns tweet_id, text, label, probability_happiness. Ties are broken by
    tweet_id so the choice is deterministic.
    """
    df = scored.reset_index(drop=True)
    p = df[PROBABILITY_COLUMN].astype(float)
    y = df[settings.LABEL_COLUMN].astype(int)
    readable = df[settings.TEXT_COLUMN].map(is_readable).astype(bool)
    predicted = (p >= float(threshold)).astype(int)
    confidence = np.where(predicted == 1, p, 1.0 - p)
    work = df.assign(
        _p=p,
        _gap=(p - float(threshold)).abs(),
        _confidence=confidence,
        _wrong=predicted != y,
    )
    orders = {
        "clear_happiness": work.loc[y == 1].sort_values(
            ["_p", settings.ID_COLUMN], ascending=[False, True], kind="stable"
        ),
        "clear_sadness": work.loc[y == 0].sort_values(
            ["_p", settings.ID_COLUMN], ascending=[True, True], kind="stable"
        ),
        "close_call": work.sort_values(
            ["_gap", settings.ID_COLUMN], ascending=[True, True], kind="stable"
        ),
        "model_miss": work.loc[work["_wrong"]].sort_values(
            ["_confidence", settings.ID_COLUMN], ascending=[False, True], kind="stable"
        ),
    }
    chosen: dict[str, int] = {}
    used: set[int] = set()
    for preset_id in PRESET_IDS:
        idx = _pick(orders[preset_id], readable, used)
        if idx is None:
            log.warning("no held-out row qualifies for %s; skipping it", preset_id)
            continue
        chosen[preset_id] = idx
        used.add(idx)
    return chosen


def build_presets(scored: pd.DataFrame, threshold: float, model_version: str, n: int) -> dict:
    """The configs/presets.json payload: at most n presets in PRESET_IDS order."""
    df = scored.reset_index(drop=True)
    chosen = choose_presets(df, threshold)
    presets = []
    for preset_id, title, _reason in PRESET_SPECS:
        if preset_id not in chosen:
            continue
        row = df.iloc[chosen[preset_id]]
        text = str(row[settings.TEXT_COLUMN])
        presets.append(
            {
                "id": preset_id,
                "title": title,
                "text": text,
                "label": settings.LABELS[int(row[settings.LABEL_COLUMN])],
                "description": describe(preset_id, is_readable(text)),
                PROBABILITY_COLUMN: round(float(row[PROBABILITY_COLUMN]), 4),
                "tweet_id": int(row[settings.ID_COLUMN]),
            }
        )
    return {
        "presets": presets[: max(int(n), 0)],
        "chosen_by": CHOSEN_BY,
        "model_version": model_version,
    }


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def read_raw_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, dtype={settings.TEXT_COLUMN: str}, keep_default_na=False, encoding="utf-8"
    )
    missing = [c for c in settings.CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    return df


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        params = settings.load_params()
        n = int(params["presets"]["n"])
        with open(settings.METRICS_PATH, encoding="utf-8") as fh:
            metrics = json.load(fh)
        pipeline = joblib.load(settings.MODEL_PATH)
        test_df = read_raw_test(settings.RAW_TEST_PATH)
        scored = test_df.assign(
            **{PROBABILITY_COLUMN: score_texts(pipeline, test_df[settings.TEXT_COLUMN])}
        )
        log.info("scored %d held-out tweets", len(scored))
        out = build_presets(scored, float(metrics["threshold"]), metrics["model_version"], n)

        settings.CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = settings.PRESETS_PATH.with_name(settings.PRESETS_PATH.name + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(out, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, settings.PRESETS_PATH)
        for preset in out["presets"]:
            log.info(
                "%s: tweet_id=%d p=%.4f label=%s",
                preset["id"],
                preset["tweet_id"],
                preset[PROBABILITY_COLUMN],
                preset["label"],
            )
        log.info("wrote %s (%d presets)", settings.PRESETS_PATH, len(out["presets"]))
    except Exception as exc:
        log.error("presets failed: %s: %s", type(exc).__name__, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

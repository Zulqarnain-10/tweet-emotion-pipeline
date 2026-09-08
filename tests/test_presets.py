"""Unit tests for tweet_emotion.presets: the four held-out tweets chosen by the model's own
scores, the readability preference, and the file the stage writes.

The stage runs on a tiny pipeline fitted in the tests and a held-out frame whose every text
qualifies for the readability rule, so the expected picks follow from the scores alone.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from tweet_emotion import presets, settings

IDS = ["clear_happiness", "clear_sadness", "close_call", "model_miss"]
PRESET_KEYS = {"id", "title", "text", "label", "description", "probability_happiness", "tweet_id"}
CHOSEN_BY = "model scores on the held-out test split"
MODEL_VERSION = "0.1.0+abc1234"
URL = re.compile(r"https?://\S+|www\.\S+")
EM_DASH = "—"


def qualifies(text: str) -> bool:
    return 6 <= len(text) <= 140 and "@" not in text and not URL.search(text)


def expected_picks(test_df: pd.DataFrame, p: np.ndarray, threshold: float = 0.5) -> dict:
    """The contract's four choices, restricted to readable tweets when any qualify."""
    y = test_df["label"].to_numpy().astype(int)
    ids = test_df["tweet_id"].to_numpy()
    ok = np.array([qualifies(t) for t in test_df["text"]])
    if not ok.any():
        ok = np.ones(len(y), dtype=bool)
    pred = (p >= threshold).astype(int)
    wrong = (pred != y) & ok

    def pick(mask: np.ndarray, score: np.ndarray, best) -> int:
        idx = np.flatnonzero(mask)
        return int(ids[idx[best(score[idx])]])

    return {
        "clear_happiness": pick((y == 1) & ok, p, np.argmax),
        "clear_sadness": pick((y == 0) & ok, p, np.argmin),
        "close_call": pick(ok, np.abs(p - 0.5), np.argmin),
        "model_miss": pick(wrong, np.abs(p - 0.5), np.argmax) if wrong.any() else None,
    }


@pytest.fixture
def stage(tmp_path: Path, monkeypatch, toy_params, corpus_split, helpers):
    """Everything the presets stage reads, with one guaranteed confident miss."""

    def _stage(test_df: pd.DataFrame | None = None) -> dict:
        train_df, held_out = corpus_split
        pipeline = helpers["fit_toy_pipeline"](train_df, "logreg")
        if test_df is None:
            test_df = held_out.copy()
            p0 = pipeline.predict_proba(test_df["text"].tolist())[:, 1]
            # Flip the second most confident happiness row so a confident miss exists.
            happy = np.flatnonzero(test_df["label"].to_numpy() == 1)
            second = happy[np.argsort(p0[happy])[::-1][1]]
            test_df.loc[second, "label"] = 0
        models = tmp_path / "models"
        reports = tmp_path / "reports"
        raw = tmp_path / "raw"
        configs = tmp_path / "configs"
        models.mkdir(exist_ok=True)
        joblib.dump(pipeline, models / "model.joblib", compress=3)
        metrics = {
            "model": "logreg",
            "model_version": MODEL_VERSION,
            "threshold": 0.5,
            "accuracy": 0.9,
            "roc_auc": 0.95,
            "n_test": len(test_df),
        }
        helpers["write_json"](reports / "metrics.json", metrics)
        helpers["write_json"](
            models / "version.json", {"model": "logreg", "model_version": MODEL_VERSION}
        )
        helpers["write_csv"](raw / "test.csv", test_df)
        monkeypatch.setattr(settings, "load_params", lambda path=None: toy_params)
        monkeypatch.setattr(settings, "MODEL_PATH", models / "model.joblib")
        monkeypatch.setattr(settings, "VERSION_PATH", models / "version.json")
        monkeypatch.setattr(settings, "METRICS_PATH", reports / "metrics.json")
        monkeypatch.setattr(settings, "RAW_TEST_PATH", raw / "test.csv")
        monkeypatch.setattr(settings, "CONFIGS_DIR", configs)
        monkeypatch.setattr(settings, "PRESETS_PATH", configs / "presets.json")
        p = pipeline.predict_proba(test_df["text"].tolist())[:, 1]
        return {
            "pipeline": pipeline,
            "test": test_df.reset_index(drop=True),
            "p": p,
            "path": configs / "presets.json",
        }

    return _stage


def run(s: dict, helpers) -> dict:
    assert presets.main([]) == 0
    return helpers["read_json"](s["path"])


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------


def test_main_writes_the_presets_file(stage, helpers):
    s = stage()
    doc = run(s, helpers)
    assert set(doc) == {"presets", "chosen_by", "model_version"}
    assert doc["chosen_by"] == CHOSEN_BY
    assert doc["model_version"] == MODEL_VERSION
    assert [preset["id"] for preset in doc["presets"]] == IDS
    raw = s["path"].read_bytes()
    assert b"\r\n" not in raw


def test_each_preset_has_the_contract_keys(stage, helpers):
    doc = run(stage(), helpers)
    for preset in doc["presets"]:
        assert set(preset) == PRESET_KEYS, preset["id"]


def test_titles_are_sentence_case(stage, helpers):
    doc = run(stage(), helpers)
    titles = [preset["title"] for preset in doc["presets"]]
    assert titles == ["Clear happiness", "Clear sadness", "Close call", "Model miss"]


def test_descriptions_are_one_plain_sentence(stage, helpers):
    doc = run(stage(), helpers)
    seen = set()
    for preset in doc["presets"]:
        text = preset["description"]
        assert text[0].isupper()
        assert text.endswith(".")
        assert text.count(".") == 1
        assert EM_DASH not in text
        seen.add(text)
    assert len(seen) == 4


def test_descriptions_say_the_pick_is_among_readable_tweets(stage, helpers):
    doc = run(stage(), helpers)
    for preset in doc["presets"]:
        assert qualifies(preset["text"]), preset["id"]
        assert preset["description"].startswith("Among readable held-out tweets, "), preset["id"]
    by_id = {preset["id"]: preset["description"] for preset in doc["presets"]}
    assert by_id["clear_happiness"] == (
        "Among readable held-out tweets, the one the model is most sure is happiness."
    )
    assert by_id["clear_sadness"] == (
        "Among readable held-out tweets, the one the model is most sure is sadness."
    )
    assert by_id["close_call"] == (
        "Among readable held-out tweets, the one whose score sits closest to the decision "
        "threshold."
    )
    assert by_id["model_miss"] == (
        "Among readable held-out tweets, the one the model gets wrong with the most confidence."
    )


@pytest.mark.parametrize("preset_id", IDS)
def test_describe_names_the_pool(preset_id):
    readable = presets.describe(preset_id, True)
    fallback = presets.describe(preset_id, False)
    assert readable.startswith("Among readable held-out tweets, the one ")
    assert fallback.startswith("Among all held-out tweets, the one ")
    assert readable.split(", ", 1)[1] == fallback.split(", ", 1)[1]
    for text in (readable, fallback):
        assert text.endswith(".")
        assert text.count(".") == 1
        assert EM_DASH not in text


def test_describe_rejects_an_unknown_preset():
    with pytest.raises(KeyError):
        presets.describe("clear_anger", True)


def test_presets_carry_real_held_out_rows(stage, helpers):
    s = stage()
    doc = run(s, helpers)
    by_id = s["test"].set_index("tweet_id")
    for preset in doc["presets"]:
        assert preset["tweet_id"] in by_id.index
        row = by_id.loc[preset["tweet_id"]]
        assert preset["text"] == row["text"]
        assert preset["label"] == settings.LABELS[int(row["label"])]


def test_probabilities_are_the_pipelines_own_scores(stage, helpers):
    s = stage()
    doc = run(s, helpers)
    ids = s["test"]["tweet_id"].tolist()
    for preset in doc["presets"]:
        p = float(s["p"][ids.index(preset["tweet_id"])])
        assert preset["probability_happiness"] == round(p, 4)
        assert 0.0 <= preset["probability_happiness"] <= 1.0


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_selection_follows_the_scores(stage, helpers):
    s = stage()
    doc = run(s, helpers)
    expected = expected_picks(s["test"], s["p"])
    got = {preset["id"]: preset["tweet_id"] for preset in doc["presets"]}
    assert got["clear_happiness"] == expected["clear_happiness"]
    assert got["clear_sadness"] == expected["clear_sadness"]
    assert got["close_call"] == expected["close_call"]
    assert expected["model_miss"] is not None
    assert got["model_miss"] == expected["model_miss"]


def test_model_miss_is_a_wrong_prediction(stage, helpers):
    s = stage()
    doc = run(s, helpers)
    miss = next(preset for preset in doc["presets"] if preset["id"] == "model_miss")
    predicted = "happiness" if miss["probability_happiness"] >= 0.5 else "sadness"
    assert predicted != miss["label"]


def test_clear_presets_carry_their_labels(stage, helpers):
    doc = run(stage(), helpers)
    by_id = {preset["id"]: preset for preset in doc["presets"]}
    assert by_id["clear_happiness"]["label"] == "happiness"
    assert by_id["clear_happiness"]["probability_happiness"] >= 0.5
    assert by_id["clear_sadness"]["label"] == "sadness"
    assert by_id["clear_sadness"]["probability_happiness"] < 0.5
    assert by_id["clear_happiness"]["probability_happiness"] >= max(
        p["probability_happiness"] for p in doc["presets"]
    )
    assert by_id["clear_sadness"]["probability_happiness"] <= min(
        p["probability_happiness"] for p in doc["presets"]
    )


def test_close_call_is_the_nearest_to_one_half(stage, helpers):
    s = stage()
    doc = run(s, helpers)
    close = next(preset for preset in doc["presets"] if preset["id"] == "close_call")
    nearest = float(np.min(np.abs(s["p"] - 0.5)))
    assert abs(close["probability_happiness"] - 0.5) <= nearest + 1e-4


def test_readability_preference_skips_mentions_and_urls(stage, helpers):
    base = stage()
    ids = base["test"]["tweet_id"].tolist()
    picks = expected_picks(base["test"], base["p"])
    modified = base["test"].copy()
    happy_row = ids.index(picks["clear_happiness"])
    sad_row = ids.index(picks["clear_sadness"])
    modified.loc[happy_row, "text"] = modified.loc[happy_row, "text"] + " @someone"
    modified.loc[sad_row, "text"] = modified.loc[sad_row, "text"] + " http://t.co/x"
    s = stage(modified)
    doc = run(s, helpers)
    expected = expected_picks(s["test"], s["p"])
    got = {preset["id"]: preset["tweet_id"] for preset in doc["presets"]}
    assert got["clear_happiness"] != picks["clear_happiness"]
    assert got["clear_sadness"] != picks["clear_sadness"]
    assert got["clear_happiness"] == expected["clear_happiness"]
    assert got["clear_sadness"] == expected["clear_sadness"]
    for preset in doc["presets"]:
        assert qualifies(preset["text"]), preset["id"]


def test_falls_back_to_any_tweet_when_none_qualify(stage, helpers):
    base = stage()
    unreadable = base["test"].copy()
    unreadable["text"] = "@" + unreadable["text"]
    s = stage(unreadable)
    doc = run(s, helpers)
    assert [preset["id"] for preset in doc["presets"]] == IDS
    expected = expected_picks(s["test"], s["p"])
    got = {preset["id"]: preset["tweet_id"] for preset in doc["presets"]}
    assert got["clear_happiness"] == expected["clear_happiness"]
    assert got["clear_sadness"] == expected["clear_sadness"]
    assert all(preset["text"].startswith("@") for preset in doc["presets"])
    for preset in doc["presets"]:
        assert preset["description"].startswith("Among all held-out tweets, "), preset["id"]


# ---------------------------------------------------------------------------
# Pure helpers on a synthetic scored frame
# ---------------------------------------------------------------------------


def scored_frame() -> pd.DataFrame:
    rows = [
        (1, "very happy text here", 1, 0.97),
        (2, "so sad text here", 0, 0.03),
        (3, "middle of the road", 1, 0.52),
        (4, "confident miss here", 0, 0.91),
        (5, "another happy line", 1, 0.80),
        (6, "another sad line", 0, 0.20),
        (7, "mild miss here", 1, 0.40),
    ]
    columns = ["tweet_id", "text", "label", "probability_happiness"]
    return pd.DataFrame(rows, columns=columns)


def chosen_ids(df: pd.DataFrame, threshold: float = 0.5) -> dict[str, int]:
    chosen = presets.choose_presets(df, threshold)
    return {key: int(df.loc[idx, "tweet_id"]) for key, idx in chosen.items()}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a fine tweet", True),
        ("short", False),
        ("x" * 6, True),
        ("x" * 140, True),
        ("x" * 141, False),
        ("hello @friend", False),
        ("see http://t.co/x now", False),
        ("see www.example.com now", False),
        ("email me at home", True),
    ],
)
def test_is_readable(text, expected):
    assert presets.is_readable(text) is expected


def test_choose_presets_on_a_scored_frame():
    assert chosen_ids(scored_frame()) == {
        "clear_happiness": 1,
        "clear_sadness": 2,
        "close_call": 3,
        "model_miss": 4,
    }


def test_choose_presets_honours_the_threshold():
    ids = chosen_ids(scored_frame(), 0.45)
    assert ids["close_call"] == 7
    assert ids["model_miss"] == 4


def test_choose_presets_prefers_readable_rows():
    df = scored_frame()
    df.loc[0, "text"] = "very happy @friend"
    df.loc[1, "text"] = "so sad http://t.co/x"
    ids = chosen_ids(df)
    assert ids["clear_happiness"] == 5
    assert ids["clear_sadness"] == 6


def test_choose_presets_falls_back_when_nothing_is_readable():
    df = scored_frame()
    df["text"] = "@" + df["text"]
    assert chosen_ids(df) == {
        "clear_happiness": 1,
        "clear_sadness": 2,
        "close_call": 3,
        "model_miss": 4,
    }


def test_choose_presets_never_reuses_a_row():
    chosen = presets.choose_presets(scored_frame(), 0.5)
    assert list(chosen) == IDS
    assert len(set(chosen.values())) == 4


def test_choose_presets_without_a_miss():
    df = scored_frame().iloc[[0, 1, 2, 4, 5]].reset_index(drop=True)
    ids = chosen_ids(df)
    assert "model_miss" not in ids
    assert ids["clear_happiness"] == 1


def test_build_presets_payload():
    out = presets.build_presets(scored_frame(), 0.5, "0.0.0+test", 4)
    assert set(out) == {"presets", "chosen_by", "model_version"}
    assert out["chosen_by"] == CHOSEN_BY
    assert out["model_version"] == "0.0.0+test"
    assert [p["id"] for p in out["presets"]] == IDS
    first = out["presets"][0]
    assert set(first) == PRESET_KEYS
    assert first["title"] == "Clear happiness"
    assert first["text"] == "very happy text here"
    assert first["description"] == (
        "Among readable held-out tweets, the one the model is most sure is happiness."
    )
    assert first["label"] == "happiness"
    assert first["probability_happiness"] == 0.97
    assert first["tweet_id"] == 1
    miss = out["presets"][3]
    assert miss["label"] == "sadness"
    assert miss["tweet_id"] == 4


def test_build_presets_caps_the_count():
    out = presets.build_presets(scored_frame(), 0.5, "0.0.0+test", 2)
    assert [p["id"] for p in out["presets"]] == IDS[:2]


def test_main_creates_the_configs_directory(stage, helpers):
    s = stage()
    assert not s["path"].parent.exists()
    run(s, helpers)
    assert s["path"].is_file()


def test_stage_is_deterministic(stage, helpers):
    s = stage()
    first = run(s, helpers)
    second = run(s, helpers)
    assert first == second

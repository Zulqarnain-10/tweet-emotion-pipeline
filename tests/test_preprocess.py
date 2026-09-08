"""Unit tests for tweet_emotion.preprocess: TextNormalizer, the stop-word loader, WordNet
setup, and the stage.

Every normalisation case runs with lemmatize=False so it needs no corpus. The lemmatiser
cases use the wordnet fixture, which skips when the corpus is not installed.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from tweet_emotion import preprocess, settings
from tweet_emotion.preprocess import TextNormalizer, ensure_wordnet, load_stopwords, normalize_text

FLAGS = (
    "lowercase",
    "strip_urls",
    "strip_mentions",
    "strip_numbers",
    "strip_punctuation",
    "remove_stopwords",
    "lemmatize",
)
MESSY = [
    "I am SO happy today!!! :)",
    "@jenny lost my keys AGAIN http://bit.ly/abc123",
    "&quot;Best day ever&quot; said the 3 kids &amp; me",
    "Rainy day, cancelled plans, feeling low\u2026",
    "",
    "   ",
    "12345",
    "www.example.com/party 2009",
]


def make(**overrides) -> TextNormalizer:
    """Every flag on except lemmatize, with the given overrides."""
    kwargs = dict.fromkeys(FLAGS, True)
    kwargs["lemmatize"] = False
    kwargs.update(overrides)
    return TextNormalizer(**kwargs)


def one(text, **overrides) -> str:
    return make(**overrides).transform([text])[0]


# ---------------------------------------------------------------------------
# Stop words
# ---------------------------------------------------------------------------


def test_load_stopwords_default_path():
    words = load_stopwords()
    assert isinstance(words, frozenset)
    assert len(words) == 198
    assert {"the", "is", "and", "a"} <= words
    assert "happy" not in words


def test_load_stopwords_custom_file(tmp_path: Path):
    path = tmp_path / "stop.txt"
    path.write_text("alpha\nBeta \n\n gamma\n", encoding="utf-8")
    words = load_stopwords(path)
    assert isinstance(words, frozenset)
    assert {"alpha", "gamma"} <= words
    assert "" not in words
    assert len(words) == 3


def test_normalizer_loads_stopwords_when_none_given():
    normalizer = make(remove_stopwords=True)
    assert isinstance(normalizer.stopwords, frozenset)
    assert normalizer.stopwords == load_stopwords()


def test_normalizer_keeps_a_custom_stopword_set():
    normalizer = make(stopwords=frozenset({"cat"}))
    assert normalizer.transform(["the cat sat"]) == ["the sat"]


# ---------------------------------------------------------------------------
# keep_words
# ---------------------------------------------------------------------------


def test_keep_words_default_to_none():
    normalizer = make()
    assert normalizer.keep_words is None
    assert "keep_words" in normalizer.get_params()


def test_keep_words_are_taken_out_of_the_default_stop_list():
    normalizer = make(keep_words=["not", "NOT", "no", "Nor"])
    assert normalizer.keep_words == ("no", "nor", "not")
    assert normalizer.stopwords == load_stopwords() - {"no", "nor", "not"}
    assert normalizer.transform(["i do not like it, no"]) == ["not like no"]


def test_keep_words_are_taken_out_of_a_custom_stop_list():
    normalizer = make(stopwords=frozenset({"cat", "sat", "the"}), keep_words=("sat",))
    assert normalizer.stopwords == frozenset({"cat", "the"})
    assert normalizer.transform(["the cat sat"]) == ["sat"]


def test_keep_words_accept_any_iterable_and_sort_them():
    assert make(keep_words={"b", "a"}).keep_words == ("a", "b")
    assert make(keep_words=iter(["b", "a", "b"])).keep_words == ("a", "b")
    assert make(keep_words=[]).keep_words == ()


def test_keep_words_reject_a_bare_string():
    with pytest.raises(TypeError):
        make(keep_words="not")


def test_keep_words_absent_from_the_stop_list_change_nothing():
    plain = make()
    kept = make(keep_words=["happy", "sunshine"])
    assert kept.stopwords == plain.stopwords
    assert kept.transform(MESSY) == plain.transform(MESSY)


def test_direct_constructor_and_from_params_agree(params):
    block = dict(params["preprocess"], lemmatize=False)
    direct = TextNormalizer(lemmatize=False, keep_words=block["keep_words"])
    via_params = TextNormalizer.from_params(block)
    assert direct.stopwords == via_params.stopwords
    assert direct.keep_words == via_params.keep_words
    assert direct.get_params() == via_params.get_params()
    texts = [*MESSY, "I do NOT like this, don't you know", "no, nor that, can't say"]
    assert direct.transform(texts) == via_params.transform(texts)


def test_clone_preserves_keep_words(params):
    normalizer = TextNormalizer.from_params(dict(params["preprocess"], lemmatize=False))
    copy = clone(normalizer)
    assert copy.keep_words == normalizer.keep_words
    assert copy.stopwords == normalizer.stopwords
    assert copy.transform(MESSY) == normalizer.transform(MESSY)


# ---------------------------------------------------------------------------
# One string at a time
# ---------------------------------------------------------------------------


def test_lowercase():
    assert one("HELLO World") == "hello world"


def test_case_is_kept_when_lowercase_is_off():
    assert one("HELLO World", lowercase=False, remove_stopwords=False) == "HELLO World"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("check https://example.com/a?b=1 now", "check now"),
        ("check http://t.co/xyz now", "check now"),
        ("see www.example.com/party today", "see today"),
        ("https://only.example.com", ""),
    ],
)
def test_strip_urls(text, expected):
    assert one(text, remove_stopwords=False) == expected


def test_urls_survive_when_strip_urls_is_off():
    out = one("see https://example.com now", strip_urls=False, remove_stopwords=False)
    assert "example" in out
    assert "https" in out


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("@bob thanks", "thanks"),
        ("thanks @bob_smith and @amy2", "thanks and"),
        ("@only", ""),
    ],
)
def test_strip_mentions(text, expected):
    assert one(text, remove_stopwords=False) == expected


def test_mentions_survive_when_strip_mentions_is_off():
    assert one("@bob thanks", strip_mentions=False, remove_stopwords=False) == "bob thanks"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("&quot;wow&quot; &amp; fine", "wow fine"),
        ("love you &lt;3", "love you"),
        ("a &gt; b", "a b"),
        ("&amp;&amp;", ""),
    ],
)
def test_html_entities_are_unescaped_then_treated_as_punctuation(text, expected):
    assert one(text, remove_stopwords=False) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("i have 2 cats and 10 dogs", "i have cats and dogs"),
        ("2009", ""),
        ("h1n1 flu", "hn flu"),
    ],
)
def test_strip_numbers(text, expected):
    assert one(text, remove_stopwords=False) == expected


def test_numbers_survive_when_strip_numbers_is_off():
    assert one("i have 2 cats", strip_numbers=False, remove_stopwords=False) == "i have 2 cats"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello!!! world...", "hello world"),
        ("well, (that) was: great; right?", "well that was great right"),
        ("\u201cquoted\u201d it\u2019s fine\u2026", "quoted it s fine"),
        ("what#a$day%", "what a day"),
        ("...", ""),
    ],
)
def test_strip_punctuation(text, expected):
    assert one(text, remove_stopwords=False) == expected


def test_punctuation_survives_when_strip_punctuation_is_off():
    assert one("hello, world!", strip_punctuation=False, remove_stopwords=False) == "hello, world!"


def test_remove_stopwords():
    assert one("the cat is on the mat") == "cat mat"


def test_stopwords_survive_when_remove_stopwords_is_off():
    assert one("the cat is on the mat", remove_stopwords=False) == "the cat is on the mat"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  a   b \t c\n", "a b c"),
        ("a\n\nb", "a b"),
        (" a ", "a"),
    ],
)
def test_whitespace_is_collapsed(text, expected):
    assert one(text, remove_stopwords=False) == expected


@pytest.mark.parametrize(
    "value",
    ["", "   ", None, float("nan"), np.nan, "@bob 123 !!!", "&amp;"],
    ids=["empty", "blank", "none", "nan", "np_nan", "only_noise", "only_entity"],
)
def test_empty_results_are_empty_strings(value):
    out = make().transform([value])
    assert out == [""]
    assert isinstance(out[0], str)


def test_non_string_values_are_cast():
    assert one(42, remove_stopwords=False, strip_numbers=False) == "42"
    assert one(3.5, remove_stopwords=False, strip_numbers=False, strip_punctuation=False) == "3.5"


def test_all_flags_off_only_collapses_whitespace():
    normalizer = TextNormalizer(**dict.fromkeys(FLAGS, False))
    assert normalizer.transform(["  Hello,  World! 123 @x http://t.co  "]) == [
        "Hello, World! 123 @x http://t.co"
    ]


def test_idempotent():
    normalizer = make()
    once = normalizer.transform(MESSY)
    assert normalizer.transform(once) == once


def test_full_messy_examples():
    assert make().transform(MESSY) == [
        "happy today",
        "lost keys",
        "best day ever said kids",
        "rainy day cancelled plans feeling low",
        "",
        "",
        "",
        "",
    ]


# ---------------------------------------------------------------------------
# sklearn behaviour
# ---------------------------------------------------------------------------


def test_fit_returns_self_and_transform_returns_a_list():
    normalizer = make()
    assert normalizer.fit(["a b"]) is normalizer
    out = normalizer.transform(["A b", "C"])
    assert isinstance(out, list)
    assert all(isinstance(item, str) for item in out)
    assert normalizer.fit_transform(["A b", "C"]) == out


@pytest.mark.parametrize(
    "container",
    [list, tuple, pd.Series, np.array],
    ids=["list", "tuple", "series", "ndarray"],
)
def test_transform_accepts_any_iterable_of_strings(container):
    texts = container(["Hello World", "Sad day", ""])
    out = make(remove_stopwords=False).transform(texts)
    assert out == ["hello world", "sad day", ""]


def test_transform_keeps_length_and_order(corpus):
    out = make().transform(corpus["text"])
    assert len(out) == len(corpus)
    assert out[:3] == make().transform(corpus["text"].tolist()[:3])


def test_get_params_exposes_every_flag():
    params = make().get_params()
    assert set(FLAGS) <= set(params)
    assert "stopwords" in params


def test_clone_preserves_behaviour():
    normalizer = make()
    copy = clone(normalizer)
    assert isinstance(copy, TextNormalizer)
    assert copy.transform(MESSY) == normalizer.transform(MESSY)


def test_from_params_mirrors_the_params_block(params):
    block = params["preprocess"]
    normalizer = TextNormalizer.from_params(block)
    for flag in FLAGS:
        assert getattr(normalizer, flag) == block[flag], flag
    if block["remove_stopwords"]:
        keep = {str(word).lower() for word in block.get("keep_words", [])}
        assert normalizer.stopwords == load_stopwords() - keep
        assert not keep & normalizer.stopwords
        assert normalizer.keep_words == tuple(sorted(keep))


def test_from_params_accepts_overrides(params):
    block = dict(params["preprocess"])
    block.update(lemmatize=False, strip_numbers=False)
    normalizer = TextNormalizer.from_params(block)
    assert normalizer.lemmatize is False
    assert normalizer.strip_numbers is False
    assert normalizer.transform(["2 cats"]) == ["2 cats"]


def test_normalize_text_matches_transform():
    normalizer = make()
    for text in MESSY:
        assert normalize_text(text, normalizer) == normalizer.transform([text])[0]


def test_pickle_round_trip_is_self_contained():
    normalizer = make()
    restored = pickle.loads(pickle.dumps(normalizer))
    assert isinstance(restored, TextNormalizer)
    assert isinstance(restored.stopwords, frozenset)
    assert restored.stopwords == normalizer.stopwords
    assert restored.keep_words == normalizer.keep_words
    assert restored.transform(MESSY) == normalizer.transform(MESSY)


def test_pickle_state_stores_the_stop_words_as_a_sorted_list(params):
    normalizer = TextNormalizer.from_params(dict(params["preprocess"], lemmatize=False))
    state = normalizer.__getstate__()
    assert isinstance(state["stopwords"], list)
    assert state["stopwords"] == sorted(normalizer.stopwords)
    assert "_wnl" not in state
    restored = pickle.loads(pickle.dumps(normalizer))
    assert isinstance(restored.stopwords, frozenset)
    assert restored.stopwords == normalizer.stopwords
    assert restored.keep_words == normalizer.keep_words


def test_pickle_bytes_are_identical_within_one_process(params):
    block = dict(params["preprocess"], lemmatize=False)
    first = pickle.dumps(TextNormalizer.from_params(block))
    second = pickle.dumps(TextNormalizer.from_params(block))
    assert first == second


HASH_SEED_SCRIPT = (
    "import hashlib, pickle\n"
    "from tweet_emotion import settings\n"
    "from tweet_emotion.preprocess import TextNormalizer\n"
    "normalizer = TextNormalizer.from_params(settings.load_params()['preprocess'])\n"
    "print(hashlib.sha256(pickle.dumps(normalizer)).hexdigest())\n"
)


def test_pickle_bytes_do_not_depend_on_the_hash_seed():
    """The stop set is a frozenset, whose iteration order follows PYTHONHASHSEED; the pickle
    must not, or models/model.joblib changes on every train run with an unchanged model."""
    digests = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", HASH_SEED_SCRIPT],
            cwd=settings.REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        digests.append(result.stdout.strip())
    assert re.fullmatch(r"[0-9a-f]{64}", digests[0])
    assert digests[0] == digests[1]


def test_fresh_normalizer_holds_no_lemmatizer_instance():
    normalizer = make(lemmatize=True)
    held = [type(value).__name__ for value in vars(normalizer).values()]
    assert "WordNetLemmatizer" not in held


def test_normalizer_works_inside_a_pipeline(corpus):
    pipeline = Pipeline(
        [
            ("normalize", make()),
            ("vectorize", TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b\w+\b")),
        ]
    )
    X = pipeline.fit_transform(corpus["text"])
    assert X.shape[0] == len(corpus)
    vocabulary = set(pipeline.named_steps["vectorize"].vocabulary_)
    assert not vocabulary & load_stopwords()
    assert all(term == term.lower() for term in vocabulary)


# ---------------------------------------------------------------------------
# Lemmatisation
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("wordnet")
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("cats dogs", "cat dog"),
        ("mice geese", "mouse goose"),
        ("running", "running"),
        ("the children are happy", "child happy"),
    ],
)
def test_lemmatize_uses_the_noun_default(text, expected):
    assert one(text, lemmatize=True) == expected


@pytest.mark.usefixtures("wordnet")
def test_lemmatizer_is_created_lazily_and_pickles_without_it():
    normalizer = make(lemmatize=True)
    assert normalizer.transform(["cats"]) == ["cat"]
    assert hasattr(normalizer._lemmatizer, "lemmatize")
    payload = pickle.dumps(normalizer)
    assert b"WordNetLemmatizer" not in payload
    restored = pickle.loads(payload)
    assert restored.transform(["cats dogs"]) == ["cat dog"]


@pytest.mark.usefixtures("wordnet")
def test_ensure_wordnet_is_idempotent():
    assert ensure_wordnet() is None
    assert ensure_wordnet() is None


def test_ensure_wordnet_downloads_when_the_corpus_is_missing(monkeypatch):
    import nltk

    downloaded: list[str] = []

    def missing(resource, *args, **kwargs):
        # Absent until the download has been recorded, present afterwards.
        if any(name in str(resource) for name in downloaded):
            return "found"
        raise LookupError(resource)

    def record(name, *args, **kwargs):
        downloaded.append(name)
        return True

    monkeypatch.setattr(nltk.data, "find", missing)
    monkeypatch.setattr(nltk, "download", record)
    assert ensure_wordnet() is None
    assert "wordnet" in downloaded
    assert "omw-1.4" in downloaded


def test_ensure_wordnet_skips_the_download_when_present(monkeypatch):
    import nltk

    calls: list[str] = []
    monkeypatch.setattr(nltk.data, "find", lambda resource, *a, **k: "found")
    monkeypatch.setattr(nltk, "download", lambda name, *a, **k: calls.append(name))
    assert ensure_wordnet() is None
    assert calls == []


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


def _point_settings_at(monkeypatch, root: Path, params: dict) -> None:
    monkeypatch.setattr(settings, "load_params", lambda path=None: params)
    monkeypatch.setattr(settings, "RAW_DIR", root / "raw")
    monkeypatch.setattr(settings, "RAW_TRAIN_PATH", root / "raw" / "train.csv")
    monkeypatch.setattr(settings, "RAW_TEST_PATH", root / "raw" / "test.csv")
    monkeypatch.setattr(settings, "PROCESSED_DIR", root / "processed")
    monkeypatch.setattr(settings, "PROCESSED_TRAIN_PATH", root / "processed" / "train.csv")
    monkeypatch.setattr(settings, "PROCESSED_TEST_PATH", root / "processed" / "test.csv")


def test_main_normalises_both_splits(tmp_path: Path, monkeypatch, toy_params, helpers, caplog):
    train = pd.DataFrame(
        {
            "tweet_id": [1, 2, 3, 4],
            "text": ["I am SO happy today!!!", "@bob 123 !!!", "Rainy day, feeling low", ""],
            "label": [1, 0, 0, 1],
        }
    )
    test = pd.DataFrame(
        {"tweet_id": [7, 8], "text": ["&quot;Best day&quot; &amp; me", "   "], "label": [1, 0]}
    )
    helpers["write_csv"](tmp_path / "raw" / "train.csv", train)
    helpers["write_csv"](tmp_path / "raw" / "test.csv", test)
    _point_settings_at(monkeypatch, tmp_path, toy_params)

    with caplog.at_level(logging.INFO):
        assert preprocess.main([]) == 0

    out_train = pd.read_csv(tmp_path / "processed" / "train.csv", keep_default_na=False)
    out_test = pd.read_csv(tmp_path / "processed" / "test.csv", keep_default_na=False)
    assert list(out_train.columns) == list(settings.CSV_COLUMNS)
    assert list(out_test.columns) == list(settings.CSV_COLUMNS)
    assert out_train["tweet_id"].tolist() == [1, 2, 3, 4]
    assert out_train["label"].tolist() == [1, 0, 0, 1]
    assert out_test["tweet_id"].tolist() == [7, 8]
    expected = TextNormalizer.from_params(toy_params["preprocess"])
    assert out_train["text"].tolist() == expected.transform(train["text"])
    assert out_train["text"].tolist() == ["happy today", "", "rainy day feeling low", ""]
    assert out_test["text"].tolist() == ["best day", ""]
    assert all(isinstance(value, str) for value in out_train["text"])
    assert b"\r\n" not in (tmp_path / "processed" / "train.csv").read_bytes()
    assert any("empty" in record.getMessage().lower() for record in caplog.records)


def _write_tiny_splits(tmp_path: Path, helpers) -> None:
    train = pd.DataFrame({"tweet_id": [1, 2], "text": ["cats and dogs", "so sad"], "label": [1, 0]})
    test = pd.DataFrame({"tweet_id": [7], "text": ["happy days"], "label": [1]})
    helpers["write_csv"](tmp_path / "raw" / "train.csv", train)
    helpers["write_csv"](tmp_path / "raw" / "test.csv", test)


def test_main_does_not_touch_wordnet_when_lemmatize_is_off(
    tmp_path: Path, monkeypatch, toy_params, helpers
):
    assert toy_params["preprocess"]["lemmatize"] is False
    _write_tiny_splits(tmp_path, helpers)
    _point_settings_at(monkeypatch, tmp_path, toy_params)

    def refuse() -> None:
        raise AssertionError("ensure_wordnet was called although lemmatize is false")

    monkeypatch.setattr(preprocess, "ensure_wordnet", refuse)
    assert preprocess.main([]) == 0
    out = pd.read_csv(tmp_path / "processed" / "train.csv", keep_default_na=False)
    assert out["text"].tolist() == ["cats dogs", "sad"]


@pytest.mark.usefixtures("wordnet")
def test_main_ensures_wordnet_when_lemmatize_is_on(tmp_path: Path, monkeypatch, params, helpers):
    lemmatizing = {**params, "preprocess": dict(params["preprocess"], lemmatize=True)}
    _write_tiny_splits(tmp_path, helpers)
    _point_settings_at(monkeypatch, tmp_path, lemmatizing)
    calls: list[str] = []
    monkeypatch.setattr(preprocess, "ensure_wordnet", lambda: calls.append("ensure"))
    assert preprocess.main([]) == 0
    assert calls[0] == "ensure"
    out = pd.read_csv(tmp_path / "processed" / "train.csv", keep_default_na=False)
    assert out["text"].tolist() == ["cat dog", "sad"]


def test_main_fails_without_the_raw_split(tmp_path: Path, monkeypatch, toy_params):
    _point_settings_at(monkeypatch, tmp_path, toy_params)
    try:
        code = preprocess.main([])
    except (OSError, ValueError):
        return
    assert code != 0

"""Unit tests for tweet_emotion.ingest: hashing, the binary selection, the stratified split,
the download helper, and the stage with its fetch manifest.

No network: download is exercised with a fake urlopen, and the stage with a fake download.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tweet_emotion import ingest, settings

MANIFEST_KEYS = {
    "source_url",
    "fetched_from",
    "sha256",
    "bytes",
    "downloaded_at",
    "n_rows_total",
    "n_duplicate_texts_total",
    "label_counts_total",
    "keep_labels",
    "positive_label",
    "n_kept_before_dedup",
    "n_duplicates_dropped",
    "n_kept",
    "test_size",
    "seed",
    "n_train",
    "n_test",
    "positive_rate_train",
    "positive_rate_test",
}
ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
LABEL_COUNTS = {
    "happiness": 6,
    "sadness": 7,
    "neutral": 1,
    "worry": 1,
    "love": 1,
    "surprise": 1,
    "fun": 1,
    "relief": 1,
    "hate": 1,
    "empty": 1,
    "enthusiasm": 1,
    "boredom": 1,
    "anger": 1,
}


def select(frame: pd.DataFrame, data: dict) -> pd.DataFrame:
    """select_binary with the data block; falls back to the whole params shape."""
    try:
        return ingest.select_binary(frame, data)
    except KeyError:
        return ingest.select_binary(frame, {"data": data})


def labelled(n: int, positives: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.array([1] * positives + [0] * (n - positives))
    rng.shuffle(labels)
    ids = rng.permutation(np.arange(10_000, 10_000 + n))
    return pd.DataFrame(
        {"tweet_id": ids, "text": [f"tweet number {i}" for i in ids], "label": labels},
        columns=list(settings.CSV_COLUMNS),
    )


class FakeResponse:
    """The slice of an HTTPResponse the streaming download needs."""

    def __init__(self, body: bytes) -> None:
        self._body = body
        self._pos = 0
        self.headers = {"Content-Length": str(len(body))}
        self.status = 200

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunk, self._pos = self._body[self._pos :], len(self._body)
        else:
            chunk = self._body[self._pos : self._pos + size]
            self._pos += len(chunk)
        return chunk

    def __iter__(self):
        while True:
            chunk = self.read(4096)
            if not chunk:
                return
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self) -> None:
        return None


def install_fake_urlopen(monkeypatch, body: bytes) -> list[dict]:
    seen: list[dict] = []

    def fake_urlopen(request, *args, **kwargs):
        headers = {}
        url = request
        if isinstance(request, urllib.request.Request):
            headers = {k.lower(): v for k, v in request.header_items()}
            url = request.full_url
        seen.append({"url": url, "headers": headers, "timeout": kwargs.get("timeout")})
        return FakeResponse(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    if hasattr(ingest, "urlopen"):
        monkeypatch.setattr(ingest, "urlopen", fake_urlopen)
    return seen


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def test_sha256_of_matches_hashlib_across_chunks(tmp_path: Path):
    body = bytes(range(256)) * 1200  # 300 KB, larger than any sane read chunk
    path = tmp_path / "blob.bin"
    path.write_bytes(body)
    assert ingest.sha256_of(path) == hashlib.sha256(body).hexdigest()


def test_sha256_of_an_empty_file(tmp_path: Path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert ingest.sha256_of(path) == hashlib.sha256(b"").hexdigest()


def test_verify_sha256_accepts_the_right_hash(tmp_path: Path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"abc")
    expected = hashlib.sha256(b"abc").hexdigest()
    result = ingest.verify_sha256(path, expected)
    assert result is None or result == expected


def test_verify_sha256_reports_both_hashes_on_mismatch(tmp_path: Path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"abc")
    actual = hashlib.sha256(b"abc").hexdigest()
    expected = "0" * 64
    with pytest.raises(ValueError) as excinfo:
        ingest.verify_sha256(path, expected)
    message = str(excinfo.value)
    assert expected in message
    assert actual in message


# ---------------------------------------------------------------------------
# select_binary
# ---------------------------------------------------------------------------


def test_select_binary_keeps_two_labels_without_duplicates_or_empties(raw_frame, params, helpers):
    out = select(raw_frame, params["data"])
    assert list(out.columns) == list(settings.CSV_COLUMNS)
    assert out["tweet_id"].tolist() == list(helpers["KEPT_IDS"])
    assert len(out) == 10


def test_select_binary_maps_the_positive_label_to_one(raw_frame, params, helpers):
    out = select(raw_frame, params["data"])
    got = dict(zip(out["tweet_id"], out["label"], strict=True))
    assert got == helpers["KEPT_LABELS"]
    assert set(out["label"].unique()) == {0, 1}
    assert pd.api.types.is_integer_dtype(out["label"])


def test_select_binary_resets_the_index(raw_frame, params):
    out = select(raw_frame, params["data"])
    assert out.index.tolist() == list(range(len(out)))


def test_select_binary_keeps_the_first_of_each_duplicate(raw_frame, params):
    out = select(raw_frame, params["data"])
    assert 1 in out["tweet_id"].tolist()
    assert 3 not in out["tweet_id"].tolist()
    assert 2 in out["tweet_id"].tolist()
    assert 19 not in out["tweet_id"].tolist()
    texts = out["text"].str.strip()
    assert texts.is_unique


def test_select_binary_keeps_raw_text_untouched_except_whitespace(raw_frame, params):
    out = select(raw_frame, params["data"]).set_index("tweet_id")
    assert out.loc[16, "text"].strip() == "&quot;Best day ever&quot; said the 3 kids &amp; me"
    assert out.loc[4, "text"].strip() == "Lost my keys AGAIN @jenny http://bit.ly/abc123"


def test_select_binary_drops_whitespace_only_texts(raw_frame, params):
    out = select(raw_frame, params["data"])
    assert 17 not in out["tweet_id"].tolist()
    assert (out["text"].str.strip() != "").all()


def test_select_binary_drops_missing_text(raw_frame, params):
    frame = pd.concat(
        [raw_frame, pd.DataFrame([{"tweet_id": 25, "sentiment": "happiness", "content": None}])],
        ignore_index=True,
    )
    out = select(frame, params["data"])
    assert 25 not in out["tweet_id"].tolist()
    assert out["text"].notna().all()


def test_select_binary_can_keep_duplicates(raw_frame, params):
    data = dict(params["data"], drop_duplicates=False)
    out = select(raw_frame, data)
    assert 3 in out["tweet_id"].tolist()
    assert 19 in out["tweet_id"].tolist()
    assert len(out) == 12


def test_select_binary_honours_the_positive_label(raw_frame, params, helpers):
    data = dict(params["data"], positive_label="sadness")
    out = select(raw_frame, data)
    got = dict(zip(out["tweet_id"], out["label"], strict=True))
    assert got == {k: 1 - v for k, v in helpers["KEPT_LABELS"].items()}


def test_select_binary_honours_keep_labels(raw_frame, params):
    data = dict(params["data"], keep_labels=["love", "hate"], positive_label="love")
    out = select(raw_frame, data)
    assert out["tweet_id"].tolist() == [7, 11]
    assert out["label"].tolist() == [1, 0]


def test_select_binary_leaves_the_input_untouched(raw_frame, params):
    before = raw_frame.copy()
    select(raw_frame, params["data"])
    pd.testing.assert_frame_equal(raw_frame, before)


# ---------------------------------------------------------------------------
# The duplicate rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Happy  Day", "happy day"),
        ("happy day", "happy day"),
        ("  HAPPY\tday \n", "happy day"),
        ("me &amp; you", "me & you"),
        ("&quot;wow&quot;", '"wow"'),
        ("love you &lt;3", "love you <3"),
        ("", ""),
        ("   ", ""),
        (None, ""),
        (float("nan"), ""),
        ("Straße", "strasse"),
    ],
)
def test_duplicate_text_key_folds_case_entities_and_whitespace(text, expected):
    assert ingest.duplicate_text_key(text) == expected


def test_duplicate_text_key_keeps_punctuation_mentions_and_urls():
    assert ingest.duplicate_text_key("Hi @Bob!!! http://t.co/X") == "hi @bob!!! http://t.co/x"


def variants_frame() -> pd.DataFrame:
    rows = [
        (1, "happiness", "Happy DAY &amp; night"),
        (2, "happiness", "happy   day & NIGHT"),
        (3, "sadness", "so sad"),
        (4, "sadness", "  So Sad  "),
        (5, "sadness", "so sad!"),
        (6, "happiness", ""),
        (7, "happiness", "   "),
    ]
    return pd.DataFrame(rows, columns=["tweet_id", "sentiment", "content"])


def test_select_binary_drops_case_and_whitespace_variants_and_keeps_the_original(params):
    out = select(variants_frame(), params["data"]).set_index("tweet_id")
    assert out.index.tolist() == [1, 3, 5]
    assert out.loc[1, "text"] == "Happy DAY &amp; night"
    assert out.loc[3, "text"] == "so sad"
    assert out.loc[5, "text"] == "so sad!"


def test_select_binary_keeps_variants_when_dedup_is_off(params):
    data = dict(params["data"], drop_duplicates=False)
    out = select(variants_frame(), data)
    assert out["tweet_id"].tolist() == [1, 2, 3, 4, 5]


def test_count_duplicate_texts_ignores_empty_texts():
    frame = variants_frame()
    assert ingest.count_duplicate_texts(frame, "content") == 2
    assert ingest.count_duplicate_texts(frame.iloc[[0, 2, 4, 5, 6]], "content") == 0


def test_count_duplicate_texts_on_the_tiny_frame(raw_frame):
    # 3 repeats 1 and 19 repeats 2; the whitespace-only row 17 is empty, not a duplicate.
    assert ingest.count_duplicate_texts(raw_frame, "content") == 2


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def test_split_sizes_and_stratification():
    frame = labelled(100, 60)
    train, test = ingest.split(frame, 0.2, 42)
    assert len(train) == 80
    assert len(test) == 20
    assert int(test["label"].sum()) in {11, 12, 13}
    assert int(train["label"].sum()) == 60 - int(test["label"].sum())


def test_split_is_disjoint_and_complete():
    frame = labelled(100, 60)
    train, test = ingest.split(frame, 0.25, 1)
    assert not set(train["tweet_id"]) & set(test["tweet_id"])
    assert set(train["tweet_id"]) | set(test["tweet_id"]) == set(frame["tweet_id"])
    assert list(train.columns) == list(test.columns) == list(settings.CSV_COLUMNS)


def test_split_sorts_each_side_by_tweet_id():
    frame = labelled(100, 60)
    for part in ingest.split(frame, 0.2, 42):
        assert part["tweet_id"].is_monotonic_increasing
        assert part["tweet_id"].is_unique


def test_split_is_deterministic_and_seed_sensitive():
    frame = labelled(100, 60)
    first = ingest.split(frame, 0.2, 42)
    second = ingest.split(frame, 0.2, 42)
    other = ingest.split(frame, 0.2, 7)
    pd.testing.assert_frame_equal(first[0].reset_index(drop=True), second[0].reset_index(drop=True))
    pd.testing.assert_frame_equal(first[1].reset_index(drop=True), second[1].reset_index(drop=True))
    assert set(first[1]["tweet_id"]) != set(other[1]["tweet_id"])


def test_split_ignores_the_input_row_order():
    frame = labelled(100, 60)
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    a = ingest.split(frame, 0.2, 42)[1]["tweet_id"].tolist()
    b = ingest.split(shuffled, 0.2, 42)[1]["tweet_id"].tolist()
    assert sorted(a) == a
    assert sorted(b) == b


def test_split_on_the_tiny_frame(raw_frame, params):
    kept = select(raw_frame, params["data"])
    train, test = ingest.split(kept, 0.2, 42)
    assert len(test) == 2
    assert len(train) == 8
    assert sorted(test["label"]) == [0, 1]


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


def test_download_streams_the_body_to_dest(tmp_path: Path, monkeypatch):
    body = b"tweet_id,sentiment,content\n" + b"1,happiness,hi\n" * 5000
    seen = install_fake_urlopen(monkeypatch, body)
    dest = tmp_path / "tweet_emotions.csv"
    result = ingest.download("https://example.com/tweets.csv", dest)
    assert Path(result) == dest
    assert dest.read_bytes() == body
    assert not (tmp_path / "tweet_emotions.csv.tmp").exists()
    assert len(seen) == 1
    assert seen[0]["url"] == "https://example.com/tweets.csv"
    assert "user-agent" in seen[0]["headers"]
    assert seen[0]["timeout"] == 60


def test_download_skips_when_the_hash_already_matches(tmp_path: Path, monkeypatch):
    parameters = list(inspect.signature(ingest.download).parameters)
    extra = [name for name in parameters if name not in {"url", "dest"}]
    if not extra:
        pytest.skip("download(url, dest) takes no expected hash; the stage checks before calling")
    body = b"already here\n"
    dest = tmp_path / "tweet_emotions.csv"
    dest.write_bytes(body)

    def explode(*args, **kwargs):
        raise AssertionError("the network was used although the file already matched")

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    if hasattr(ingest, "urlopen"):
        monkeypatch.setattr(ingest, "urlopen", explode)
    kwargs = {extra[0]: hashlib.sha256(body).hexdigest()}
    assert Path(ingest.download("https://example.com/tweets.csv", dest, **kwargs)) == dest
    assert dest.read_bytes() == body


def test_download_leaves_no_partial_file_on_failure(tmp_path: Path, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("connection reset")

    monkeypatch.setattr(urllib.request, "urlopen", broken)
    if hasattr(ingest, "urlopen"):
        monkeypatch.setattr(ingest, "urlopen", broken)
    dest = tmp_path / "tweet_emotions.csv"
    kwargs = {}
    if "retries" in inspect.signature(ingest.download).parameters:
        kwargs["retries"] = 1  # no back-off sleeps in the test
    with pytest.raises((OSError, RuntimeError)):
        ingest.download("https://example.com/tweets.csv", dest, **kwargs)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".tmp").exists()


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


@pytest.fixture
def staged(tmp_path: Path, monkeypatch, raw_frame, params, helpers) -> dict:
    """Settings pointed at tmp, a fake download that writes the tiny raw CSV, matching sha."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = tmp_path / "source.csv"
    helpers["write_csv"](source, raw_frame)
    body = source.read_bytes()
    sha = hashlib.sha256(body).hexdigest()
    stage_params = {**params, "data": dict(params["data"], sha256=sha)}
    calls: list[dict] = []

    def fake_download(url, dest, *args, **kwargs):
        calls.append({"url": url, "dest": Path(dest)})
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(body)
        return Path(dest)

    monkeypatch.setattr(ingest, "download", fake_download)
    monkeypatch.setattr(settings, "load_params", lambda path=None: stage_params)
    monkeypatch.setattr(settings, "RAW_DIR", raw_dir)
    monkeypatch.setattr(settings, "RAW_SOURCE_PATH", raw_dir / "tweet_emotions.csv")
    monkeypatch.setattr(settings, "FETCH_MANIFEST_PATH", raw_dir / "fetch_manifest.json")
    monkeypatch.setattr(settings, "RAW_TRAIN_PATH", raw_dir / "train.csv")
    monkeypatch.setattr(settings, "RAW_TEST_PATH", raw_dir / "test.csv")
    return {
        "raw_dir": raw_dir,
        "params": stage_params,
        "sha": sha,
        "bytes": len(body),
        "calls": calls,
    }


def test_main_writes_the_splits_and_the_manifest(staged, helpers):
    assert ingest.main([]) == 0
    raw_dir = staged["raw_dir"]
    assert staged["calls"][0]["url"] == staged["params"]["data"]["url"]
    train = pd.read_csv(raw_dir / "train.csv")
    test = pd.read_csv(raw_dir / "test.csv")
    assert list(train.columns) == list(settings.CSV_COLUMNS)
    assert list(test.columns) == list(settings.CSV_COLUMNS)
    assert len(train) == 8
    assert len(test) == 2
    assert sorted([*train["tweet_id"], *test["tweet_id"]]) == list(helpers["KEPT_IDS"])
    assert train["tweet_id"].is_monotonic_increasing
    assert test["tweet_id"].is_monotonic_increasing
    for raw in (raw_dir / "train.csv", raw_dir / "test.csv"):
        assert b"\r\n" not in raw.read_bytes()

    manifest = helpers["read_json"](raw_dir / "fetch_manifest.json")
    assert set(manifest) == MANIFEST_KEYS


def test_manifest_values_describe_the_run(staged, helpers):
    assert ingest.main([]) == 0
    manifest = helpers["read_json"](staged["raw_dir"] / "fetch_manifest.json")
    data = staged["params"]["data"]
    assert manifest["source_url"] == data["url"]
    assert manifest["sha256"] == staged["sha"]
    assert manifest["bytes"] == staged["bytes"]
    assert ISO_Z.match(manifest["downloaded_at"])
    assert manifest["n_rows_total"] == 24
    assert manifest["n_duplicate_texts_total"] == 2
    keys = list(manifest)
    assert keys.index("n_duplicate_texts_total") == keys.index("n_rows_total") + 1
    assert manifest["label_counts_total"] == LABEL_COUNTS
    assert sum(manifest["label_counts_total"].values()) == manifest["n_rows_total"]
    assert manifest["keep_labels"] == list(data["keep_labels"])
    assert manifest["positive_label"] == data["positive_label"]
    assert manifest["n_kept_before_dedup"] == 13
    # Two duplicate texts; a stage that folds the whitespace-only text into the count says 3.
    assert manifest["n_duplicates_dropped"] in {2, 3}
    assert manifest["n_kept"] == 10
    after_dedup = manifest["n_kept_before_dedup"] - manifest["n_duplicates_dropped"]
    assert after_dedup >= manifest["n_kept"]
    assert manifest["test_size"] == data["test_size"]
    assert manifest["seed"] == data["seed"]
    assert manifest["n_train"] == 8
    assert manifest["n_test"] == 2
    assert manifest["n_train"] + manifest["n_test"] == manifest["n_kept"]
    assert manifest["positive_rate_train"] == 0.5
    assert manifest["positive_rate_test"] == 0.5
    for key in ("positive_rate_train", "positive_rate_test"):
        assert manifest[key] == round(manifest[key], 4)


def test_main_is_byte_stable(staged):
    assert ingest.main([]) == 0
    first = (staged["raw_dir"] / "train.csv").read_bytes()
    first_test = (staged["raw_dir"] / "test.csv").read_bytes()
    assert ingest.main([]) == 0
    assert (staged["raw_dir"] / "train.csv").read_bytes() == first
    assert (staged["raw_dir"] / "test.csv").read_bytes() == first_test


def test_main_returns_one_on_a_hash_mismatch(staged, monkeypatch, caplog):
    bad = {**staged["params"], "data": dict(staged["params"]["data"], sha256="0" * 64)}
    monkeypatch.setattr(settings, "load_params", lambda path=None: bad)
    with caplog.at_level(logging.ERROR):
        assert ingest.main([]) == 1
    assert not (staged["raw_dir"] / "train.csv").exists()
    assert any(record.levelno >= logging.ERROR for record in caplog.records)


def test_main_returns_one_when_the_download_fails(staged, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr(ingest, "download", broken)
    with caplog.at_level(logging.ERROR):
        assert ingest.main([]) == 1
    assert any("no route to host" in record.getMessage() for record in caplog.records)

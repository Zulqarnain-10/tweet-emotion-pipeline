"""Unit tests for tweet_emotion.setup_nltk and the WordNet helpers it relies on.

Nothing here downloads: ensure_wordnet and find_corpus are stubbed, and the one real run is
guarded by the wordnet fixture, which skips when the corpus is not installed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import nltk
import pytest

from tweet_emotion import preprocess, setup_nltk


def _fake_corpora(monkeypatch, present: dict[str, str]) -> list[str]:
    """find_corpus answers from `present`; ensure_wordnet records its calls and does nothing."""
    calls: list[str] = []

    def find_corpus(package: str) -> str:
        if package in present:
            return present[package]
        raise LookupError(package)

    def ensure_wordnet() -> None:
        calls.append("ensure")

    monkeypatch.setattr(setup_nltk, "find_corpus", find_corpus)
    monkeypatch.setattr(setup_nltk, "ensure_wordnet", ensure_wordnet)
    return calls


def test_packages_are_wordnet_and_omw():
    assert setup_nltk.WORDNET_PACKAGES == ("wordnet", "omw-1.4")


def test_locate_returns_a_path_or_none(monkeypatch):
    _fake_corpora(monkeypatch, {"wordnet": "/data/corpora/wordnet"})
    assert setup_nltk.locate("wordnet") == "/data/corpora/wordnet"
    assert setup_nltk.locate("omw-1.4") is None


def test_main_prints_where_the_corpora_live(monkeypatch, capsys):
    calls = _fake_corpora(
        monkeypatch, {"wordnet": "/data/corpora/wordnet", "omw-1.4": "/data/corpora/omw-1.4"}
    )
    assert setup_nltk.main([]) == 0
    assert calls == ["ensure"]
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["wordnet: /data/corpora/wordnet", "omw-1.4: /data/corpora/omw-1.4"]


def test_main_is_idempotent(monkeypatch, capsys):
    calls = _fake_corpora(monkeypatch, {"wordnet": "/w", "omw-1.4": "/o"})
    assert setup_nltk.main([]) == 0
    assert setup_nltk.main([]) == 0
    assert calls == ["ensure", "ensure"]
    assert capsys.readouterr().out.count("wordnet: /w") == 2


def test_main_fails_when_the_download_raises(monkeypatch, caplog):
    _fake_corpora(monkeypatch, {})

    def boom() -> None:
        raise RuntimeError("no network")

    monkeypatch.setattr(setup_nltk, "ensure_wordnet", boom)
    with caplog.at_level(logging.ERROR):
        assert setup_nltk.main([]) == 1
    assert any("no network" in record.getMessage() for record in caplog.records)


def test_main_fails_when_wordnet_is_still_missing(monkeypatch, caplog):
    _fake_corpora(monkeypatch, {"omw-1.4": "/o"})
    with caplog.at_level(logging.ERROR):
        assert setup_nltk.main([]) == 1
    assert any("wordnet" in record.getMessage() for record in caplog.records)


def test_main_tolerates_a_missing_omw(monkeypatch, caplog, capsys):
    _fake_corpora(monkeypatch, {"wordnet": "/w"})
    with caplog.at_level(logging.WARNING):
        assert setup_nltk.main([]) == 0
    assert any("omw-1.4" in record.getMessage() for record in caplog.records)
    assert capsys.readouterr().out.strip() == "wordnet: /w"


@pytest.mark.usefixtures("wordnet")
def test_main_against_the_installed_corpus(capsys):
    assert setup_nltk.main([]) == 0
    out = capsys.readouterr().out
    assert out.startswith("wordnet: ")
    location = out.splitlines()[0].split(": ", 1)[1]
    # nltk 3.10 serves the corpus from corpora/wordnet.zip, so the printed location may point
    # inside the archive; either the directory or the archive itself must exist on disk.
    archive = location.split(".zip", 1)[0] + ".zip"
    assert Path(location).exists() or Path(archive).is_file()


# ---------------------------------------------------------------------------
# The helpers in preprocess that setup_nltk builds on
# ---------------------------------------------------------------------------


def test_download_dir_defaults_to_none(monkeypatch):
    monkeypatch.delenv("NLTK_DATA", raising=False)
    assert preprocess._nltk_download_dir() is None
    monkeypatch.setenv("NLTK_DATA", "   ")
    assert preprocess._nltk_download_dir() is None


def test_download_dir_uses_the_first_nltk_data_entry(monkeypatch, tmp_path):
    first = tmp_path / "nltk_a"
    second = tmp_path / "nltk_b"
    monkeypatch.setenv("NLTK_DATA", os.pathsep.join([str(first), str(second)]))
    assert preprocess._nltk_download_dir() == str(first)
    assert first.is_dir()
    assert not second.exists()


def test_find_corpus_falls_back_to_the_zip_layout(monkeypatch):
    seen: list[str] = []

    def find(resource: str):
        seen.append(resource)
        if resource.endswith(".zip/wordnet/"):
            return "zipped"
        raise LookupError(resource)

    monkeypatch.setattr(nltk.data, "find", find)
    assert preprocess.find_corpus("wordnet") == "zipped"
    assert seen == ["corpora/wordnet", "corpora/wordnet.zip/wordnet/"]


def test_find_corpus_raises_when_absent(monkeypatch):
    monkeypatch.setattr(nltk.data, "find", lambda resource: (_ for _ in ()).throw(LookupError()))
    with pytest.raises(LookupError):
        preprocess.find_corpus("wordnet")


def test_ensure_wordnet_downloads_into_nltk_data(monkeypatch, tmp_path):
    downloaded: list[tuple[str, str | None]] = []
    monkeypatch.setenv("NLTK_DATA", str(tmp_path / "nltk"))

    def find(resource: str):
        if any(name in resource for name, _ in downloaded):
            return "found"
        raise LookupError(resource)

    def download(name: str, download_dir=None, quiet=False, raise_on_error=False, **kwargs):
        assert quiet is True
        downloaded.append((name, download_dir))
        return True

    monkeypatch.setattr(nltk.data, "find", find)
    monkeypatch.setattr(nltk, "download", download)
    assert preprocess.ensure_wordnet() is None
    assert [name for name, _ in downloaded] == ["wordnet", "omw-1.4"]
    assert all(target == str(tmp_path / "nltk") for _, target in downloaded)
    assert (tmp_path / "nltk").is_dir()


def test_ensure_wordnet_raises_when_the_download_fails(monkeypatch):
    monkeypatch.delenv("NLTK_DATA", raising=False)
    monkeypatch.setattr(nltk.data, "find", lambda resource: (_ for _ in ()).throw(LookupError()))
    monkeypatch.setattr(nltk, "download", lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match="wordnet"):
        preprocess.ensure_wordnet()

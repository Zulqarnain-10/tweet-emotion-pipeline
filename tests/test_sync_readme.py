"""Unit tests for tweet_emotion.sync_readme: receipts, rendering, block replacement, --check.

A temporary repository root holds hand-built receipt files whose numbers are obviously
synthetic; nothing here reads the real reports.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tweet_emotion import sync_readme as sr

GIT_SHA = "abc1234" * 5 + "abcde"
DATA_SHA = "0123456789abcdef" * 4
CANDIDATES = {
    "logreg": {
        "cv_roc_auc_mean": 0.8901,
        "cv_roc_auc_std": 0.0123,
        "cv_accuracy_mean": 0.8012,
        "cv_f1_mean": 0.7989,
        "fit_seconds": 1.23,
    },
    "nb": {
        "cv_roc_auc_mean": 0.8765,
        "cv_roc_auc_std": 0.0234,
        "cv_accuracy_mean": 0.7901,
        "cv_f1_mean": 0.7877,
        "fit_seconds": 0.45,
    },
}
METRICS = {
    "model": "logreg",
    "model_version": "0.1.0+abc1234",
    "trained_at": "2026-01-01T00:00:00Z",
    "git_sha": GIT_SHA,
    "data_sha256": DATA_SHA,
    "n_train": 731,
    "n_test": 183,
    "n_features": 4321,
    "positive_label": "happiness",
    "positive_rate_test": 0.5023,
    "threshold": 0.5,
    "accuracy": 0.8123,
    "precision": 0.8234,
    "recall": 0.8345,
    "f1": 0.8289,
    "roc_auc": 0.9012,
    "pr_auc": 0.9134,
    "brier": 0.1357,
    "log_loss": 0.4321,
    "confusion": {"tn": 87, "fp": 4, "fn": 5, "tp": 87},
    "majority_baseline_accuracy": 0.5023,
    "candidates_cv": CANDIDATES,
    "selection_metric": "roc_auc",
}
LOADTEST = {
    "p50_ms": 5.67,
    "p95_ms": 123.45,
    "p99_ms": 234.56,
    "mean_ms": 8.9,
    "rps": 456.7,
    "error_rate": 0.0,
    "requests": 300,
    "concurrency": 10,
    "host": "test host",
    "timestamp": "2026-01-02T00:00:00Z",
    "url": "http://localhost:8000/predict",
}
VERSION = {
    "model": "logreg",
    "package_version": "0.1.0",
    "model_version": "0.1.0+abc1234",
    "git_sha": GIT_SHA,
    "git_sha_short": "abc1234",
    "git_dirty": False,
    "data_sha256": DATA_SHA,
    "trained_at": "2026-01-01T00:00:00Z",
    "n_train": 731,
    "n_features": 4321,
    "selection_metric": "roc_auc",
    "cv_folds": 5,
    "candidates": CANDIDATES,
    "winner_params": {"C": 1.0, "max_iter": 1000},
    "libraries": {"python": "3.12.0"},
}
DOCS = {"metrics": METRICS, "loadtest": LOADTEST, "version": VERSION}
README = "# Title\n\nIntro.\n\n<!-- metrics:start -->\nold block\n<!-- metrics:end -->\n\nOutro.\n"
RENDERED_VALUES = (
    "0.8123",
    "0.8234",
    "0.8345",
    "0.8289",
    "0.9012",
    "0.9134",
    "0.1357",
    "0.5023",
    "0.8901",
    "0.8765",
    "123.45",
    "test host",
    "2026-01-01T00:00:00Z",
    "abc1234",
    "0123456",
    "731",
    "183",
    "logreg",
    "nb",
)
EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")
EM_DASH = "—"


def write_receipts(root: Path, *names: str) -> None:
    for name in names or tuple(DOCS):
        path = root / sr.SOURCES[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DOCS[name]), encoding="utf-8")


def readme_text(root: Path) -> str:
    return (root / "README.md").read_bytes().decode("utf-8")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_bytes(README.encode("utf-8"))
    return tmp_path


# ---------------------------------------------------------------------------
# Receipts and rendering
# ---------------------------------------------------------------------------


def test_sources_name_the_three_receipts():
    assert set(sr.SOURCES) == {"metrics", "loadtest", "version"}
    assert sr.SOURCES["metrics"] == "reports/metrics.json"
    assert sr.SOURCES["loadtest"] == "reports/loadtest.json"
    assert sr.SOURCES["version"] == "models/version.json"


def test_markers_and_note():
    assert sr.START == "<!-- metrics:start -->"
    assert sr.END == "<!-- metrics:end -->"
    assert sr.TODO == "[todo]"
    assert "python -m tweet_emotion.sync_readme" in sr.GENERATED_NOTE
    assert sr.GENERATED_NOTE.startswith("<!--")
    assert sr.GENERATED_NOTE.endswith("-->")


def test_load_receipts_reads_what_exists(tmp_path: Path):
    write_receipts(tmp_path, "metrics", "version")
    (tmp_path / "reports" / "loadtest.json").write_text("[1, 2]", encoding="utf-8")
    loaded = sr.load_receipts(tmp_path)
    assert loaded.metrics == METRICS
    assert loaded.version == VERSION
    assert loaded.loadtest is None
    assert loaded.missing() == ["reports/loadtest.json"]


def test_load_receipts_from_an_empty_root(tmp_path: Path):
    loaded = sr.load_receipts(tmp_path)
    assert loaded.metrics is None
    assert loaded.loadtest is None
    assert loaded.version is None
    assert loaded.missing() == list(sr.SOURCES.values())


def test_receipts_get_walks_nested_keys(tmp_path: Path):
    write_receipts(tmp_path)
    receipts = sr.load_receipts(tmp_path)
    assert receipts.get("metrics", "roc_auc") == 0.9012
    assert receipts.get("metrics", "candidates_cv", "nb", "cv_roc_auc_mean") == 0.8765
    assert receipts.get("metrics", "nope") is None
    assert receipts.get("metrics", "roc_auc", "deeper") is None
    assert receipts.get("loadtest", "host") == "test host"


def test_render_readme_carries_every_receipt(tmp_path: Path):
    write_receipts(tmp_path)
    block = sr.render_readme(sr.load_receipts(tmp_path))
    assert block.startswith(sr.GENERATED_NOTE)
    for value in RENDERED_VALUES:
        assert value in block, value
    assert sr.TODO not in block
    assert sr.START not in block
    assert sr.END not in block


def test_render_readme_without_receipts_renders_todo(tmp_path: Path):
    block = sr.render_readme(sr.load_receipts(tmp_path))
    assert block.startswith(sr.GENERATED_NOTE)
    assert block.count(sr.TODO) >= 10
    assert "0.8123" not in block
    assert "test host" not in block


def test_render_readme_fills_only_what_is_present(tmp_path: Path):
    write_receipts(tmp_path, "metrics")
    block = sr.render_readme(sr.load_receipts(tmp_path))
    assert "0.8123" in block
    assert "test host" not in block
    assert sr.TODO in block


def test_render_readme_follows_the_copy_rules(tmp_path: Path):
    write_receipts(tmp_path)
    for receipts in (sr.load_receipts(tmp_path), sr.load_receipts(tmp_path / "nowhere")):
        block = sr.render_readme(receipts)
        assert EM_DASH not in block
        assert not EMOJI.search(block)
        assert "course" not in block.lower()
        assert "lecture" not in block.lower()
        assert "tutorial" not in block.lower()


def test_render_readme_lists_the_candidates_table(tmp_path: Path):
    write_receipts(tmp_path)
    block = sr.render_readme(sr.load_receipts(tmp_path))
    rows = [line for line in block.splitlines() if line.startswith("|")]
    logreg_line = next(line for line in rows if "logreg" in line)
    nb_line = next(line for line in rows if "0.8765" in line)
    assert "0.8901" in logreg_line
    assert "nb" in nb_line
    assert logreg_line != nb_line


def test_formatters_render_numbers_and_todo():
    names = ("f4", "f2", "fint", "ftext", "fshort")
    if not all(hasattr(sr, name) for name in names):
        pytest.skip("the formatting helpers are not exposed under the FLAGSHIP names")
    assert sr.f4(0.79094) == "0.7909"
    assert sr.f4(1) == "1.0000"
    assert sr.f2(123.456) == "123.46"
    assert sr.fint(18000) == "18,000"
    assert sr.fint(6000.0) == "6,000"
    assert sr.ftext("logreg") == "logreg"
    assert sr.fshort("abcdef0123456789", 7) == "abcdef0"
    for fn in (sr.f4, sr.f2, sr.fint, sr.ftext, sr.fshort):
        assert fn(None) == sr.TODO
    for fn in (sr.f4, sr.f2, sr.fint):
        assert fn(True) == sr.TODO
        assert fn("12") == sr.TODO
    assert sr.ftext("") == sr.TODO


# ---------------------------------------------------------------------------
# Block replacement
# ---------------------------------------------------------------------------


def test_replace_block_keeps_markers_on_their_own_lines():
    out = sr.replace_block(README, "new block")
    assert out == README.replace("old block", "new block")
    assert out.count(sr.START) == 1
    assert out.count(sr.END) == 1
    assert sr.replace_block(out, "new block") == out


@pytest.mark.parametrize(
    "text",
    ["no markers", README + sr.START, README + sr.END, f"{sr.END}\n{sr.START}\n"],
    ids=["none", "two_starts", "two_ends", "reversed"],
)
def test_replace_block_rejects_malformed_markers(text):
    with pytest.raises(ValueError):
        sr.replace_block(text, "x")


# ---------------------------------------------------------------------------
# sync and main
# ---------------------------------------------------------------------------


def test_sync_updates_then_is_idempotent(root: Path):
    write_receipts(root)
    assert sr.sync(root, check=True) == 1
    assert readme_text(root) == README

    assert sr.sync(root, check=False) == 0
    first = (root / "README.md").read_bytes()
    assert b"\r\n" not in first
    updated = first.decode("utf-8")
    assert updated.startswith("# Title\n\nIntro.\n\n<!-- metrics:start -->\n" + sr.GENERATED_NOTE)
    assert updated.endswith("<!-- metrics:end -->\n\nOutro.\n")
    assert "old block" not in updated
    for value in RENDERED_VALUES:
        assert value in updated, value
    assert sr.TODO not in updated

    assert sr.sync(root, check=True) == 0
    assert sr.sync(root, check=False) == 0
    assert (root / "README.md").read_bytes() == first


def test_sync_renders_todo_when_receipts_are_missing(root: Path, capsys):
    assert sr.sync(root, check=False) == 0
    updated = readme_text(root)
    assert sr.TODO in updated
    assert "old block" not in updated
    assert sr.GENERATED_NOTE in updated
    out = capsys.readouterr().out
    for path in sr.SOURCES.values():
        assert path in out


def test_sync_fills_only_what_is_present(root: Path):
    write_receipts(root, "metrics", "version")
    assert sr.sync(root, check=False) == 0
    updated = readme_text(root)
    assert "0.8123" in updated
    assert "abc1234" in updated
    assert "test host" not in updated
    assert sr.TODO in updated


def test_sync_normalises_crlf_readmes(root: Path):
    (root / "README.md").write_bytes(README.replace("\n", "\r\n").encode("utf-8"))
    write_receipts(root)
    assert sr.sync(root, check=False) == 0
    assert b"\r\n" not in (root / "README.md").read_bytes()


def test_sync_fails_on_a_readme_without_markers(root: Path, capsys):
    (root / "README.md").write_bytes(b"# No markers\n")
    write_receipts(root)
    assert sr.sync(root, check=False) == 2
    assert "error" in capsys.readouterr().err
    assert readme_text(root) == "# No markers\n"


def test_sync_skips_when_the_readme_is_absent(tmp_path: Path):
    write_receipts(tmp_path)
    assert sr.sync(tmp_path, check=True) == 0
    assert not (tmp_path / "README.md").exists()


def test_main_check_reports_a_stale_block(root: Path, capsys):
    write_receipts(root)
    assert sr.main(["--root", str(root), "--check"]) == 1
    assert "would change" in capsys.readouterr().err
    assert readme_text(root) == README
    assert sr.main(["--root", str(root)]) == 0
    assert sr.main(["--root", str(root), "--check"]) == 0
    assert "0.8123" in readme_text(root)


def test_main_check_passes_on_a_fresh_block(root: Path):
    write_receipts(root)
    assert sr.main(["--root", str(root)]) == 0
    before = (root / "README.md").read_bytes()
    assert sr.main(["--root", str(root), "--check"]) == 0
    assert (root / "README.md").read_bytes() == before

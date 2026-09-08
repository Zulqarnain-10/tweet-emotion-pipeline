"""Download the tweet CSV, verify its sha256, keep two labels, drop duplicate texts, split.

Usage:
    python -m tweet_emotion.ingest

The source file is fetched from params.data.url and checked against params.data.sha256 before
anything else runs; a mismatch is a hard failure. Rows labelled with the two kept emotions are
renamed to the contract columns (tweet_id, text, label), texts that repeat an earlier row after
HTML unescaping, case folding, and whitespace collapsing are dropped so nothing leaks between
the splits, and a stratified split is written as data/raw/train.csv and data/raw/test.csv.
data/raw/fetch_manifest.json records every count.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from tweet_emotion import __version__, settings

log = logging.getLogger(__name__)

USER_AGENT = (
    f"tweet-emotion-pipeline/{__version__} "
    "(+https://github.com/Zulqarnain-10/tweet-emotion-pipeline)"
)
DOWNLOAD_TIMEOUT_SECONDS = 60
CHUNK_BYTES = 1 << 20


# ---------------------------------------------------------------------------
# Fetch and verify
# ---------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    """Hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    url: str,
    dest: Path,
    expected_sha256: str | None = None,
    retries: int = 3,
    backoff: float = 2.0,
) -> Path:
    """Stream url to dest.tmp, then move it into place, retrying with exponential backoff.

    When dest already exists and expected_sha256 matches it, nothing is downloaded.
    """
    if expected_sha256 and dest.exists() and sha256_of(dest) == expected_sha256.lower():
        log.info("using cached %s (sha256 matches)", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _stream(url, tmp)
            os.replace(tmp, dest)
            log.info("downloaded %s (%d bytes)", url, dest.stat().st_size)
            return dest
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            log.warning("download attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff**attempt)
    raise RuntimeError(f"could not download {url} after {retries} attempts") from last_error


def _stream(url: str, tmp: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with (
        urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
        open(tmp, "wb") as out,
    ):
        while True:
            chunk = response.read(CHUNK_BYTES)
            if not chunk:
                break
            out.write(chunk)


def acquire_source(cfg: dict, dest: Path) -> tuple[Path, str]:
    """Put the source CSV at dest and say where it came from.

    The repository keeps a byte-identical copy at data.local_source, so a fresh clone rebuilds
    without network access; that copy is used whenever it exists and its sha256 matches. Otherwise
    the file is downloaded from data.url. Either way verify_sha256 runs on dest afterwards.
    Returns (dest, "repository copy" | "download").
    """
    expected = str(cfg["sha256"]).lower()
    local = cfg.get("local_source")
    if local:
        local_path = settings.REPO_ROOT / str(local)
        if local_path.is_file() and sha256_of(local_path) == expected:
            if dest.exists() and sha256_of(dest) == expected:
                log.info("using cached %s (sha256 matches)", dest)
                return dest, "repository copy"
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".tmp")
            shutil.copyfile(local_path, tmp)
            os.replace(tmp, dest)
            log.info("copied %s to %s (sha256 matches)", local_path, dest)
            return dest, "repository copy"
        if local_path.is_file():
            log.warning("%s does not match the sha256 in params.yaml; downloading", local_path)
    return download(cfg["url"], dest, expected), "download"


def verify_sha256(path: Path, expected: str) -> str:
    """Return the file's sha256 or raise ValueError carrying both hashes."""
    actual = sha256_of(path)
    if actual != expected.lower():
        raise ValueError(
            f"sha256 mismatch for {path.name}: expected {expected}, got {actual}. "
            "The upstream file changed or the download is corrupt."
        )
    log.info("sha256 verified for %s: %s", path.name, actual)
    return actual


# ---------------------------------------------------------------------------
# Select and split
# ---------------------------------------------------------------------------


def _data_block(params: dict) -> dict:
    """Accept either the whole params dict or its data block."""
    if isinstance(params.get("data"), dict):
        return params["data"]
    return params


def duplicate_text_key(value) -> str:
    """The key two texts must share to count as the same tweet.

    HTML entities are unescaped, case is folded, and runs of whitespace collapse to one space,
    so "Happy  Day &amp; night" and "happy day & night" are one tweet. Only the comparison uses
    the key; the text written out is the original.
    """
    return " ".join(html.unescape(_clean_cell(value)).casefold().split())


def count_duplicate_texts(df: pd.DataFrame, text_col: str) -> int:
    """Rows whose text repeats an earlier row under duplicate_text_key.

    Empty texts do not count: they are dropped for being empty, not for being duplicates.
    """
    keys = df[text_col].map(duplicate_text_key)
    return int((keys.duplicated(keep="first") & (keys != "")).sum())


def select_binary(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Keep the two task labels, map them to 0/1, drop duplicate and empty texts.

    Returns a frame with exactly settings.CSV_COLUMNS: tweet_id, text, label. label is 1 when
    the sentiment equals params positive_label, else 0. Duplicates are judged with
    duplicate_text_key (HTML unescaped, case folded, whitespace collapsed); the first occurrence
    is kept with its original text.
    """
    cfg = _data_block(params)
    text_col, label_col, id_col = cfg["text_column"], cfg["label_column"], cfg["id_column"]
    keep = [str(x) for x in cfg["keep_labels"]]
    positive = str(cfg["positive_label"])
    if positive not in keep:
        raise ValueError(f"positive_label {positive!r} is not in keep_labels {keep}")

    kept = df.loc[df[label_col].astype(str).isin(keep)]
    out = pd.DataFrame(
        {
            settings.ID_COLUMN: kept[id_col].to_numpy(),
            settings.TEXT_COLUMN: kept[text_col].map(_clean_cell).to_numpy(),
            settings.LABEL_COLUMN: (kept[label_col].astype(str) == positive).astype(int).to_numpy(),
        }
    )
    if bool(cfg.get("drop_duplicates", True)):
        keys = out[settings.TEXT_COLUMN].map(duplicate_text_key)
        out = out.loc[~keys.duplicated(keep="first")]
    out = out.loc[out[settings.TEXT_COLUMN] != ""]
    return out.reset_index(drop=True)


def _clean_cell(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


def split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split on label; both halves are sorted by tweet_id so the CSVs are byte-stable."""
    train, test = train_test_split(
        df,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=df[settings.LABEL_COLUMN],
    )
    train = train.sort_values(settings.ID_COLUMN, kind="stable").reset_index(drop=True)
    test = test.sort_values(settings.ID_COLUMN, kind="stable").reset_index(drop=True)
    return train, test


def write_split(df: pd.DataFrame, path: Path) -> None:
    """Write the three contract columns, atomically, with LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.loc[:, list(settings.CSV_COLUMNS)].to_csv(
        tmp, index=False, lineterminator="\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def label_counts(df: pd.DataFrame, label_col: str) -> dict[str, int]:
    """Label counts of the full file, largest first, ties alphabetical."""
    counts = df[label_col].astype(str).value_counts()
    ordered = sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))
    return {label: int(n) for label, n in ordered}


def build_manifest(
    *,
    source_url: str,
    sha256: str,
    fetched_from: str = "download",
    n_bytes: int,
    raw: pd.DataFrame,
    params: dict,
    n_kept_before_dedup: int,
    selected: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    cfg = _data_block(params)
    return {
        "source_url": source_url,
        "fetched_from": fetched_from,
        "sha256": sha256,
        "bytes": int(n_bytes),
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "n_rows_total": len(raw),
        "n_duplicate_texts_total": count_duplicate_texts(raw, cfg["text_column"]),
        "label_counts_total": label_counts(raw, cfg["label_column"]),
        "keep_labels": [str(x) for x in cfg["keep_labels"]],
        "positive_label": str(cfg["positive_label"]),
        "n_kept_before_dedup": int(n_kept_before_dedup),
        "n_duplicates_dropped": int(n_kept_before_dedup - len(selected)),
        "n_kept": len(selected),
        "test_size": float(cfg["test_size"]),
        "seed": int(cfg["seed"]),
        "n_train": len(train),
        "n_test": len(test),
        "positive_rate_train": round(float(train[settings.LABEL_COLUMN].mean()), 4),
        "positive_rate_test": round(float(test[settings.LABEL_COLUMN].mean()), 4),
    }


def read_source(path: Path, params: dict) -> pd.DataFrame:
    """Read the raw CSV with every column as text so NA-looking tweets survive."""
    cfg = _data_block(params)
    df = pd.read_csv(
        path,
        dtype={cfg["text_column"]: str, cfg["label_column"]: str},
        keep_default_na=False,
        encoding="utf-8",
    )
    expected = [cfg["id_column"], cfg["label_column"], cfg["text_column"]]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"source CSV is missing columns {missing}; got {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        params = settings.load_params()
        cfg = params["data"]
        source, fetched_from = acquire_source(cfg, settings.RAW_SOURCE_PATH)
        sha = verify_sha256(source, cfg["sha256"])
        raw = read_source(source, cfg)
        log.info(
            "read %d rows, %d labels, %d texts that repeat an earlier row",
            len(raw),
            raw[cfg["label_column"]].nunique(),
            count_duplicate_texts(raw, cfg["text_column"]),
        )

        n_before_dedup = int(raw[cfg["label_column"]].astype(str).isin(cfg["keep_labels"]).sum())
        selected = select_binary(raw, cfg)
        log.info(
            "kept %d rows for %s (%d duplicate or empty texts dropped)",
            len(selected),
            cfg["keep_labels"],
            n_before_dedup - len(selected),
        )
        train, test = split(selected, cfg["test_size"], cfg["seed"])
        write_split(train, settings.RAW_TRAIN_PATH)
        write_split(test, settings.RAW_TEST_PATH)
        log.info(
            "wrote %s (%d rows) and %s (%d rows)",
            settings.RAW_TRAIN_PATH,
            len(train),
            settings.RAW_TEST_PATH,
            len(test),
        )

        manifest = build_manifest(
            source_url=cfg["url"],
            fetched_from=fetched_from,
            sha256=sha,
            n_bytes=source.stat().st_size,
            raw=raw,
            params=cfg,
            n_kept_before_dedup=n_before_dedup,
            selected=selected,
            train=train,
            test=test,
        )
        tmp = settings.FETCH_MANIFEST_PATH.with_name(settings.FETCH_MANIFEST_PATH.name + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, settings.FETCH_MANIFEST_PATH)
        log.info("wrote %s", settings.FETCH_MANIFEST_PATH)
    except Exception as exc:
        log.error("ingest failed: %s: %s", type(exc).__name__, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

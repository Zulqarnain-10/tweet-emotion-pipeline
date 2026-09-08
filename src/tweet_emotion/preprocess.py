"""Normalise tweet text: case, URLs, mentions, entities, numbers, punctuation, stop words, lemmas.

Usage:
    python -m tweet_emotion.preprocess

This is the one place text normalisation lives. The preprocess stage applies TextNormalizer to
the raw splits, the train stage puts the same class at the front of the saved Pipeline, and the
API therefore normalises a request exactly the way the training data was normalised. The stop
words are read once from configs/stopwords_en.txt, minus params keep_words, and stored on the
instance, so the pickled Pipeline needs neither params.yaml nor that file at serving time.
"""

from __future__ import annotations

import html
import logging
import math
import os
import re
import string
import sys
from collections.abc import Iterable
from pathlib import Path

import nltk
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from tweet_emotion import settings

log = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
DIGIT_RE = re.compile(r"\d+")
# string.punctuation plus the curly single and double quotes and the ellipsis tweets carry.
CURLY_QUOTES_AND_ELLIPSIS = "".join(chr(c) for c in (0x2018, 0x2019, 0x201C, 0x201D, 0x2026))
PUNCTUATION_CHARS = string.punctuation + CURLY_QUOTES_AND_ELLIPSIS
_PUNCTUATION_TABLE = str.maketrans(dict.fromkeys(PUNCTUATION_CHARS, " "))

# NLTK packages the lemmatiser needs, in download order.
WORDNET_PACKAGES: tuple[str, ...] = ("wordnet", "omw-1.4")


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


def load_stopwords(path: Path = settings.STOPWORDS_PATH) -> frozenset[str]:
    """One lower-cased stop word per line; blank lines are ignored."""
    with open(path, encoding="utf-8") as fh:
        words = (line.strip().lower() for line in fh)
        return frozenset(word for word in words if word)


def _nltk_download_dir() -> str | None:
    """First NLTK_DATA entry when set (created if needed), else NLTK's default location."""
    raw = os.environ.get("NLTK_DATA", "").strip()
    if not raw:
        return None
    first = raw.split(os.pathsep)[0].strip()
    if not first:
        return None
    Path(first).mkdir(parents=True, exist_ok=True)
    return first


def find_corpus(package: str):
    """Path pointer to an NLTK corpus package, unzipped or still zipped. Raises LookupError."""
    try:
        return nltk.data.find(f"corpora/{package}")
    except LookupError:
        # nltk 3.10 leaves wordnet as corpora/wordnet.zip and no longer tries the zip itself.
        return nltk.data.find(f"corpora/{package}.zip/{package}/")


def ensure_wordnet() -> None:
    """Make the WordNet corpus available, downloading it quietly on first use.

    Looks for the wordnet package on nltk.data.path (which honours NLTK_DATA). When it is
    missing, downloads wordnet and omw-1.4 into the first NLTK_DATA directory, or NLTK's
    default path.
    """
    try:
        find_corpus("wordnet")
        return
    except LookupError:
        pass
    download_dir = _nltk_download_dir()
    log.info("wordnet not found, downloading %s", ", ".join(WORDNET_PACKAGES))
    for package in WORDNET_PACKAGES:
        ok = nltk.download(package, download_dir=download_dir, quiet=True, raise_on_error=True)
        if not ok:
            raise RuntimeError(f"nltk could not download {package!r}")
    find_corpus("wordnet")


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------


def _as_text(value) -> str:
    """Cast any cell to str; None and NaN become the empty string."""
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _keep_words_tuple(words: Iterable[str] | None) -> tuple[str, ...] | None:
    """keep_words as a lower-cased, de-duplicated, sorted tuple; None stays None.

    An input that already has that form is returned as the same object, which keeps
    sklearn.base.clone happy: it checks that the constructor stores each parameter unchanged.
    """
    if words is None:
        return None
    if isinstance(words, str):
        raise TypeError("keep_words expects an iterable of strings, not a single string")
    normalised = tuple(sorted({str(word).lower() for word in words}))
    return words if isinstance(words, tuple) and words == normalised else normalised


def _without(stopwords: frozenset[str], keep_words: tuple[str, ...] | None) -> frozenset[str]:
    """stopwords minus keep_words; the same object when nothing would change (see above)."""
    if keep_words and not stopwords.isdisjoint(keep_words):
        return stopwords.difference(keep_words)
    return stopwords


class TextNormalizer(BaseEstimator, TransformerMixin):
    """Stateless sklearn transformer that maps raw tweet strings to normalised token strings.

    Steps, in order, each behind its own flag: lower-case; drop URLs; drop @mentions; unescape
    HTML entities; drop digits; replace punctuation with spaces; split on whitespace; drop stop
    words; lemmatise (WordNet, noun default); join with single spaces. The result may be empty,
    never NaN. fit is a no-op that returns self.

    The stop-word set lives on the instance so a pickled Pipeline is self-contained; keep_words
    are taken out of it whichever way the set was supplied, so TextNormalizer(keep_words=...) and
    from_params agree. The WordNet lemmatiser is created lazily and left out of the pickle;
    ensure_wordnet runs before first use.
    """

    def __init__(
        self,
        lowercase: bool = True,
        strip_urls: bool = True,
        strip_mentions: bool = True,
        strip_numbers: bool = True,
        strip_punctuation: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        stopwords: frozenset[str] | None = None,
        keep_words: Iterable[str] | None = None,
    ) -> None:
        self.lowercase = lowercase
        self.strip_urls = strip_urls
        self.strip_mentions = strip_mentions
        self.strip_numbers = strip_numbers
        self.strip_punctuation = strip_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        # Words the stop list must not remove. NLTK's list contains negations such as "not",
        # and dropping them scores "not happy" like "happy".
        self.keep_words = _keep_words_tuple(keep_words)
        if stopwords is not None:
            stopwords = frozenset(stopwords)
        elif remove_stopwords:
            stopwords = load_stopwords()
        else:
            stopwords = frozenset()
        self.stopwords = _without(stopwords, self.keep_words)

    @classmethod
    def from_params(cls, params: dict) -> TextNormalizer:
        """Build from the params.yaml preprocess block (the whole params dict is accepted too)."""
        cfg = params.get("preprocess", params) if isinstance(params, dict) else params
        remove_stopwords = bool(cfg.get("remove_stopwords", True))
        stopwords: frozenset[str] | None = None
        if remove_stopwords:
            path = settings.STOPWORDS_PATH
            if cfg.get("stopwords_file"):
                path = settings.REPO_ROOT / str(cfg["stopwords_file"])
            stopwords = load_stopwords(path)
        return cls(
            lowercase=bool(cfg.get("lowercase", True)),
            strip_urls=bool(cfg.get("strip_urls", True)),
            strip_mentions=bool(cfg.get("strip_mentions", True)),
            strip_numbers=bool(cfg.get("strip_numbers", True)),
            strip_punctuation=bool(cfg.get("strip_punctuation", True)),
            remove_stopwords=remove_stopwords,
            lemmatize=bool(cfg.get("lemmatize", True)),
            stopwords=stopwords,
            keep_words=cfg.get("keep_words"),
        )

    # -- pickling -----------------------------------------------------------

    def __getstate__(self) -> dict:
        state = dict(super().__getstate__())
        state.pop("_wnl", None)
        # A frozenset pickles in iteration order, which follows the interpreter's string hash
        # seed, so the same stop list would give different bytes on every run and the saved
        # model would change without the model changing. A sorted list pickles identically.
        state["stopwords"] = sorted(state.get("stopwords", ()))
        return state

    def __setstate__(self, state: dict) -> None:
        state = dict(state)
        state["stopwords"] = frozenset(state.get("stopwords", ()))
        state.setdefault("keep_words", None)
        super().__setstate__(state)

    def __sklearn_is_fitted__(self) -> bool:
        return True

    @property
    def _lemmatizer(self):
        """WordNet lemmatiser, created on first use so the pickle carries no corpus state."""
        wnl = getattr(self, "_wnl", None)
        if wnl is None:
            ensure_wordnet()
            from nltk.stem import WordNetLemmatizer

            wnl = WordNetLemmatizer()
            self._wnl = wnl
        return wnl

    # -- sklearn API --------------------------------------------------------

    def fit(self, X, y=None) -> TextNormalizer:
        return self

    def transform(self, X: Iterable) -> list[str]:
        if isinstance(X, str):
            raise TypeError("transform expects an iterable of strings, not a single string")
        return [self.normalize(text) for text in X]

    def normalize(self, text) -> str:
        """Normalise one value. Returns the empty string for None, NaN, or nothing left."""
        text = _as_text(text)
        if self.lowercase:
            text = text.lower()
        if self.strip_urls:
            text = URL_RE.sub(" ", text)
        if self.strip_mentions:
            text = MENTION_RE.sub(" ", text)
        text = html.unescape(text)
        if self.strip_numbers:
            text = DIGIT_RE.sub("", text)
        if self.strip_punctuation:
            text = text.translate(_PUNCTUATION_TABLE)
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        if self.lemmatize and tokens:
            lemmatize = self._lemmatizer.lemmatize
            tokens = [lemmatize(token) for token in tokens]
        return " ".join(tokens).strip()


def normalize_text(text: str, normalizer: TextNormalizer) -> str:
    """Convenience wrapper around TextNormalizer.normalize for one string."""
    return normalizer.normalize(text)


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def read_split(path: Path) -> pd.DataFrame:
    """Read a pipeline CSV keeping empty and NA-looking texts as strings."""
    df = pd.read_csv(
        path, dtype={settings.TEXT_COLUMN: str}, keep_default_na=False, encoding="utf-8"
    )
    missing = [c for c in settings.CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    return df


def write_split(df: pd.DataFrame, path: Path) -> None:
    """Write the three contract columns, atomically, with LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.loc[:, list(settings.CSV_COLUMNS)].to_csv(
        tmp, index=False, lineterminator="\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def process_split(df: pd.DataFrame, normalizer: TextNormalizer) -> pd.DataFrame:
    out = df.loc[:, list(settings.CSV_COLUMNS)].copy()
    out[settings.TEXT_COLUMN] = normalizer.transform(out[settings.TEXT_COLUMN].tolist())
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        params = settings.load_params()
        normalizer = TextNormalizer.from_params(params["preprocess"])
        if normalizer.lemmatize:
            ensure_wordnet()
        for src, dest in (
            (settings.RAW_TRAIN_PATH, settings.PROCESSED_TRAIN_PATH),
            (settings.RAW_TEST_PATH, settings.PROCESSED_TEST_PATH),
        ):
            df = read_split(src)
            out = process_split(df, normalizer)
            empty = int((out[settings.TEXT_COLUMN] == "").sum())
            share = empty / len(out) if len(out) else 0.0
            write_split(out, dest)
            log.info(
                "wrote %s: %d rows, %d empty after normalisation (%.4f)",
                dest,
                len(out),
                empty,
                share,
            )
    except Exception as exc:
        log.error("preprocess failed: %s: %s", type(exc).__name__, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

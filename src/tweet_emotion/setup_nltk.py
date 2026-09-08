"""Fetch the NLTK corpora the lemmatiser needs (wordnet, omw-1.4) if they are missing.

Usage:
    python -m tweet_emotion.setup_nltk

Idempotent: a second run finds the corpora and downloads nothing. Set NLTK_DATA to choose where
they land (the Dockerfile uses /opt/nltk_data); otherwise NLTK's default user path is used.
"""

from __future__ import annotations

import logging
import sys

from tweet_emotion.preprocess import WORDNET_PACKAGES, ensure_wordnet, find_corpus

log = logging.getLogger(__name__)


def locate(package: str) -> str | None:
    """Filesystem location of an NLTK corpus package, or None when it is not installed."""
    try:
        return str(find_corpus(package))
    except LookupError:
        return None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        ensure_wordnet()
    except Exception as exc:
        log.error("setup_nltk failed: %s: %s", type(exc).__name__, exc)
        return 1
    status = 0
    for package in WORDNET_PACKAGES:
        location = locate(package)
        if location is None:
            if package == "wordnet":
                log.error("%s is still missing after download", package)
                status = 1
            else:
                log.warning("%s is missing; lemmatisation works without it", package)
            continue
        print(f"{package}: {location}")
    return status


if __name__ == "__main__":
    sys.exit(main())

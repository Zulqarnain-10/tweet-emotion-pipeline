"""Post-deploy smoke test for a running tweet-emotion-pipeline service.

Standard library only, so it runs on a bare CI runner. Polls /health until it answers 200,
checks /version, and scores the first preset tweet through /predict. Prints one line per check
and exits 1 on any failure.

    python scripts/smoke_live.py --url https://<user>-tweet-emotion-pipeline.hf.space --sha <sha>
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PREDICT_KEYS = ("label", "probability_happiness", "model_version")
LABELS = frozenset({"happiness", "sadness"})


class SmokeError(Exception):
    """A check failed. The message is the detail shown in the table."""


def request(url: str, payload: dict | None = None, timeout: float = 15) -> tuple[int, Any]:
    """GET, or POST JSON when a payload is given. Returns (status, parsed body or raw text)."""
    headers = {"Accept": "application/json", "User-Agent": "tweet-emotion-smoke/1"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    text = raw.decode("utf-8", errors="replace")
    try:
        return status, json.loads(text)
    except ValueError:
        return status, text


def wait_for_health(base: str, timeout: float, interval: float) -> tuple[dict, float]:
    start = time.monotonic()
    last = "no response yet"
    while True:
        try:
            status, body = request(f"{base}/health", timeout=10)
            if status == 200 and isinstance(body, dict) and body.get("status") == "ok":
                return body, time.monotonic() - start
            last = f"HTTP {status}: {str(body)[:80]!r}"
        except (OSError, http.client.HTTPException) as exc:
            last = f"{type(exc).__name__}: {exc}"
        if time.monotonic() - start >= timeout:
            raise SmokeError(f"/health did not answer 200 within {timeout:.0f} s ({last})")
        time.sleep(interval)


def check_version(base: str, sha: str | None) -> tuple[str, str | None]:
    """Return (detail, warning). The sha check warns rather than fails: the model may have
    been trained on an earlier commit than the one being deployed."""
    status, body = request(f"{base}/version")
    if status != 200 or not isinstance(body, dict):
        raise SmokeError(f"/version returned HTTP {status}: {str(body)[:120]!r}")
    model_version = body.get("model_version")
    if not model_version:
        raise SmokeError("/version has no model_version")
    git_sha = str(body.get("git_sha") or "")
    detail = f"model_version={model_version} git_sha={git_sha or 'unknown'}"
    if not sha:
        return detail, None
    if git_sha and (git_sha.startswith(sha) or sha.startswith(git_sha)):
        return f"{detail} (matches deploy sha)", None
    warning = (
        f"version.json sha {git_sha or 'unknown'} differs from deploy sha {sha[:7]}; "
        "the model was trained on an earlier commit"
    )
    return detail, warning


def first_preset(base: str, presets_path: Path | None) -> tuple[dict, str]:
    """The /predict payload for the first preset, from a local file or GET /presets."""
    if presets_path is not None:
        doc = json.loads(presets_path.read_text(encoding="utf-8"))
        source = presets_path.as_posix()
    else:
        status, doc = request(f"{base}/presets")
        if status != 200:
            raise SmokeError(f"/presets returned HTTP {status}")
        source = "/presets"
    presets = doc.get("presets") if isinstance(doc, dict) else None
    first = presets[0] if isinstance(presets, list) and presets else None
    text = first.get("text") if isinstance(first, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise SmokeError(f"no usable preset found in {source}")
    return {"text": text}, f"{first.get('id', 'preset')} from {source}"


def check_predict(base: str, payload: dict) -> str:
    status, body = request(f"{base}/predict", payload=payload)
    if status != 200 or not isinstance(body, dict):
        raise SmokeError(f"/predict returned HTTP {status}: {str(body)[:120]!r}")
    missing = [key for key in PREDICT_KEYS if key not in body]
    if missing:
        raise SmokeError(f"/predict response is missing {', '.join(missing)}")
    probability = body["probability_happiness"]
    if isinstance(probability, bool) or not isinstance(probability, int | float):
        raise SmokeError(f"probability_happiness is not a number: {probability!r}")
    if not 0.0 <= probability <= 1.0:
        raise SmokeError(f"probability_happiness out of range: {probability!r}")
    if body["label"] not in LABELS:
        raise SmokeError(f"unexpected label: {body['label']!r}")
    if not body["model_version"]:
        raise SmokeError("/predict response has an empty model_version")
    return (
        f"label={body['label']} probability_happiness={probability:.4f} "
        f"threshold={body.get('threshold')} model={body.get('model')} "
        f"model_version={body['model_version']}"
    )


def print_table(rows: list[tuple[str, str, str]]) -> None:
    width = max(len(row[0]) for row in rows)
    print(f"{'check'.ljust(width)}  {'status'.ljust(7)}  detail")
    for name, status, detail in rows:
        print(f"{name.ljust(width)}  {status.ljust(7)}  {detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", required=True, help="base URL of the running service")
    parser.add_argument("--sha", help="commit being deployed; compared with /version git_sha")
    parser.add_argument(
        "--timeout", type=float, default=600, help="seconds to wait for /health (default 600)"
    )
    parser.add_argument(
        "--interval", type=float, default=10, help="seconds between /health polls (default 10)"
    )
    parser.add_argument(
        "--presets",
        type=Path,
        help="local presets.json to score; by default the first preset comes from GET /presets",
    )
    args = parser.parse_args(argv)
    base = args.url.rstrip("/")

    rows: list[tuple[str, str, str]] = []
    failed = False
    try:
        health, waited = wait_for_health(base, args.timeout, args.interval)
        rows.append(
            (
                "health",
                "ok",
                f"model_loaded={health.get('model_loaded')} "
                f"model_version={health.get('model_version')} after {waited:.1f} s",
            )
        )
        detail, warning = check_version(base, args.sha)
        rows.append(("version", "ok", detail))
        if warning:
            rows.append(("version", "warn", warning))
        payload, source = first_preset(base, args.presets)
        rows.append(("preset", "ok", source))
        rows.append(("predict", "ok", check_predict(base, payload)))
    except SmokeError as exc:
        rows.append(("failed", "fail", str(exc)))
        failed = True
    except (OSError, http.client.HTTPException, ValueError, KeyError) as exc:
        rows.append(("failed", "fail", f"{type(exc).__name__}: {exc}"))
        failed = True

    print(f"Smoke test against {base}")
    print_table(rows)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

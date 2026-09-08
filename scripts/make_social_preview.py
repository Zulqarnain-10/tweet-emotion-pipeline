"""Render the GitHub social preview card (1280 x 640) from the measured metrics.

The card follows the personal brand spec: navy ground with faint radial tints, the repo name in
Bricolage Grotesque, a Fraunces italic payoff with a coral underline, one mono metric line with
its receipt, and the Z. mark bottom right. Numbers come from reports/metrics.json only; when that
file is missing the line renders [todo] chips instead of failing.

Fonts are optional. The script looks for .woff2, .ttf or .otf files in --font-dir (by default
the personal site fonts folder next to this checkout, when it exists) and matches the brand
faces by file name: Bricolage Grotesque for the display line, Fraunces italic for the payoff,
Plus Jakarta Sans for the support lines, IBM Plex Mono for the kicker and the metric line. It
falls back to DejaVu, then to the font Pillow bundles, and never fails the build over fonts.

Usage:
    python scripts/make_social_preview.py [--out docs/social-preview.png] [--font-dir <dir>]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    PIL_ERROR: str | None = f"{type(exc).__name__}: {exc}"
else:
    PIL_ERROR = None

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "social-preview.png"
DEFAULT_METRICS = REPO_ROOT / "reports" / "metrics.json"
DEFAULT_FONT_DIR = REPO_ROOT.parent / "zulqarnain-site" / "fonts"
FONT_SUFFIXES = (".woff2", ".ttf", ".otf")

# Brand tokens: navy ground, fog and mist text, coral accent, blue for receipts, chip for the mark.
NAVY = (10, 22, 40)
FOG = (234, 240, 249)
MIST = (162, 180, 206)
CORAL = (255, 107, 53)
BLUE = (59, 150, 255)
CHIP = (15, 31, 56)

W, H = 1280, 640
SCALE = 2  # render at 2x, downsample for clean edges
MARGIN = 72
MARK_SIZE = 96
DOT = "·"
TODO = "[todo]"

# Role -> (name fragments that identify the brand file, system fallbacks tried in order).
FONT_ROLES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "display": (("bricolage",), ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf")),
    "payoff": (
        ("fraunces", "italic"),
        ("DejaVuSerif-Italic.ttf", "DejaVuSerif.ttf", "DejaVuSans.ttf"),
    ),
    "body": (("jakarta",), ("DejaVuSans.ttf",)),
    "mono": (("plex", "mono"), ("DejaVuSansMono.ttf",)),
}


def find_brand_fonts(font_dir: Path | None, fragments: tuple[str, ...]) -> list[Path]:
    """Every font file in font_dir whose name contains all the fragments, sorted by name."""
    if font_dir is None or not font_dir.is_dir():
        return []
    return [
        path
        for path in sorted(font_dir.iterdir())
        if path.suffix.lower() in FONT_SUFFIXES and all(f in path.name.lower() for f in fragments)
    ]


def pick_weight(candidates: list[Path], wght: float | None) -> Path | None:
    """The file named for the requested weight in a static family, else the first candidate."""
    if wght is not None:
        tag = f"-{int(wght)}"
        for path in candidates:
            if tag in path.stem.lower():
                return path
    return candidates[0] if candidates else None


def set_axes(font, wght: float | None, opsz: float | None):
    """Select weight and optical size on a variable font. A static font comes back unchanged."""
    if wght is None and opsz is None:
        return font
    try:
        axes = font.get_variation_axes()
    except (OSError, AttributeError):
        return font
    values = []
    for axis in axes:
        raw = axis["name"]
        name = raw.decode() if isinstance(raw, bytes) else str(raw)
        key = name.lower()
        lo, hi = axis.get("minimum", 0), axis.get("maximum", 1000)
        if key == "weight" and wght is not None:
            values.append(min(max(wght, lo), hi))
        elif key in ("optical size", "opsz") and opsz is not None:
            values.append(min(max(opsz, lo), hi))
        else:
            values.append(axis["default"])
    with contextlib.suppress(OSError):
        font.set_variation_by_axes(values)
    return font


class FontSet:
    """Resolves a font per role and size, and records which file served each role."""

    def __init__(self, font_dir: Path | None) -> None:
        self.sources: dict[str, str] = {}
        self._brand: dict[str, list[Path]] = {}
        self._fallbacks: dict[str, tuple[str, ...]] = {}
        for role, (fragments, fallbacks) in FONT_ROLES.items():
            self._brand[role] = find_brand_fonts(font_dir, fragments)
            self._fallbacks[role] = fallbacks

    def _record(self, role: str, name: str) -> None:
        """Remember every file that served a role, in first-use order."""
        used = self.sources.get(role, "")
        if name not in used.split(" + "):
            self.sources[role] = f"{used} + {name}" if used else name

    def load(self, role: str, size: int, wght: float | None = None, opsz: float | None = None):
        px = size * SCALE
        brand = pick_weight(self._brand[role], wght)
        if brand is not None:
            with contextlib.suppress(OSError):
                font = ImageFont.truetype(str(brand), px)
                self._record(role, brand.name)
                return set_axes(font, wght, opsz)
        for name in self._fallbacks[role]:
            try:
                font = ImageFont.truetype(name, px)
            except OSError:
                continue
            self._record(role, name)
            return font
        self._record(role, "Pillow default")
        try:
            return ImageFont.load_default(size=px)
        except TypeError:  # Pillow before 10.1 takes no size
            return ImageFont.load_default()


def fit(draw, fonts: FontSet, role: str, text: str, size: int, max_width: float, **axes):
    """The largest font at or below size whose rendering of text fits inside max_width."""
    while True:
        font = fonts.load(role, size, **axes)
        if draw.textlength(text, font=font) <= max_width or size <= 12:
            return font
        size -= 2


def descender_bottom(font, text: str) -> float:
    """Distance from the drawing origin to the bottom of the descenders, in pixels."""
    try:
        ascent, descent = font.getmetrics()
        return float(ascent + descent)
    except AttributeError:
        return float(font.getbbox(text)[3])


def radial_tint(base, center: tuple[int, int], radius: int, color: tuple, alpha: float) -> None:
    """Paint a soft radial tint onto base (in place)."""
    overlay = Image.new("RGBA", base.size, (*color, 0))
    mask = Image.new("L", base.size, 0)
    draw = ImageDraw.Draw(mask)
    steps = 40
    for i in range(steps, 0, -1):
        r = int(radius * i / steps)
        level = int(255 * alpha * (1 - i / steps) ** 1.6)
        draw.ellipse([center[0] - r, center[1] - r, center[0] + r, center[1] + r], fill=level)
    overlay.putalpha(mask)
    base.alpha_composite(overlay)


def z_mark(draw, x: float, y: float, size: float) -> None:
    """The Z. chip mark at (x, y) with the given side, scaled from the 64-unit brand viewBox.

    Geometry: rounded rect rx 14, the Z path M12 13 H44 V22.5 L25.5 41.5 H44 V51 H12 V41.5
    L30.5 22.5 H12 Z, and a coral dot at (51.5, 45.5) with radius 5.5.
    """
    s = size / 64
    draw.rounded_rectangle([x, y, x + size, y + size], radius=int(14 * s), fill=CHIP)
    pts = [
        (12, 13),
        (44, 13),
        (44, 22.5),
        (25.5, 41.5),
        (44, 41.5),
        (44, 51),
        (12, 51),
        (12, 41.5),
        (30.5, 22.5),
        (12, 22.5),
    ]
    draw.polygon([(x + px * s, y + py * s) for px, py in pts], fill=FOG)
    cx, cy, r = x + 51.5 * s, y + 45.5 * s, 5.5 * s
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=CORAL)


def load_metrics(path: Path) -> dict | None:
    """reports/metrics.json as a dict, or None (with a warning) when missing or unreadable."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"warning: {path.as_posix()} not readable ({exc}); rendering {TODO}", file=sys.stderr)
        return None
    if not isinstance(doc, dict):
        print(f"warning: {path.as_posix()} is not a JSON object; rendering {TODO}", file=sys.stderr)
        return None
    return doc


def metric_line(metrics: dict | None) -> str:
    """One mono line: accuracy, ROC-AUC, held-out size, and the file they come from."""

    def score(key: str) -> str:
        value = metrics.get(key) if metrics else None
        if isinstance(value, bool) or not isinstance(value, int | float):
            return TODO
        return f"{value:.4f}"

    n_test = metrics.get("n_test") if metrics else None
    n = f"{n_test:,}" if isinstance(n_test, int) and not isinstance(n_test, bool) else TODO
    return (
        f"accuracy {score('accuracy')} {DOT} roc-auc {score('roc_auc')} {DOT} "
        f"held-out n={n} {DOT} reports/metrics.json"
    )


def render(metrics: dict | None, fonts: FontSet):
    img = Image.new("RGBA", (W * SCALE, H * SCALE), (*NAVY, 255))
    radial_tint(img, (W * SCALE, 0), 620 * SCALE, BLUE, 0.16)
    radial_tint(img, (0, H * SCALE), 560 * SCALE, CORAL, 0.12)
    draw = ImageDraw.Draw(img)

    margin = MARGIN * SCALE
    content_width = (W - 2 * MARGIN) * SCALE

    # Kicker: tracked uppercase mono.
    kicker_font = fonts.load("mono", 22, wght=500)
    kicker = f"TWEET-EMOTION-PIPELINE  {DOT}  GITHUB.COM/ZULQARNAIN-10"
    x, y = float(margin), float(margin)
    for ch in kicker:
        draw.text((x, y), ch, font=kicker_font, fill=MIST)
        x += draw.textlength(ch, font=kicker_font) + 2.2 * SCALE

    # Headline: display line, then the italic payoff with a coral underline.
    headline = "Tweet emotion detection,"
    payoff = "shipped as a pipeline."
    y = 168 * SCALE
    head_font = fit(draw, fonts, "display", headline, 86, content_width, wght=800, opsz=96)
    draw.text((margin, y), headline, font=head_font, fill=FOG)
    y2 = y + 100 * SCALE
    payoff_font = fit(draw, fonts, "payoff", payoff, 86, content_width, wght=550, opsz=96)
    draw.text((margin, y2), payoff, font=payoff_font, fill=FOG)
    pw = draw.textlength(payoff, font=payoff_font)
    # The underline sits below the descenders, like the site's gradient underline.
    uy = y2 + descender_bottom(payoff_font, payoff) + 4 * SCALE
    draw.rectangle([margin, uy, margin + pw, uy + 5 * SCALE], fill=CORAL)

    # Support lines.
    support_font = fonts.load("body", 28, wght=450)
    support = [
        "Hashed data, six DVC stages, cross-validated model selection,",
        "a tested FastAPI service, CI that reproduces the numbers, a live demo.",
    ]
    for i, line in enumerate(support):
        draw.text((margin, (408 + 40 * i) * SCALE), line, font=support_font, fill=MIST)

    # Metric line, bottom left, kept clear of the mark.
    line = metric_line(metrics)
    max_width = (W - 2 * MARGIN - MARK_SIZE - 32) * SCALE
    receipt_font = fit(draw, fonts, "mono", line, 22, max_width)
    draw.text((margin, (H - MARGIN - 28) * SCALE), line, font=receipt_font, fill=BLUE)

    # Z. mark, bottom right.
    size = MARK_SIZE * SCALE
    z_mark(draw, W * SCALE - margin - size, H * SCALE - margin - size, size)
    return img


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="PNG to write (default docs/social-preview.png)",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS,
        help="metrics file to read (default reports/metrics.json)",
    )
    parser.add_argument(
        "--font-dir",
        type=Path,
        default=DEFAULT_FONT_DIR,
        help="folder holding .woff2, .ttf or .otf brand fonts (default: ../zulqarnain-site/fonts)",
    )
    args = parser.parse_args(argv)

    if PIL_ERROR:
        print(
            f"error: Pillow is not importable ({PIL_ERROR}); install it with: "
            "python -m pip install pillow",
            file=sys.stderr,
        )
        return 2

    metrics = load_metrics(args.metrics)
    fonts = FontSet(args.font_dir)
    img = render(metrics, fonts)

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").resize((W, H), Image.Resampling.LANCZOS).save(out, optimize=True)
    used = ", ".join(f"{role}={name}" for role, name in fonts.sources.items())
    print(f"wrote {out.as_posix()} ({out.stat().st_size // 1024} KB); fonts: {used}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

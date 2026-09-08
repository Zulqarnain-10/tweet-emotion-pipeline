"""Score the held-out test split once; write metrics, top terms, and figures.

Usage:
    python -m tweet_emotion.evaluate

The classifier step scores the saved test matrix, and the full Pipeline scores the raw test
texts; the two probability vectors must agree to 1e-6 or the stage fails. That check is what
proves the API scores a request exactly the way the model was evaluated. Every headline number
in reports/metrics.json comes from this one pass over the test split.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from tweet_emotion import settings
from tweet_emotion.features import feature_names, load_npz

log = logging.getLogger(__name__)

# Brand colours for the figures.
CREAM, INK, BLUE_INK, CORAL_INK, MUTED = "#FBF8F2", "#13233B", "#1560D6", "#D9480F", "#5B6B80"
FIGURE_SIZE_INCHES = (8.0, 16.0 / 3.0)  # 1200 x 800 px at dpi 150
FIGURE_DPI = 150
FIGURE_NAMES: tuple[str, ...] = ("roc", "pr", "confusion", "top_terms")
PARITY_TOLERANCE = 1e-6


def _r(value, digits: int = 4) -> float:
    return round(float(value), digits)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def confusion_at(y, p, threshold: float) -> dict[str, int]:
    """tn, fp, fn, tp when predicting happiness for p >= threshold."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(p, dtype=float) >= float(threshold)
    return {
        "tn": int((~pred & (y == 0)).sum()),
        "fp": int((pred & (y == 0)).sum()),
        "fn": int((~pred & (y == 1)).sum()),
        "tp": int((pred & (y == 1)).sum()),
    }


def majority_baseline_accuracy(y) -> float:
    """Accuracy of always predicting the more common class."""
    y = np.asarray(y).astype(int)
    if len(y) == 0:
        return 0.0
    rate = float(np.mean(y))
    return _r(max(rate, 1.0 - rate))


def compute_metrics(y, p, threshold: float) -> dict:
    """Threshold metrics, ranking metrics, and the confusion block, all rounded to 4 dp."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= float(threshold)).astype(int)
    return {
        "threshold": _r(threshold),
        "accuracy": _r(accuracy_score(y, pred)),
        "precision": _r(precision_score(y, pred, zero_division=0)),
        "recall": _r(recall_score(y, pred, zero_division=0)),
        "f1": _r(f1_score(y, pred, zero_division=0)),
        "roc_auc": _r(roc_auc_score(y, p)),
        "pr_auc": _r(average_precision_score(y, p)),
        "brier": _r(brier_score_loss(y, p)),
        "log_loss": _r(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "confusion": confusion_at(y, p, threshold),
        "majority_baseline_accuracy": majority_baseline_accuracy(y),
    }


def parity_check(pipeline: Pipeline, texts, p_matrix: np.ndarray) -> float:
    """Max abs difference between the Pipeline on raw text and the classifier on the matrix."""
    p_full = pipeline.predict_proba(list(texts))[:, 1]
    p_matrix = np.asarray(p_matrix, dtype=float)
    if p_full.shape != p_matrix.shape:
        raise RuntimeError(f"parity shapes differ: {p_full.shape} vs {p_matrix.shape}")
    diff = float(np.max(np.abs(p_full - p_matrix))) if len(p_matrix) else 0.0
    if diff >= PARITY_TOLERANCE:
        raise RuntimeError(
            f"train/serve parity failed: max abs probability difference {diff:.3e} "
            f"(tolerance {PARITY_TOLERANCE:.0e})"
        )
    return diff


# ---------------------------------------------------------------------------
# Top terms
# ---------------------------------------------------------------------------


def _ranked(weights: np.ndarray, names: list[str], n: int, sign: int) -> list[dict]:
    """Top n terms by signed weight (sign=+1 largest, sign=-1 most negative), magnitudes 4 dp."""
    order = np.argsort(-sign * weights, kind="stable")[: max(int(n), 0)]
    return [
        {"term": names[i], "weight": _r(abs(weights[i]))} for i in order if sign * weights[i] > 0
    ]


def top_terms(clf, names: list[str], n: int) -> dict:
    """Terms that move the score, from the classifier's own weights.

    Linear models (coef_, or naive Bayes log-probability ratios) give one list per class with
    sadness weights reported as positive magnitudes. Tree models give a single gain list and
    available=false, since gain has no direction.
    """
    names = list(names)
    if hasattr(clf, "coef_"):
        weights = np.asarray(clf.coef_, dtype=float).ravel()
        method = "logistic regression coefficients"
    elif hasattr(clf, "feature_log_prob_"):
        logp = np.asarray(clf.feature_log_prob_, dtype=float)
        weights = logp[1] - logp[0]
        method = "log-probability ratio"
    elif hasattr(clf, "feature_importances_"):
        importances = np.asarray(clf.feature_importances_, dtype=float).ravel()
        return {
            "available": False,
            "method": "gain importance",
            "terms": _ranked(importances, names, n, +1),
        }
    else:
        return {"available": False, "method": "none", "terms": []}
    if len(weights) != len(names):
        raise RuntimeError(f"{len(weights)} weights for {len(names)} feature names")
    return {
        "available": True,
        "method": method,
        settings.POSITIVE_LABEL: _ranked(weights, names, n, +1),
        settings.NEGATIVE_LABEL: _ranked(weights, names, n, -1),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    return plt


def _figure(plt):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES)
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)
    return fig, ax


def _style(ax, title: str, xlabel: str, ylabel: str, grid: bool = True) -> None:
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    if grid:
        ax.grid(color=MUTED, alpha=0.18, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_alpha(0.5)


def _save(plt, fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=FIGURE_DPI, facecolor=CREAM)
    plt.close(fig)
    return path


def plot_roc(y, p, model_name: str, auc: float, threshold: float, out_dir: Path) -> Path:
    plt = _pyplot()
    fig, ax = _figure(plt)
    fpr, tpr, _ = roc_curve(y, p)
    point = confusion_at(y, p, threshold)
    fpr_t = point["fp"] / max(point["fp"] + point["tn"], 1)
    tpr_t = point["tp"] / max(point["tp"] + point["fn"], 1)
    ax.plot(fpr, tpr, color=BLUE_INK, lw=2.2, label=f"{model_name} (AUC {auc:.4f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="chance")
    ax.scatter(
        [fpr_t], [tpr_t], color=CORAL_INK, s=60, zorder=5, label=f"threshold {threshold:.2f}"
    )
    _style(ax, "ROC curve, held-out test split", "False positive rate", "True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return _save(plt, fig, "roc", out_dir)


def plot_pr(y, p, model_name: str, ap: float, threshold: float, out_dir: Path) -> Path:
    plt = _pyplot()
    fig, ax = _figure(plt)
    precision, recall, _ = precision_recall_curve(y, p)
    point = confusion_at(y, p, threshold)
    precision_t = point["tp"] / max(point["tp"] + point["fp"], 1)
    recall_t = point["tp"] / max(point["tp"] + point["fn"], 1)
    base_rate = float(np.mean(np.asarray(y).astype(int)))
    ax.plot(recall, precision, color=BLUE_INK, lw=2.2, label=f"{model_name} (AP {ap:.4f})")
    ax.axhline(base_rate, color=MUTED, lw=1, ls="--", label=f"positive rate ({base_rate:.4f})")
    ax.scatter(
        [recall_t],
        [precision_t],
        color=CORAL_INK,
        s=60,
        zorder=5,
        label=f"threshold {threshold:.2f}",
    )
    _style(ax, "Precision-recall curve, held-out test split", "Recall", "Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    return _save(plt, fig, "pr", out_dir)


def plot_confusion(confusion: dict, threshold: float, out_dir: Path) -> Path:
    plt = _pyplot()
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = _figure(plt)
    counts = np.array(
        [[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]], dtype=float
    )
    total = counts.sum() or 1.0
    cmap = LinearSegmentedColormap.from_list("cream_to_blue", [CREAM, BLUE_INK])
    ax.imshow(counts, cmap=cmap, vmin=0, vmax=counts.max() or 1.0)
    labels = list(settings.LABELS)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    for i in range(2):
        for j in range(2):
            share = counts[i, j] / total
            colour = CREAM if counts[i, j] > 0.55 * counts.max() else INK
            ax.text(
                j,
                i,
                f"{int(counts[i, j]):,}\n{share:.1%}",
                ha="center",
                va="center",
                color=colour,
                fontsize=14,
            )
    _style(
        ax,
        f"Confusion matrix at threshold {threshold:.2f}, held-out test split",
        "Predicted",
        "Actual",
        grid=False,
    )
    ax.tick_params(colors=INK, length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return _save(plt, fig, "confusion", out_dir)


def plot_top_terms(terms: dict, out_dir: Path) -> Path | None:
    """Two panels of horizontal bars; only when the classifier exposes signed weights."""
    if not terms.get("available"):
        return None
    plt = _pyplot()
    fig, (ax_pos, ax_neg) = plt.subplots(1, 2, figsize=FIGURE_SIZE_INCHES)
    fig.patch.set_facecolor(CREAM)
    for ax, label, colour in (
        (ax_pos, settings.POSITIVE_LABEL, CORAL_INK),
        (ax_neg, settings.NEGATIVE_LABEL, BLUE_INK),
    ):
        rows = list(terms.get(label, []))[::-1]
        ax.set_facecolor(CREAM)
        ax.barh([r["term"] for r in rows], [r["weight"] for r in rows], color=colour, height=0.7)
        _style(ax, f"Terms that push toward {label}", "Weight", "")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", colors=INK, length=0)
    fig.suptitle(f"Top terms by {terms.get('method', 'weight')}", color=INK, x=0.02, ha="left")
    fig.tight_layout()
    return _save(plt, fig, "top_terms", out_dir)


def clear_figures(out_dir: Path) -> list[Path]:
    """Remove every PNG in out_dir, creating the directory if needed; returns what went.

    Runs whether or not figures are drawn, so a run with evaluate.figures off leaves no
    figure from an earlier model behind.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted(out_dir.glob("*.png"))
    for path in stale:
        path.unlink()
    return stale


def write_figures(y, p, metrics: dict, terms: dict, out_dir: Path) -> list[Path]:
    """Write every figure, removing stale ones first so the directory mirrors this run."""
    clear_figures(out_dir)
    threshold = float(metrics["threshold"])
    written = [
        plot_roc(y, p, metrics["model"], metrics["roc_auc"], threshold, out_dir),
        plot_pr(y, p, metrics["model"], metrics["pr_auc"], threshold, out_dir),
        plot_confusion(metrics["confusion"], threshold, out_dir),
    ]
    top = plot_top_terms(terms, out_dir)
    if top is not None:
        written.append(top)
    return written


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def read_raw_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, dtype={settings.TEXT_COLUMN: str}, keep_default_na=False, encoding="utf-8"
    )
    missing = [c for c in settings.CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    return df


def build_metrics(version: dict, y_test, p_test, threshold: float, n_features: int) -> dict:
    scored = compute_metrics(y_test, p_test, threshold)
    return {
        "model": version["model"],
        "model_version": version["model_version"],
        "trained_at": version["trained_at"],
        "git_sha": version["git_sha"],
        "data_sha256": version["data_sha256"],
        "n_train": int(version["n_train"]),
        "n_test": len(y_test),
        "n_features": int(n_features),
        "positive_label": settings.POSITIVE_LABEL,
        "positive_rate_test": _r(np.mean(np.asarray(y_test).astype(int))),
        "threshold": scored["threshold"],
        "accuracy": scored["accuracy"],
        "precision": scored["precision"],
        "recall": scored["recall"],
        "f1": scored["f1"],
        "roc_auc": scored["roc_auc"],
        "pr_auc": scored["pr_auc"],
        "brier": scored["brier"],
        "log_loss": scored["log_loss"],
        "confusion": scored["confusion"],
        "majority_baseline_accuracy": scored["majority_baseline_accuracy"],
        "candidates_cv": version["candidates"],
        "selection_metric": version["selection_metric"],
    }


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        params = settings.load_params()
        cfg = params["evaluate"]
        with open(settings.VERSION_PATH, encoding="utf-8") as fh:
            version = json.load(fh)
        pipeline: Pipeline = joblib.load(settings.MODEL_PATH)
        X_test, y_test = load_npz(settings.FEATURES_TEST_PATH)
        test_df = read_raw_test(settings.RAW_TEST_PATH)
        if len(test_df) != X_test.shape[0]:
            raise RuntimeError(
                f"test.csv has {len(test_df)} rows but test.npz has {X_test.shape[0]}"
            )
        if not np.array_equal(test_df[settings.LABEL_COLUMN].to_numpy().astype(int), y_test):
            raise RuntimeError("labels in test.csv and test.npz disagree")

        clf = pipeline.named_steps["classify"]
        vectorizer = pipeline.named_steps["vectorize"]
        p_test = clf.predict_proba(X_test)[:, 1]
        diff = parity_check(pipeline, test_df[settings.TEXT_COLUMN], p_test)
        log.info(
            "train/serve parity: pipeline on raw text matches the saved matrix "
            "(max abs diff %.2e over %d rows)",
            diff,
            len(p_test),
        )

        n_features = len(vectorizer.vocabulary_)
        if int(version.get("n_features", n_features)) != n_features:
            log.warning(
                "version.json n_features %s differs from the vectoriser (%d)",
                version.get("n_features"),
                n_features,
            )
        metrics = build_metrics(version, y_test, p_test, float(cfg["threshold"]), n_features)
        terms = top_terms(clf, feature_names(vectorizer), int(cfg["top_terms"]))
        _write_json(metrics, settings.METRICS_PATH)
        _write_json(terms, settings.TOP_TERMS_PATH)
        log.info("wrote %s and %s", settings.METRICS_PATH, settings.TOP_TERMS_PATH)

        if bool(cfg.get("figures", True)):
            written = write_figures(y_test, p_test, metrics, terms, settings.FIGURES_DIR)
            log.info("wrote %d figures to %s", len(written), settings.FIGURES_DIR)
        else:
            removed = clear_figures(settings.FIGURES_DIR)
            log.info(
                "figures disabled in params.yaml; removed %d stale figures from %s",
                len(removed),
                settings.FIGURES_DIR,
            )
    except Exception as exc:
        log.error("evaluate failed: %s: %s", type(exc).__name__, exc)
        return 1
    print(
        f"test accuracy={metrics['accuracy']} f1={metrics['f1']} roc_auc={metrics['roc_auc']} "
        f"pr_auc={metrics['pr_auc']} brier={metrics['brier']} n_test={metrics['n_test']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

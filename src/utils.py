"""
Shared utilities for the ENGR422 used vehicle prices project.

Imported by notebooks 03a, 03c, and 04. (03b was implemented before this
module existed; refactor in passing if you touch it for another reason.)

Notebooks need ``sys.path.insert(0, "../src")`` before
``from utils import ...`` so Python can find this file.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Paths and project-wide constants
# ---------------------------------------------------------------------------

# Anchor to the project root so paths work regardless of the caller's CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_train():
    """Return (X_train, y_train) — DataFrame and Series."""
    X = pd.read_csv(DATA_DIR / "X_train.csv")
    y = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    return X, y


def load_test():
    """Return (X_test, y_test). Use ONLY in notebook 04 (held-out)."""
    X = pd.read_csv(DATA_DIR / "X_test.csv")
    y = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")
    return X, y


def load_preprocessor(kind: str):
    """Load Aybars's fitted preprocessor pipeline.

    Parameters
    ----------
    kind : {"linear", "tree"}
        ``linear`` for OLS/Ridge/Lasso (includes scaling and log-transforms);
        ``tree`` for Random Forest / XGBoost (lighter, no scaling).
    """
    if kind not in {"linear", "tree"}:
        raise ValueError(f"kind must be 'linear' or 'tree', got {kind!r}")
    return joblib.load(MODELS_DIR / f"preprocessor_{kind}.pkl")


def load_model(name: str):
    """Load a trained model pipeline saved by 03a/03b/03c."""
    return joblib.load(MODELS_DIR / f"{name}.pkl")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(y_true, y_pred) -> dict:
    """The project's four regression metrics in one dict."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def metrics_table(predictions: dict, y_true) -> pd.DataFrame:
    """Build a comparison table across models.

    Parameters
    ----------
    predictions : dict[str, array-like]
        ``{"linear_regression": y_pred1, "random_forest": y_pred2, ...}``
    """
    rows = {name: regression_metrics(y_true, pred) for name, pred in predictions.items()}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Plots — each accepts an optional ``ax`` for grid composition
# ---------------------------------------------------------------------------

def plot_pred_vs_actual(y_true, y_pred, ax=None, title: str = ""):
    """Scatter of predicted vs actual prices with a y=x reference line."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ax.scatter(y_true, y_pred, alpha=0.25, s=6)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect")
    ax.set_xlabel("Actual price ($)")
    ax.set_ylabel("Predicted price ($)")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left")
    return ax


def plot_residuals(y_true, y_pred, ax=None, title: str = "", bins: int = 80):
    """Histogram of residuals (actual - predicted)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    ax.hist(residuals, bins=bins)
    ax.axvline(0, color="k", linestyle="--", lw=1)
    ax.set_xlabel("Residual = actual - predicted ($)")
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    return ax


def plot_top_coefficients(coefs, feature_names, n: int = 15, ax=None, title: str = ""):
    """Top |coef| bars for a linear model — red = negative, blue = positive."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(4, n * 0.3)))
    df = (
        pd.DataFrame({"feature": list(feature_names), "coef": np.asarray(coefs)})
        .assign(abs_coef=lambda d: d["coef"].abs())
        .nlargest(n, "abs_coef")
        .iloc[::-1]
    )
    colors = ["tab:red" if c < 0 else "tab:blue" for c in df["coef"]]
    ax.barh(df["feature"], df["coef"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Coefficient (on scaled features)")
    if title:
        ax.set_title(title)
    return ax


def plot_feature_importances(importances, feature_names, n: int = 15, ax=None, title: str = ""):
    """Top-N bar chart for tree/permutation importances (always positive)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(4, n * 0.3)))
    df = (
        pd.DataFrame({"feature": list(feature_names), "importance": np.asarray(importances)})
        .nlargest(n, "importance")
        .iloc[::-1]
    )
    ax.barh(df["feature"], df["importance"], color="tab:blue")
    ax.set_xlabel("Importance")
    if title:
        ax.set_title(title)
    return ax

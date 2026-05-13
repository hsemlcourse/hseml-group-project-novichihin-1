"""Model interpretability helpers (permutation importance)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import SEED


def permutation_importance_table(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int = 5,
    random_state: int = SEED,
    top_n: int = 20,
) -> pd.DataFrame:
    """Compute permutation importance; return top_n features by mean drop in RMSE proxy.

    Uses ``neg_root_mean_squared_error`` so higher importance means worse score when feature shuffled.
    """
    r = permutation_importance(
        estimator,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    names = list(X.columns)
    df = pd.DataFrame(
        {
            "feature": names,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return df.head(top_n).reset_index(drop=True)


def save_permutation_importance(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path,
    csv_name: str = "permutation_importance_cp2.csv",
    png_name: str = "permutation_importance_cp2.png",
    top_n: int = 20,
) -> tuple[Path, Path | None]:
    """Save CSV and horizontal bar chart for top permutation importances."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl = permutation_importance_table(estimator, X, y, top_n=top_n)
    csv_path = out_dir / csv_name
    tbl.to_csv(csv_path, index=False)
    sub = tbl.head(min(15, len(tbl))).iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(sub["feature"], sub["importance_mean"], xerr=sub["importance_std"])
    plt.xlabel("Increase in RMSE when feature shuffled (mean ± std)")
    plt.title("Permutation importance (validation)")
    plt.tight_layout()
    png_path = out_dir / png_name
    plt.savefig(png_path, dpi=150)
    plt.close()
    return csv_path, png_path

"""Train vs test summary statistics for numeric features (covariate shift diagnostics)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import TARGET
from src.preprocessing import prepare_features


def numeric_shift_table(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare mean, median, std between train and test on engineered numeric columns (no target)."""
    tr = prepare_features(train_df)
    te = prepare_features(test_df)
    if TARGET in tr.columns:
        tr = tr.drop(columns=[TARGET])
    if TARGET in te.columns:
        te = te.drop(columns=[TARGET])
    num_cols = tr.select_dtypes(include=[np.number]).columns.intersection(
        te.select_dtypes(include=[np.number]).columns
    )
    rows = []
    for c in sorted(num_cols):
        m_tr, m_te = float(tr[c].mean()), float(te[c].mean())
        med_tr, med_te = float(tr[c].median()), float(te[c].median())
        s_tr_raw = float(tr[c].std())
        s_tr = s_tr_raw if s_tr_raw > 0 else 1e-9
        s_te = float(te[c].std())
        delta_mean = m_te - m_tr
        delta_median = med_te - med_tr
        rows.append(
            {
                "feature": c,
                "mean_train": m_tr,
                "mean_test": m_te,
                "median_train": med_tr,
                "median_test": med_te,
                "std_train": s_tr_raw,
                "std_test": s_te,
                "delta_mean": delta_mean,
                "delta_median": delta_median,
                "abs_delta_mean_over_std_train": abs(delta_mean) / s_tr,
                "abs_delta_median_over_std_train": abs(delta_median) / s_tr,
            }
        )
    # Sort by mean-based shift (primary diagnostic); median columns still exported for review.
    return pd.DataFrame(rows).sort_values("abs_delta_mean_over_std_train", ascending=False)


def save_shift_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
    csv_name: str = "train_test_shift_cp2.csv",
    png_name: str = "train_test_shift_cp2.png",
    bar_top: int = 12,
) -> tuple[Path, Path]:
    """Write shift CSV and a bar chart of top features by |Δmean|/std_train."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl = numeric_shift_table(train_df, test_df)
    csv_path = out_dir / csv_name
    tbl.to_csv(csv_path, index=False)
    sub = tbl.head(bar_top).iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(sub["feature"], sub["abs_delta_mean_over_std_train"])
    plt.xlabel("|mean_test − mean_train| / std_train")
    plt.title("Top numeric features: train vs test shift")
    plt.tight_layout()
    png_path = out_dir / png_name
    plt.savefig(png_path, dpi=150)
    plt.close()
    return csv_path, png_path

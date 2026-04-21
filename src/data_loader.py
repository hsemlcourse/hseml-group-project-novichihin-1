"""Download the Kaggle dataset into data/raw/ and load it as a DataFrame.

Usage (programmatic):

    from src.data_loader import ensure_dataset, load_raw
    csv_path = ensure_dataset()
    df = load_raw()

CLI:

    python -m src.data_loader
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW_DIR, KAGGLE_DATASET, RAW_CSV_NAME

_KAGGLE_PUBLIC_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"


def _target_csv() -> Path:
    return DATA_RAW_DIR / RAW_CSV_NAME


def _download_with_kaggle_api(raw_dir: Path) -> Path | None:
    """Try to download via the official Kaggle API. Return CSV path on success, None otherwise."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-not-found]
    except Exception:
        return None

    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET, path=str(raw_dir), unzip=True, quiet=False)
    except Exception:
        return None

    csv_path = _target_csv()
    return csv_path if csv_path.exists() else None


def _download_anonymously(raw_dir: Path) -> Path | None:
    """Fallback: fetch the zip directly from Kaggle's public download endpoint."""
    import urllib.request

    zip_path = raw_dir / "dataset.zip"
    try:
        urllib.request.urlretrieve(_KAGGLE_PUBLIC_URL, zip_path)
    except Exception:
        return None

    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
    except zipfile.BadZipFile:
        return None
    finally:
        if zip_path.exists():
            try:
                os.remove(zip_path)
            except OSError:
                pass

    csv_path = _target_csv()
    return csv_path if csv_path.exists() else None


def ensure_dataset(raw_dir: Path = DATA_RAW_DIR) -> Path:
    """Ensure the raw CSV is available locally. Try Kaggle API first, then anonymous download."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / RAW_CSV_NAME
    if csv_path.exists():
        return csv_path

    for downloader in (_download_with_kaggle_api, _download_anonymously):
        path = downloader(raw_dir)
        if path is not None:
            return path

    raise FileNotFoundError(
        f"Could not obtain dataset. Place {RAW_CSV_NAME} into {raw_dir} manually "
        "or configure Kaggle API credentials in ~/.kaggle/kaggle.json."
    )


def load_raw(raw_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    """Load the raw dataset as a pandas DataFrame, downloading it if necessary."""
    csv_path = ensure_dataset(raw_dir)
    df = pd.read_csv(csv_path)
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    return df


if __name__ == "__main__":
    path = ensure_dataset()
    print(f"Dataset ready at: {path}")

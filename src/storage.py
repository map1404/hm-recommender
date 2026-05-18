"""
storage.py

Small helpers for reading and writing dataframe artifacts without assuming
parquet is reliable in the local Python/pyarrow environment.
"""

from pathlib import Path

import pandas as pd


def save_table(df: pd.DataFrame, path_without_suffix: Path) -> Path:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)

    pickle_path = path_without_suffix.with_suffix(".pkl")
    df.to_pickle(pickle_path)

    parquet_path = path_without_suffix.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        pass

    return pickle_path


def load_table(path_without_suffix: Path) -> pd.DataFrame:
    pickle_path = path_without_suffix.with_suffix(".pkl")
    if pickle_path.exists():
        return pd.read_pickle(pickle_path)

    parquet_path = path_without_suffix.with_suffix(".parquet")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    raise FileNotFoundError(
        f"Neither {pickle_path.name} nor {parquet_path.name} exists in {path_without_suffix.parent}"
    )

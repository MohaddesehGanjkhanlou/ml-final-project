"""
Jet Engine Hospital - Data Loading Module
Centralized, leakage-safe loading and label construction for NASA C-MAPSS.
Imported by BOTH the training notebook and the deployment app (ensures parity).
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

import config


def load_raw(subset: str, split: str) -> pd.DataFrame:
    # Load a raw C-MAPSS file and assign explicit column names
    if subset not in config.DATASET_INFO:
        raise ValueError(f"Unknown subset: {subset}")
    if split not in ('train', 'test'):
        raise ValueError(f"split must be 'train' or 'test', got {split}")

    path = config.DATA_RAW / f"{split}_{subset}.txt"
    if not path.exists(): raise FileNotFoundError(path)

    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != 26: raise ValueError(f"Expected 26 columns, got {df.shape[1]} in {path.name}")
    df.columns = config.ALL_COLS

    # Validate engine count against official spec
    expected = config.DATASET_INFO[subset][f"{split}_engines"]
    actual = df["engine_id"].nunique()
    if actual != expected:
        raise ValueError(f"{split}_{subset}: expected {expected} engines, found {actual}")

    # Enforce integer types for keys
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    if df.duplicated(["engine_id","cycle"]).any():
        raise ValueError(f"duplicate (engine_id, cycle) keys in {path.name}")
    if not pd.notna(df).all().all(): raise ValueError(f"missing values in {path.name}")
    if sorted(df.engine_id.unique().tolist()) != list(range(1, expected+1)):
        raise ValueError(f"non-sequential official engine IDs in {path.name}")
    return df


def add_train_rul(df: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """
    Add RUL target for TRAIN data (run-to-failure).
    RUL(i, t) = T(i) - t, where T(i) = max cycle of engine i.

    Parameters
    ----------
    cap : optional int. If given, RUL is clipped to [0, cap] (piecewise-linear).
    """
    df = df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def load_test_with_rul(subset: str, cap: int | None = None) -> pd.DataFrame:
    """
    Load TEST data and attach the true RUL for every row using the official
    RUL_FD00x.txt file
    For a test engine with last observed cycle t_last and official RUL r:
        the engine's (virtual) failure cycle is T = t_last + r
        so RUL(i, t) = T - t  for every observed cycle t
    """
    df = load_raw(subset, "test").copy()

    rul_path = config.DATA_RAW / f"RUL_{subset}.txt"
    if not rul_path.exists(): raise FileNotFoundError(rul_path)
    rul_final = pd.read_csv(rul_path, header=None).iloc[:, 0].values

    engine_ids = sorted(df["engine_id"].unique())
    if len(rul_final) != len(engine_ids):
        raise ValueError(f"RUL_{subset}: {len(rul_final)} values but {len(engine_ids)} test engines")
    if engine_ids != list(range(1,len(engine_ids)+1)):
        raise ValueError("official test RUL alignment requires sequential engine IDs")

    # Map each engine to its official final-cycle RUL
    rul_map = dict(zip(engine_ids, rul_final))
    last_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["rul_final"] = df["engine_id"].map(rul_map)
    # Virtual failure cycle T = last_cycle + rul_final
    df["RUL"] = (last_cycle + df["rul_final"]) - df["cycle"]
    df = df.drop(columns=["rul_final"])

    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def add_classification_labels(df: pd.DataFrame,
                              horizons: list[int] | None = None) -> pd.DataFrame:
    """
    Add binary failure-horizon labels from the RUL column.
    label_h{h} = 1 if RUL <= h else 0.
    Requires an existing 'RUL' column (uncapped or capped >= max horizon).
    """
    if "RUL" not in df.columns: raise ValueError("construct RUL before labels")
    if horizons is None:
        horizons = config.CLASSIFICATION_HORIZONS
    df = df.copy()
    for h in horizons:
        df[f"label_h{h}"] = (df["RUL"] <= h).astype(int)
    return df


if __name__ == "__main__":
    tr = load_raw("FD001", "train")
    tr = add_train_rul(tr)
    tr = add_classification_labels(tr)
    print("TRAIN FD001:", tr.shape)
    print(tr[["engine_id", "cycle", "RUL", "label_h10", "label_h20", "label_h30"]].head())
    print("\nMax RUL:", tr["RUL"].max(), "| Positive rate h30:",
          round(tr["label_h30"].mean(), 3))

    te = load_test_with_rul("FD001")
    print("\nTEST FD001:", te.shape)
    print("Test min/max RUL:", te["RUL"].min(), te["RUL"].max())

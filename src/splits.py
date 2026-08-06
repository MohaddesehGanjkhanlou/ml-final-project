"""
Jet Engine Hospital - Engine-Level Split Module

Creates and persists disjoint engine-ID splits. The split is by ENGINE ID,
never by row, so no engine's cycles leak across partitions.

Design for FD00x:
  - Official TRAIN engines are split into (train, val) by engine ID.
  - Official TEST engines form the locked test set (touched once).
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

import config
import data_loader as dl


def make_engine_split(subset: str,
                      val_fraction: float = config.VAL_FRACTION,
                      seed: int = None) -> dict:
    """
    Split official-train engine IDs into train/val; official-test IDs become test.

    Returns a dict with lists: {'train': [...], 'val': [...], 'test': [...]}.
    """
    if seed is None:
        seed = config.RANDOM_SEED

    # Official train engines -> split into train/val by ID
    train_df = dl.load_raw(subset, "train")
    train_engines = np.array(sorted(train_df["engine_id"].unique()))

    rng = np.random.RandomState(seed)
    shuffled = train_engines.copy()
    rng.shuffle(shuffled)

    n_val = int(round(len(shuffled) * val_fraction))
    val_ids = sorted(shuffled[:n_val].tolist())
    tr_ids = sorted(shuffled[n_val:].tolist())

    # Official test engines -> locked test set
    test_df = dl.load_raw(subset, "test")
    test_ids = sorted(test_df["engine_id"].unique().tolist())

    split = {
        "schema_version": 2,
        "subset": subset,
        "seed": int(seed),
        "sources": {"train": "official_train", "val": "official_train",
                    "test": "official_test"},
        "train": tr_ids, "val": val_ids, "test": test_ids,
        "qualified": {
            "train": [f"{subset}:official_train:{i}" for i in tr_ids],
            "val": [f"{subset}:official_train:{i}" for i in val_ids],
            "test": [f"{subset}:official_test:{i}" for i in test_ids],
        },
    }

    # --- Leakage assertions ---
    assert set(tr_ids).isdisjoint(val_ids), "train/val overlap!"
    assert len(tr_ids) + len(val_ids) == len(train_engines), "train+val != official train"
    q = split["qualified"]
    assert set(q["train"]).isdisjoint(q["val"])
    assert set(q["train"]).isdisjoint(q["test"])
    assert set(q["val"]).isdisjoint(q["test"])
    # (train/val engine IDs come from the official-train pool; test IDs come from the
    #  official-test pool. They share numeric IDs but are DIFFERENT engines in
    #  different files, so we do NOT compare them numerically.)

    return split

def make_validation_subdivision(
    split: dict,
    tune_fraction: float = config.VAL_TUNE_FRACTION,
    seed: int | None = None,
) -> dict:
    """
    Deterministically subdivide validation engine IDs into val_tune and
    val_calib before model selection, calibration, uncertainty fitting,
    anomaly-policy selection, or threshold tuning.

    Only engine IDs from the official-training validation partition are used.
    No lifetime, RUL, sensor value, label, or official-test information is read.
    """
    if seed is None:
        seed = config.RANDOM_SEED

    if not 0.0 < tune_fraction < 1.0:
        raise ValueError(
            f"tune_fraction must be between 0 and 1, got {tune_fraction}"
        )

    required_keys = {
        "schema_version",
        "subset",
        "sources",
        "train",
        "val",
        "test",
        "qualified",
    }

    missing_keys = required_keys.difference(split)

    if missing_keys:
        raise ValueError(
            f"parent split is missing required keys: {sorted(missing_keys)}"
        )

    subset = split["subset"]

    if split["sources"].get("val") != "official_train":
        raise ValueError(
            "validation subdivision must originate from official_train"
        )

    val_ids = sorted(int(engine_id) for engine_id in split["val"])

    if len(val_ids) < 2:
        raise ValueError(
            "at least two validation engines are required for subdivision"
        )

    if len(set(val_ids)) != len(val_ids):
        raise ValueError("duplicate engine IDs in validation partition")

    shuffled = np.asarray(val_ids, dtype=int)

    rng = np.random.RandomState(seed)
    rng.shuffle(shuffled)

    n_tune = int(round(len(shuffled) * tune_fraction))
    n_tune = min(max(n_tune, 1), len(shuffled) - 1)

    tune_ids = sorted(shuffled[:n_tune].tolist())
    calib_ids = sorted(shuffled[n_tune:].tolist())

    tune_set = set(tune_ids)
    calib_set = set(calib_ids)
    parent_val_set = set(val_ids)

    assert tune_set.isdisjoint(calib_set), (
        "val_tune and val_calib overlap"
    )

    assert tune_set.union(calib_set) == parent_val_set, (
        "val_tune + val_calib do not reproduce the parent validation set"
    )

    qualified_tune = [
        f"{subset}:official_train:{engine_id}"
        for engine_id in tune_ids
    ]

    qualified_calib = [
        f"{subset}:official_train:{engine_id}"
        for engine_id in calib_ids
    ]

    train_qualified = set(split["qualified"]["train"])
    test_qualified = set(split["qualified"]["test"])

    assert set(qualified_tune).isdisjoint(train_qualified)
    assert set(qualified_calib).isdisjoint(train_qualified)
    assert set(qualified_tune).isdisjoint(test_qualified)
    assert set(qualified_calib).isdisjoint(test_qualified)

    return {
        "schema_version": 1,
        "subset": subset,
        "seed": int(seed),
        "source": "official_train",
        "selection_basis": "engine_id_only",
        "tune_fraction": float(tune_fraction),
        "parent_validation_engine_ids": val_ids,
        "val_tune": tune_ids,
        "val_calib": calib_ids,
        "qualified": {
            "val_tune": qualified_tune,
            "val_calib": qualified_calib,
        },
    }


def make_stratified_engine_split(subset: str,
                                 val_fraction: float = 0.30,
                                 n_bins: int = 3,
                                 seed: int = None) -> dict:
    """
    Stratified engine-level split based on engine LIFETIME (total cycles).

    Engines are binned into lifetime strata (e.g. short/medium/long), then
    within each stratum a fixed fraction is assigned to validation. This keeps
    the lifetime (and therefore degradation-pattern) distribution similar
    across train and val, improving tuning reliability.

    Leakage-safe: uses only official-train engines' own lifetimes; test set
    (official-test file) is untouched and forms the locked test partition.
    """
    if seed is None:
        seed = config.RANDOM_SEED

    train_df = dl.load_raw(subset, "train")
    lifetimes = train_df.groupby("engine_id")["cycle"].max()  # T(i) per engine
    engine_ids = lifetimes.index.values

    # Quantile bins -> roughly equal-count strata (short/medium/long)
    # Use rank to avoid duplicate-edge issues with qcut.
    ranks = lifetimes.rank(method="first")
    strata = np.floor((ranks - 1) / len(ranks) * n_bins).astype(int)
    strata = strata.clip(upper=n_bins - 1)

    rng = np.random.RandomState(seed)
    tr_ids, val_ids = [], []
    for b in range(n_bins):
        ids_in_bin = engine_ids[strata.values == b]
        ids_in_bin = ids_in_bin.copy()
        rng.shuffle(ids_in_bin)
        n_val = int(round(len(ids_in_bin) * val_fraction))
        val_ids.extend(ids_in_bin[:n_val].tolist())
        tr_ids.extend(ids_in_bin[n_val:].tolist())

    tr_ids = sorted(tr_ids)
    val_ids = sorted(val_ids)

    test_df = dl.load_raw(subset, "test")
    test_ids = sorted(test_df["engine_id"].unique().tolist())

    split = {"train": tr_ids, "val": val_ids, "test": test_ids}

    # --- Integrity assertions ---
    assert set(tr_ids).isdisjoint(val_ids), "train/val overlap!"
    assert len(tr_ids) + len(val_ids) == len(engine_ids), \
        "train+val engine count != official-train engine count"
    assert len(set(tr_ids)) == len(tr_ids), "duplicate train IDs"
    assert len(set(val_ids)) == len(val_ids), "duplicate val IDs"
    return split


def save_split(subset: str, split: dict) -> Path:
    """Persist a split to data/splits/<subset>_split.json."""
    config.DATA_SPLITS.mkdir(parents=True, exist_ok=True)
    path = config.DATA_SPLITS / f"{subset}_split.json"
    with open(path, "w") as f:
        json.dump(split, f, indent=2)
    return path


def save_validation_subdivision(
    subset: str,
    subdivision: dict,
) -> Path:
    """
    Atomically persist the deterministic val_tune/val_calib contract.

    The temporary file is completely written and validated before it replaces
    the final JSON file.
    """
    if subset not in config.DATASET_INFO:
        raise ValueError(f"Unknown subset: {subset}")

    if subdivision.get("subset") != subset:
        raise ValueError(
            "subdivision subset does not match the requested subset"
        )

    required_keys = {
        "schema_version",
        "subset",
        "seed",
        "source",
        "selection_basis",
        "tune_fraction",
        "parent_validation_engine_ids",
        "val_tune",
        "val_calib",
        "qualified",
    }

    missing_keys = required_keys.difference(subdivision)

    if missing_keys:
        raise ValueError(
            f"subdivision is missing keys: {sorted(missing_keys)}"
        )

    tune_ids = set(subdivision["val_tune"])
    calib_ids = set(subdivision["val_calib"])
    parent_ids = set(subdivision["parent_validation_engine_ids"])

    if tune_ids.intersection(calib_ids):
        raise ValueError("val_tune and val_calib overlap")

    if tune_ids.union(calib_ids) != parent_ids:
        raise ValueError(
            "val_tune + val_calib do not reconstruct parent validation"
        )

    if subdivision["source"] != "official_train":
        raise ValueError(
            "validation subdivision source must be official_train"
        )

    if subdivision["selection_basis"] != "engine_id_only":
        raise ValueError(
            "validation subdivision must use engine_id_only"
        )

    config.DATA_SPLITS.mkdir(parents=True, exist_ok=True)

    path = (
        config.DATA_SPLITS
        / f"{subset}_validation_subdivision.json"
    )

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    payload = json.dumps(
        subdivision,
        indent=2,
        sort_keys=True,
    ) + "\n"

    temporary_path.write_text(
        payload,
        encoding="utf-8",
    )

    # Validate the complete temporary JSON before promotion.
    temporary_payload = json.loads(
        temporary_path.read_text(encoding="utf-8")
    )

    if temporary_payload != subdivision:
        raise ValueError(
            "temporary validation-subdivision JSON failed parity check"
        )

    temporary_path.replace(path)

    return path


def load_validation_subdivision(subset: str) -> dict:
    """
    Load and validate a persisted val_tune/val_calib contract.
    """
    if subset not in config.DATASET_INFO:
        raise ValueError(f"Unknown subset: {subset}")

    path = (
        config.DATA_SPLITS
        / f"{subset}_validation_subdivision.json"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"No saved validation subdivision at {path}"
        )

    subdivision = json.loads(
        path.read_text(encoding="utf-8")
    )

    if subdivision.get("subset") != subset:
        raise ValueError(
            "saved subdivision subset does not match filename subset"
        )

    if subdivision.get("source") != "official_train":
        raise ValueError(
            "saved subdivision source must be official_train"
        )

    if subdivision.get("selection_basis") != "engine_id_only":
        raise ValueError(
            "saved subdivision must use engine_id_only"
        )

    tune_ids = set(subdivision.get("val_tune", []))
    calib_ids = set(subdivision.get("val_calib", []))
    parent_ids = set(
        subdivision.get("parent_validation_engine_ids", [])
    )

    if not tune_ids:
        raise ValueError("saved val_tune partition is empty")

    if not calib_ids:
        raise ValueError("saved val_calib partition is empty")

    if tune_ids.intersection(calib_ids):
        raise ValueError(
            "saved val_tune and val_calib partitions overlap"
        )

    if tune_ids.union(calib_ids) != parent_ids:
        raise ValueError(
            "saved subdivision does not reconstruct parent validation"
        )

    return subdivision


def load_split(subset: str) -> dict:
    """Load a previously saved split."""
    path = config.DATA_SPLITS / f"{subset}_split.json"
    assert path.exists(), f"No saved split at {path}. Run make/save first."
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    sp = make_engine_split("FD001")
    print("FD001 split sizes:",
          {k: len(v) for k, v in sp.items()})
    p = save_split("FD001", sp)
    print("Saved to:", p)
    print("First 5 train IDs:", sp["train"][:5])
    print("First 5 val IDs:  ", sp["val"][:5])
    print("First 5 test IDs: ", sp["test"][:5])

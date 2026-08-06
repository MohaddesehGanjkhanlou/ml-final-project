"""
Jet Engine Hospital - Feature Engineering Module
Leakage-safe, causal feature construction for NASA C-MAPSS.

Design:
  * X_df contains ONLY model features (no engine_id / cycle columns).
  * metadata_df (separate) carries original_index, engine_id, cycle.
  * Alignment preserved by shared DataFrame INDEX (not positional .values).
  * Pipeline CONFIGURATION metadata is returned separately from row-level metadata.
  * Arbitrary input row order is allowed; each engine is sorted by cycle
    INTERNALLY for temporal calculations, then results are restored to the
    original input index/order.

Guarantees:
  * Windowed statistics computed PER ENGINE (never cross engine bounds).
  * Rolling windows are TRAILING (right-aligned) -> only past/current cycles.
  * Feature at cycle t depends only on rows with cycle <= t (causality).
  * StandardScaler fitted on TRAINING features only and reused unchanged.

Representations:
  * 'baseline' : current-cycle sensor values (+ optional settings)
  * 'window'   : current-cycle values + trailing temporal features
                 (mean/std/min/max/slope/diff/ewma) + effective_window_size
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

import config

def _trailing_slope(arr: np.ndarray) -> float:
    """Trailing linear-regression slope. <2 obs -> 0.0 (no trend definable)."""
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = arr.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (arr - y_mean)).sum() / denom)


def _add_window_features(g: pd.DataFrame, sensors: list[str],
                         window: int) -> pd.DataFrame:
    """
    Trailing-window features for ONE engine. Caller must pass g already sorted
    by cycle. Includes RAW current value plus temporal stats. min_periods=1
    avoids uncontrolled NaNs.
    """
    out = {}
    for s in sensors:
        col = g[s]
        roll = col.rolling(window=window, min_periods=1)
        out[f"{s}"] = col                          # raw current value (causal)
        out[f"{s}_mean"] = roll.mean()
        out[f"{s}_std"] = roll.std().fillna(0.0)   # 1 obs -> 0.0
        out[f"{s}_min"] = roll.min()
        out[f"{s}_max"] = roll.max()
        # Vectorized trailing OLS slope.  With local x=0..n-1,
        # slope=(n*sum(x*y)-sum(x)*sum(y))/(n*sum(x^2)-sum(x)^2).
        # Rolling sums use only the current and preceding rows, preserving causality.
        y = col.to_numpy(dtype=float)
        global_x = np.arange(len(y), dtype=float)
        nobs = np.minimum(global_x + 1, window)
        start = np.maximum(0.0, global_x - window + 1)
        sum_y = pd.Series(y, index=g.index).rolling(window, min_periods=1).sum().to_numpy()
        sum_global_xy = pd.Series(global_x * y, index=g.index).rolling(
            window, min_periods=1).sum().to_numpy()
        sum_xy = sum_global_xy - start * sum_y
        sum_x = nobs * (nobs - 1.0) / 2.0
        sum_x2 = nobs * (nobs - 1.0) * (2.0 * nobs - 1.0) / 6.0
        denom = nobs * sum_x2 - sum_x ** 2
        slope = np.divide(nobs * sum_xy - sum_x * sum_y, denom,
                          out=np.zeros_like(sum_y), where=denom > 0)
        out[f"{s}_slope"] = pd.Series(slope, index=g.index)
        out[f"{s}_diff"] = col.diff().fillna(0.0)  # first cycle -> 0.0
        out[f"{s}_ewma"] = col.ewm(span=window, adjust=False).mean()
    feat = pd.DataFrame(out, index=g.index)
    n = len(g)
    eff = np.minimum(np.arange(1, n + 1), window).astype(float)
    feat["effective_window_size"] = pd.Series(eff, index=g.index)
    return feat


@dataclass
class FeaturePipeline:
    # Leakage-safe feature pipeline.
    keep_sensors: list[str]
    keep_settings: list[str] = field(default_factory=list)
    representation: str = "window"
    window: int = 15
    include_settings: bool = False
    require_continuity: bool = False

    feature_names_: list[str] = field(default=None, init=False)
    scaler_mean_: np.ndarray = field(default=None, init=False)
    scaler_scale_: np.ndarray = field(default=None, init=False)
    _fitted: bool = field(default=False, init=False)

    def _validate_input(self, df: pd.DataFrame):
        assert self.window >= 1, f"window must be >= 1, got {self.window}"
        assert df.index.is_unique, "DataFrame index must be unique."

        required = ["engine_id", "cycle"] + list(self.keep_sensors)
        if self.include_settings:
            required += list(self.keep_settings)
        missing = [c for c in required if c not in df.columns]
        assert not missing, f"Missing required columns: {missing}"

        dup = df.duplicated(subset=["engine_id", "cycle"]).sum()
        assert dup == 0, f"{dup} duplicate (engine_id, cycle) keys."

        cyc = df["cycle"].to_numpy()
        assert np.isfinite(cyc).all(), "Non-finite cycle values present."
        assert (cyc > 0).all(), "Cycle values must be positive."

        feat_cols = list(self.keep_sensors)
        if self.include_settings:
            feat_cols += list(self.keep_settings)
        vals = df[feat_cols].to_numpy(dtype=float)
        assert np.isfinite(vals).all(), \
            "Non-finite (NaN/Inf) values in sensor/setting inputs."

        if self.require_continuity:
            for e, g in df.groupby("engine_id"):
                s = np.sort(g["cycle"].to_numpy())
                expected = np.arange(s[0], s[0] + len(s))
                assert np.array_equal(s, expected), \
                    f"engine {e}: cycles not continuous after sorting."

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return ONLY feature columns, indexed by the input DataFrame index.
        Each engine is sorted by cycle INTERNALLY for temporal computation;
        the result is then reindexed back to the ORIGINAL input order.
        """
        assert self.representation in ("baseline", "window")

        if self.representation == "baseline":
            feat = df[self.keep_sensors].copy()
        else:
            parts = []
            for _, g in df.groupby("engine_id", sort=False):
                g_sorted = g.sort_values("cycle")
                parts.append(_add_window_features(g_sorted, self.keep_sensors,
                                                  self.window))
            feat = pd.concat(parts)
            feat = feat.reindex(df.index)

        if self.include_settings and self.keep_settings:
            feat = feat.join(df[self.keep_settings])

        assert np.isfinite(feat.to_numpy(dtype=float)).all(), \
            "Generated feature matrix contains NaN/Inf after construction."
        return feat

    def fit(self, df_train: pd.DataFrame) -> "FeaturePipeline":
        self._validate_input(df_train)
        feat = self._build_features(df_train)
        self.feature_names_ = list(feat.columns)
        X = feat.to_numpy(dtype=float)
        self.scaler_mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        self.scaler_scale_ = std
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, scaled: bool = True):

        assert self._fitted, "FeaturePipeline must be fitted before transform."
        self._validate_input(df)

        feat = self._build_features(df)[self.feature_names_]

        if scaled:
            X = feat.to_numpy(dtype=float)
            X = (X - self.scaler_mean_) / self.scaler_scale_
            X_df = pd.DataFrame(X, index=feat.index, columns=self.feature_names_)
        else:
            X_df = feat.copy()

        y_rul = df["RUL"].copy() if "RUL" in df.columns else None
        label_cols = [c for c in df.columns if c.startswith("label_h")]
        y_labels = df[label_cols].copy() if label_cols else None

        metadata_df = df[["engine_id", "cycle"]].copy()
        metadata_df.insert(0, "original_index", metadata_df.index)

        config_meta = {
            "feature_cols": list(self.feature_names_),
            "n_features": len(self.feature_names_),
            "representation": self.representation,
            "window": self.window,
            "include_settings": self.include_settings,
            "scaled": scaled,
        }
        return X_df, y_rul, y_labels, metadata_df, config_meta

def _toy_frame(n_engines=3, base_cycle=25, seed=0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for e in range(1, n_engines + 1):
        T = base_cycle + e
        for t in range(1, T + 1):
            rows.append({
                "engine_id": e, "cycle": t,
                "sensor_2": 640 + 0.05 * t + rng.randn(),
                "sensor_4": 1400 + 0.1 * t + rng.randn(),
                "RUL": T - t,
            })
    df = pd.DataFrame(rows)
    for h in config.CLASSIFICATION_HORIZONS:
        df[f"label_h{h}"] = (df["RUL"] <= h).astype(int)
    return df

def test_causality_middle_cycle():
    df = _toy_frame()
    fp = FeaturePipeline(keep_sensors=["sensor_2", "sensor_4"],
                         representation="window", window=15).fit(df)
    X1, *_ = fp.transform(df, scaled=False)

    e = 2
    eng = df[df.engine_id == e].sort_values("cycle")
    t = eng["cycle"].iloc[len(eng) // 2]
    future_idx = eng[eng["cycle"] > t].index
    past_idx = eng[eng["cycle"] <= t].index

    df2 = df.copy()
    df2.loc[future_idx, ["sensor_2", "sensor_4"]] += 500.0
    X2, *_ = fp.transform(df2, scaled=False)

    a = X1.loc[past_idx].to_numpy()
    b = X2.loc[past_idx].to_numpy()
    assert np.allclose(a, b), \
        "Past features changed after modifying future rows (causality bug)."
    return True

def test_no_engine_boundary_crossing():
    df = _toy_frame()
    fp = FeaturePipeline(keep_sensors=["sensor_2"],
                         representation="window", window=15).fit(df)
    X, *_ = fp.transform(df, scaled=False)
    for e in df.engine_id.unique():
        first_idx = df[df.engine_id == e].sort_values("cycle").index[0]
        val = df.loc[first_idx, "sensor_2"]
        assert np.isclose(X.loc[first_idx, "sensor_2_mean"], val)
        assert np.isclose(X.loc[first_idx, "sensor_2_std"], 0.0)
        assert X.loc[first_idx, "effective_window_size"] == 1.0
    return True


def test_shuffled_input_alignment():
    df = _toy_frame()
    fp = FeaturePipeline(keep_sensors=["sensor_2", "sensor_4"],
                         representation="window", window=10).fit(df)
    X_ref, *_ = fp.transform(df, scaled=False)

    df_shuf = df.sample(frac=1.0, random_state=123)
    Xs, y_rul, y_lab, meta_df, _ = fp.transform(df_shuf, scaled=False)

    assert (Xs.index == df_shuf.index).all()
    assert (y_rul.index == df_shuf.index).all()
    assert (y_lab.index == df_shuf.index).all()
    assert (meta_df.index == df_shuf.index).all()
    assert (meta_df["engine_id"].values == df_shuf["engine_id"].values).all()
    assert (meta_df["cycle"].values == df_shuf["cycle"].values).all()
    assert (meta_df["original_index"].values == df_shuf.index.values).all()
    assert (y_rul.values == df_shuf["RUL"].values).all()

    common = df_shuf.index
    assert np.allclose(Xs.loc[common].to_numpy(),
                       X_ref.loc[common].to_numpy()), \
        "Shuffled features differ from sorted reference (alignment/causality bug)."
    return True

def test_scaler_learned_from_train_only():
    df = _toy_frame()
    train = df[df.engine_id.isin([1, 2])].copy()
    test  = df[df.engine_id == 3].copy()
    fp = FeaturePipeline(keep_sensors=["sensor_2"],
                         representation="window", window=15).fit(train)
    feat_train = fp._build_features(train)[fp.feature_names_].to_numpy(float)
    assert np.allclose(fp.scaler_mean_, feat_train.mean(axis=0))
    exp_std = feat_train.std(axis=0); exp_std[exp_std == 0] = 1.0
    assert np.allclose(fp.scaler_scale_, exp_std)
    m_before = fp.scaler_mean_.copy()
    fp.transform(test, scaled=True)
    assert np.allclose(fp.scaler_mean_, m_before), "scaler changed during transform!"
    return True

def test_input_validation_catches_bad_data():
    base = _toy_frame(n_engines=2)
    fp = FeaturePipeline(keep_sensors=["sensor_2"], representation="window", window=5)

    dup = pd.concat([base, base.iloc[[0]]], ignore_index=True)
    try:
        fp.fit(dup); raise RuntimeError("did not catch duplicate keys")
    except AssertionError:
        pass

    bad_idx = base.copy(); bad_idx.index = [0] * len(bad_idx)
    try:
        fp.fit(bad_idx); raise RuntimeError("did not catch non-unique index")
    except AssertionError:
        pass

    bad_cyc = base.copy(); bad_cyc.loc[bad_cyc.index[0], "cycle"] = 0
    try:
        fp.fit(bad_cyc); raise RuntimeError("did not catch non-positive cycle")
    except AssertionError:
        pass

    return True

def test_rejects_nan_inf_in_sensors():
    base = _toy_frame(n_engines=2)
    fp = FeaturePipeline(keep_sensors=["sensor_2", "sensor_4"],
                         representation="window", window=5)
    bad_nan = base.copy()
    bad_nan.loc[bad_nan.index[3], "sensor_2"] = np.nan
    try:
        fp.fit(bad_nan); raise RuntimeError("did not catch NaN sensor")
    except AssertionError:
        pass
    bad_inf = base.copy()
    bad_inf.loc[bad_inf.index[5], "sensor_4"] = np.inf
    try:
        fp.fit(bad_inf); raise RuntimeError("did not catch Inf sensor")
    except AssertionError:
        pass
    return True

def run_all_tests(verbose=True):
    tests = [
        ("causality (middle cycle, all future rows)", test_causality_middle_cycle),
        ("no engine-boundary crossing", test_no_engine_boundary_crossing),
        ("shuffled-input alignment", test_shuffled_input_alignment),
        ("scaler train-only", test_scaler_learned_from_train_only),
        ("input validation catches bad data", test_input_validation_catches_bad_data),
        ("rejects NaN/Inf in sensors", test_rejects_nan_inf_in_sensors),
    ]
    results = {}
    ok = True
    for name, fn in tests:
        try:
            fn(); results[name] = "PASS"
        except AssertionError as e:
            results[name] = f"FAIL: {e}"; ok = False
        except Exception as e:
            results[name] = f"ERROR: {type(e).__name__}: {e}"; ok = False
    if verbose:
        print(" FeaturePipeline automated tests ")
        for k, v in results.items():
            tag = "OK" if v == "PASS" else "XX"
            print(f"  [{tag}] {k}: {v}")
    return results, ok

if __name__ == "__main__":
    _, ok = run_all_tests()
    print("\nALL PASSED" if ok else "\nSOME TESTS FAILED")

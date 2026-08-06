"""
Jet Engine Hospital - Classification Module
=
Per-horizon binary "failure within h cycles" classifiers with:
  * Model factory (logistic / rf / gbr).
  * Metrics: Average Precision (AP, primary PR-based), precision/recall/F1,
    Brier, confusion matrix; PR curves produced in the notebook.
  * Platt scaling on DECISION SCORES (logit-transform for prob-only models).
  * Predeclared asymmetric threshold cost (FN:FP = 10:1 primary; 5:1 & 20:1
    planned as sensitivity). This is a ROW-LEVEL threshold cost, NOT a universal
    maintenance cost or the final engine-level alert policy (which later adds
    persistence, lead time, miss rate, late-warning delay, early-warning burden).
  * Cross-horizon CONSERVATIVE cumulative-max correction for P10<=P20<=P30
    (not described as a minimal isotonic projection).

Data discipline (enforced by notebook):
  train -> fit; val-tune -> model/feature/hparam selection;
  val-calib -> Platt calibration + threshold; official test -> final locked eval.
"""
from __future__ import annotations
import numpy as np
import pandas as _pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, f1_score, brier_score_loss,
                             confusion_matrix)

import config

# Predeclared ROW-LEVEL threshold cost. Primary 10:1; sensitivity 5:1 and 20:1.
FN_COST = 10.0
FP_COST = 1.0
SENSITIVITY_RATIOS = [5.0, 10.0, 20.0]   # FN:FP ratios for sensitivity analysis
_EPS = 1e-6                               # for probability->logit clipping
LABEL_MAP = {10: "label_h10", 20: "label_h20", 30: "label_h30"}


# --------------------------------------------------------------------------
# Model factory
# --------------------------------------------------------------------------
def make_classifier(name: str, **kw):
    """
      'logistic' -> LogisticRegression (SCALED features)
      'rf'       -> RandomForestClassifier (UNSCALED)
      'gbr'      -> GradientBoostingClassifier (UNSCALED)
    """
    seed = config.RANDOM_SEED
    if name == "logistic":
        return LogisticRegression(
            C=kw.get("C", 1.0), class_weight=kw.get("class_weight", None),
            max_iter=kw.get("max_iter", 2000), random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=kw.get("n_estimators", 200),
            min_samples_leaf=kw.get("min_samples_leaf", 5),
            class_weight=kw.get("class_weight", None),
            n_jobs=-1, random_state=seed)
    if name == "gbr":
        return GradientBoostingClassifier(
            n_estimators=kw.get("n_estimators", 300),
            learning_rate=kw.get("learning_rate", 0.05),
            max_depth=kw.get("max_depth", 3), random_state=seed)
    raise ValueError(f"Unknown classifier: {name}")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def average_precision(y_true, p) -> float:
    """Average Precision (AP) — primary PR-based selection metric."""
    return float(average_precision_score(y_true, p))


def brier(y_true, p) -> float:
    return float(brier_score_loss(y_true, p))


def clf_metrics_at_threshold(y_true, p, thr):
    y_pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def _model_inference_input(model, X):
    """
    Match inference input type to the estimator's fitted schema.

    - Estimator fitted with feature names:
      require a DataFrame with exactly the same names and order.
    - Estimator fitted without feature names:
      pass a finite NumPy matrix to avoid sklearn mismatch warnings.
    """
    if hasattr(model, "feature_names_in_"):
        assert isinstance(X, _pd.DataFrame), (
            "Estimator was fitted with feature names; "
            "inference input must be a DataFrame."
        )

        expected_names = [
            str(name)
            for name in model.feature_names_in_
        ]
        actual_names = [
            str(name)
            for name in X.columns
        ]

        assert actual_names == expected_names, (
            "Inference feature names/order differ from "
            "the estimator's fitted schema."
        )

        X_model = X

    else:
        if isinstance(X, _pd.DataFrame):
            X_model = X.to_numpy(
                dtype=float
            )
        else:
            X_model = np.asarray(
                X,
                dtype=float,
            )

    assert X_model.ndim == 2, (
        "Inference feature matrix must be 2-D."
    )

    if hasattr(model, "n_features_in_"):
        assert X_model.shape[1] == int(
            model.n_features_in_
        ), (
            "Inference feature count differs from "
            "model.n_features_in_."
        )

    numeric_values = (
        X_model.to_numpy(dtype=float)
        if isinstance(X_model, _pd.DataFrame)
        else np.asarray(X_model, dtype=float)
    )

    assert np.isfinite(numeric_values).all(), (
        "Inference features contain NaN or Inf."
    )

    return X_model
# --------------------------------------------------------------------------
# Calibration input: decision scores where available, else logit(clipped prob)
# --------------------------------------------------------------------------
def calibration_scores(model, X):
    """
    Return a real-valued score suitable for Platt scaling.

    If decision_function exists, use it directly.
    Otherwise convert P(class=1) to logit.
    """
    X_model = _model_inference_input(
        model,
        X,
    )

    if hasattr(model, "decision_function"):
        return np.asarray(
            model.decision_function(X_model),
            dtype=float,
        )

    proba = model.predict_proba(
        X_model
    )[:, 1]

    proba = np.clip(
        proba,
        _EPS,
        1 - _EPS,
    )

    return np.log(
        proba / (1 - proba)
    )


# --------------------------------------------------------------------------
# Platt scaling (per horizon) — regularized; requires both classes
# --------------------------------------------------------------------------
def fit_platt(scores_calib, y_calib, C=1.0):
    """
    Fit a REGULARIZED Platt logistic calibrator (C=1.0) mapping decision scores
    -> calibrated probability. Asserts both classes present in calibration labels.
    """
    y = np.asarray(y_calib)
    classes = np.unique(y)
    assert set(classes.tolist()) >= {0, 1}, \
        f"Platt calibration needs both classes; got {classes.tolist()}"
    platt = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
    platt.fit(np.asarray(scores_calib, dtype=float).reshape(-1, 1), y)
    return platt


def apply_platt(platt, scores):
    p = platt.predict_proba(np.asarray(scores, dtype=float).reshape(-1, 1))[:, 1]
    return p


# --------------------------------------------------------------------------
# Predeclared asymmetric threshold selection (row-level cost)
# --------------------------------------------------------------------------
def best_cost_threshold(y_true, p, fn_cost=FN_COST, fp_cost=FP_COST, grid=None):
    """Threshold minimizing fn_cost*FN + fp_cost*FP (row-level).
    Ascending grid + strict '<' keeps the LOWEST threshold on ties
    (predeclared conservative tie-break)."""
    p = np.asarray(p, dtype=float)
    if grid is None:
        grid = np.unique(np.concatenate([[0.0], np.sort(p), [1.0]]))
    y_true = np.asarray(y_true)
    best_t, best_c = 0.5, np.inf
    for t in grid:
        yp = (p >= t).astype(int)
        fn = int(np.sum((yp == 0) & (y_true == 1)))
        fp = int(np.sum((yp == 1) & (y_true == 0)))
        c = fn_cost * fn + fp_cost * fp
        if c < best_c:
            best_c, best_t = c, t
    return float(best_t), float(best_c)


def f1_optimal_threshold(y_true, p, grid=None):
    p = np.asarray(p, dtype=float)
    if grid is None:
        grid = np.unique(np.concatenate([[0.0], np.sort(p), [1.0]]))
    best_t, best_f = 0.5, -1.0
    for t in grid:
        f = f1_score(y_true, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t), float(best_f)


def recall_constrained_threshold(y_true, p, min_recall=0.90, grid=None):
    p = np.asarray(p, dtype=float)
    if grid is None:
        grid = np.unique(np.concatenate([[0.0], np.sort(p), [1.0]]))
    y_true = np.asarray(y_true)
    feasible = [t for t in grid
                if recall_score(y_true, (p >= t).astype(int), zero_division=0) >= min_recall]
    return float(max(feasible)) if feasible else 0.0


# --------------------------------------------------------------------------
# Cross-horizon CONSERVATIVE cumulative-max correction: P10<=P20<=P30
# --------------------------------------------------------------------------
def enforce_monotone(p10, p20, p30):
    """Row-wise non-decreasing via cumulative max (identity if already ordered)."""
    p10 = np.asarray(p10, float); p20 = np.asarray(p20, float); p30 = np.asarray(p30, float)
    p20m = np.maximum(p10, p20)
    p30m = np.maximum(p20m, p30)
    return p10, p20m, p30m


def monotonicity_violation_rate(p10, p20, p30):
    p10 = np.asarray(p10, float); p20 = np.asarray(p20, float); p30 = np.asarray(p30, float)
    viol = (p20 < p10 - 1e-9) | (p30 < p20 - 1e-9)
    return float(np.mean(viol))


# --------------------------------------------------------------------------
# Reliability bins (explicit quantile assignment with counts)
# --------------------------------------------------------------------------
def reliability_bins(y_true, p, n_bins=8):
    """
    Explicit quantile-bin reliability with per-bin sample counts.
    Returns DataFrame: bin_id, lo, hi, count, mean_pred, frac_pos.
    """
    y = np.asarray(y_true, dtype=float); p = np.asarray(p, dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)                     # drop duplicate edges robustly
    if len(edges) < 2:                           # degenerate (all equal p)
        return _pd.DataFrame(columns=["bin_id","lo","hi","count","mean_pred","frac_pos"])
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        c = int(m.sum())
        if c == 0:
            continue
        rows.append({"bin_id": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "count": c, "mean_pred": float(p[m].mean()),
                     "frac_pos": float(y[m].mean())})
    return _pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Recursive NumPy/None -> JSON-native conversion
# --------------------------------------------------------------------------
def to_jsonable(obj):
    """Recursively convert numpy scalars/arrays/None to JSON-native types."""
    if isinstance(obj, dict):
        return {to_jsonable(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj                                   # None, str, int, float, bool pass through


# --------------------------------------------------------------------------
# Joint 3-horizon calibrated inference (indexed; used by notebook AND app)
# --------------------------------------------------------------------------
def joint_calibrated_probabilities(bundles, df, thresholds=None):
    """
    JOINT three-horizon inference contract (shared by notebook & app).
    Returns dict of pandas objects indexed by df.index.
    """
    assert set(bundles.keys()) == {10, 20, 30}, \
        f"bundles must have keys exactly {{10,20,30}}, got {set(bundles.keys())}"
    idx = df.index
    n = len(df)
    raw, platt_p = {}, {}
    for h in (10, 20, 30):
        b = bundles[h]
        Xdf = b["fp"].transform(df, scaled=b["scaled"])[0]
        assert Xdf.index.equals(idx), f"h{h}: transformed X index != df.index"
        # Match the inference input type to how this estimator was fitted.
        X_model = _model_inference_input(
            b["clf"],
            Xdf,
        )

        raw[h] = b["clf"].predict_proba(
            X_model
        )[:, 1]

        scores = calibration_scores(
            b["clf"],
            X_model,
        )
        
        platt_p[h] = apply_platt(b["platt"], scores)
        assert len(raw[h]) == n and len(platt_p[h]) == n, f"h{h}: length != len(df)"
        assert np.all(np.isfinite(platt_p[h])) and \
               np.all((platt_p[h] >= 0) & (platt_p[h] <= 1)), \
               f"h{h}: Platt probs not finite/bounded"
    p10, p20, p30 = enforce_monotone(platt_p[10], platt_p[20], platt_p[30])
    mono = {10: p10, 20: p20, 30: p30}
    assert np.all(p20 >= p10 - 1e-9) and np.all(p30 >= p20 - 1e-9), \
        "post-correction ordering P10<=P20<=P30 violated"

    raw_df   = _pd.DataFrame(raw,     index=idx)[[10, 20, 30]]
    platt_df = _pd.DataFrame(platt_p, index=idx)[[10, 20, 30]]
    mono_df  = _pd.DataFrame(mono,    index=idx)[[10, 20, 30]]
    dec_df = None
    if thresholds is not None:
        dec = {h: (mono[h] >= thresholds[h]).astype(int) for h in (10, 20, 30)}
        dec_df = _pd.DataFrame(dec, index=idx)[[10, 20, 30]]
    return {"raw_proba": raw_df, "platt_proba": platt_df,
            "mono_proba": mono_df, "decisions": dec_df}


# --------------------------------------------------------------------------
# Frozen-artifact evaluation (no refit)
# --------------------------------------------------------------------------
def evaluate_dataframe(bundles, df, thresholds):
    """Evaluate a DataFrame with FROZEN bundles + FROZEN thresholds (no refit)."""
    assert set(bundles.keys()) == {10, 20, 30}, "bundles keys must be {10,20,30}"
    assert set(thresholds.keys()) == {10, 20, 30}, "thresholds keys must be {10,20,30}"
    assert all(isinstance(k, int) for k in thresholds), "threshold keys must be int"
    assert df.index.is_unique, "df.index must be unique"
    need = ["engine_id", "cycle"] + [LABEL_MAP[h] for h in (10, 20, 30)]
    for col in need:
        assert col in df.columns, f"missing column: {col}"
    assert not df.duplicated(["engine_id", "cycle"]).any(), "(engine_id,cycle) not unique"
    y = {h: df[LABEL_MAP[h]].to_numpy(int) for h in (10, 20, 30)}
    assert np.all(y[10] <= y[20]) and np.all(y[20] <= y[30]), "label nesting violated"

    out = joint_calibrated_probabilities(bundles, df, thresholds=thresholds)
    idx = df.index
    for key in ("raw_proba", "platt_proba", "mono_proba", "decisions"):
        assert out[key].index.equals(idx), f"{key} index != df.index"
        assert len(out[key]) == len(df), f"{key} length mismatch"

    tbl = _pd.DataFrame(index=idx)
    tbl["engine_id"] = df["engine_id"].to_numpy()
    tbl["cycle"]     = df["cycle"].to_numpy()
    assert np.array_equal(tbl["engine_id"].to_numpy(), df["engine_id"].to_numpy())
    assert np.array_equal(tbl["cycle"].to_numpy(), df["cycle"].to_numpy())
    for h in (10, 20, 30):
        tbl[f"y_h{h}"]     = y[h]
        tbl[f"raw_h{h}"]   = out["raw_proba"][h].to_numpy()
        tbl[f"platt_h{h}"] = out["platt_proba"][h].to_numpy()
        tbl[f"mono_h{h}"]  = out["mono_proba"][h].to_numpy()
        tbl[f"dec_h{h}"]   = out["decisions"][h].to_numpy()
    return tbl


# --------------------------------------------------------------------------
# Engine-level bootstrap (multiplicity-preserving) + metric CIs
# --------------------------------------------------------------------------
def engine_bootstrap_indices(engine_ids, B, seed):
    """
    List of B row-position arrays; resample ENGINES with replacement and
    CONCATENATE positions per sampled occurrence (multiplicity preserved).
    """
    engine_ids = np.asarray(engine_ids)
    uniq = np.unique(engine_ids)
    pos_by_engine = {e: np.where(engine_ids == e)[0] for e in uniq}
    rng = np.random.RandomState(seed)
    samples = []
    for _ in range(B):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        samples.append(np.concatenate([pos_by_engine[e] for e in drawn]))
    return samples, len(uniq)


def bootstrap_metric_ci(samples, y, p, dec, fn_cost=10, fp_cost=1, metric="AP"):
    """
    Percentile 95% CI over engine-bootstrap `samples`. Returns observed point
    estimate (on full data) + bootstrap mean + CI + valid/invalid counts +
    (for precision) count of no-predicted-positive replicates.
    """
    def _compute(yy, pp, dd):
        has_pos = yy.sum() > 0
        has_pred_pos = dd.sum() > 0
        if metric == "AP":
            return (average_precision_score(yy, pp), True) if has_pos else (None, False)
        if metric == "recall":
            return (recall_score(yy, dd, zero_division=0), True) if has_pos else (None, False)
        if metric == "f1":
            return (f1_score(yy, dd, zero_division=0), True) if has_pos else (None, False)
        if metric == "precision":
            return (precision_score(yy, dd, zero_division=0), has_pred_pos)
        if metric == "Brier":
            return (brier_score_loss(yy, pp), True)
        if metric == "norm_cost":
            fn = int(np.sum((dd == 0) & (yy == 1)))
            fp = int(np.sum((dd == 1) & (yy == 0)))
            return ((fn_cost * fn + fp_cost * fp) / len(yy), True)
        raise ValueError(metric)

    obs, obs_defined = _compute(y, p, dec)

    vals, invalid, no_pred_pos = [], 0, 0
    for pos in samples:
        v, defined = _compute(y[pos], p[pos], dec[pos])
        if metric == "precision":
            if not defined:
                no_pred_pos += 1
            vals.append(v)
        else:
            if not defined:
                invalid += 1
                continue
            vals.append(v)
    vals = np.array([v for v in vals if v is not None], dtype=float)
    ci = ((float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
          if len(vals) else (float("nan"), float("nan")))
    return {"metric": metric,
            "observed": (float(obs) if obs is not None else float("nan")),
            "observed_defined": bool(obs_defined),
            "bootstrap_mean": float(vals.mean()) if len(vals) else float("nan"),
            "ci_low": ci[0], "ci_high": ci[1],
            "n_valid": int(len(vals)), "n_invalid": int(invalid),
            "n_no_predicted_positive": int(no_pred_pos)}


def per_engine_equal_weight(pe_df, role):
    """Equal-weight mean of per-engine metrics (distinct from pooled-row)."""
    sub = pe_df[pe_df.role == role]
    return sub.groupby("horizon")[["Brier", "precision", "recall", "f1", "norm_cost"]].mean()

# --------------------------------------------------------------------------
# Engine-level bootstrap that ALSO returns drawn engine IDs (multiplicity)
# --------------------------------------------------------------------------
def engine_bootstrap_draws(engine_ids, B, seed):
    """
    Like engine_bootstrap_indices, but returns per-replicate:
        row_positions   : np.ndarray (concatenated, multiplicity preserved)
        drawn_engine_ids: np.ndarray (length = n_unique_engines; WITH duplicates)

    Uses the SAME RandomState(seed) + rng.choice(uniq, size=len(uniq)) sequence
    as engine_bootstrap_indices, so row_positions are IDENTICAL for the same
    (engine_ids, B, seed). Verified by test_bootstrap_helpers_consistent().
    """
    engine_ids = np.asarray(engine_ids)
    uniq = np.unique(engine_ids)
    pos_by_engine = {e: np.where(engine_ids == e)[0] for e in uniq}
    rng = np.random.RandomState(seed)
    draws = []
    for _ in range(B):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        row_positions = np.concatenate([pos_by_engine[e] for e in drawn])
        draws.append({"row_positions": row_positions, "drawn_engine_ids": drawn})
    return draws, len(uniq)


def test_bootstrap_helpers_consistent():
    """engine_bootstrap_draws row_positions must match engine_bootstrap_indices."""
    eng = np.repeat([1, 2, 3, 4, 5], 4)
    s_idx, n1 = engine_bootstrap_indices(eng, B=50, seed=7)
    draws, n2 = engine_bootstrap_draws(eng, B=50, seed=7)
    assert n1 == n2 == 5
    for a, d in zip(s_idx, draws):
        assert np.array_equal(a, d["row_positions"]), "helpers diverged"
        assert d["drawn_engine_ids"].size == 5           # engine-draw count
    return True


# --------------------------------------------------------------------------
# Smoke tests
# --------------------------------------------------------------------------
def test_platt_output_finite_bounded():
    rng = np.random.RandomState(0)
    n = 1000
    s = rng.randn(n)
    y = (s + rng.randn(n) * 0.5 > 0).astype(int)
    platt = fit_platt(s, y, C=1.0)
    p = apply_platt(platt, rng.randn(200))
    assert np.all(np.isfinite(p)), "Platt output not finite"
    assert np.all((p >= 0) & (p <= 1)), "Platt output outside [0,1]"
    return True


def test_platt_rejects_one_class():
    s = np.random.RandomState(1).randn(100)
    y_one_class = np.zeros(100, dtype=int)
    try:
        fit_platt(s, y_one_class, C=1.0)
        raise RuntimeError("did not reject one-class calibration target")
    except AssertionError:
        return True


def test_cumulative_max_monotone():
    rng = np.random.RandomState(2)
    n = 5000
    p10 = rng.uniform(0, 1, n); p20 = rng.uniform(0, 1, n); p30 = rng.uniform(0, 1, n)
    a, b, c = enforce_monotone(p10, p20, p30)
    assert np.all(b >= a - 1e-12) and np.all(c >= b - 1e-12), "not monotone"
    assert monotonicity_violation_rate(a, b, c) == 0.0
    o10 = np.array([0.1, 0.2]); o20 = np.array([0.3, 0.4]); o30 = np.array([0.5, 0.6])
    a2, b2, c2 = enforce_monotone(o10, o20, o30)
    assert np.allclose(a2, o10) and np.allclose(b2, o20) and np.allclose(c2, o30), \
        "should be identity when already ordered"
    return True


def test_cost_threshold_prefers_fewer_fn():
    rng = np.random.RandomState(3)
    n = 2000
    y = (rng.uniform(size=n) < 0.15).astype(int)
    p = np.where(y == 1, rng.beta(3, 2, n), rng.beta(2, 3, n))
    t_asym, _ = best_cost_threshold(y, p, fn_cost=10.0, fp_cost=1.0)
    t_sym,  _ = best_cost_threshold(y, p, fn_cost=1.0,  fp_cost=1.0)
    assert t_asym <= t_sym, (f"asymmetric threshold {t_asym:.3f} should be <= "
                             f"symmetric {t_sym:.3f} (FN-heavy => more sensitive)")
    fn_asym = int(np.sum((p >= t_asym).astype(int)[y == 1] == 0))
    fn_sym  = int(np.sum((p >= t_sym).astype(int)[y == 1] == 0))
    assert fn_asym <= fn_sym, "asymmetric cost did not reduce FNs"
    return True


def test_reliability_bins_counts():
    rng = np.random.RandomState(0)
    p = rng.uniform(0, 1, 500); y = (rng.uniform(size=500) < p).astype(int)
    tb = reliability_bins(y, p, n_bins=8)
    assert tb["count"].sum() == 500, "bin counts must sum to N"
    assert (tb["count"] > 0).all(), "no empty populated bins allowed"
    return True


def test_to_jsonable():
    import json as _json
    obj = {"a": np.int64(3), "b": np.float64(1.5), "c": np.bool_(True),
           "d": None, "e": np.array([1, 2]), "f": "x"}
    out = to_jsonable(obj)
    _json.dumps(out)   # must not raise
    assert out["a"] == 3 and out["b"] == 1.5 and out["c"] is True and out["d"] is None
    return True


def test_bootstrap_reproducible_indices():
    eng = np.repeat([1, 2, 3, 4, 5], 4)
    s1, _ = engine_bootstrap_indices(eng, B=100, seed=7)
    s2, _ = engine_bootstrap_indices(eng, B=100, seed=7)
    assert all(np.array_equal(a, b) for a, b in zip(s1, s2)), "same seed must reproduce"
    return True


def test_engine_bootstrap_multiplicity():
    eng = np.array([1, 1, 2, 2, 3, 3])
    samples, n = engine_bootstrap_indices(eng, B=50, seed=0)
    assert all(len(s) == 6 for s in samples) and n == 3, "multiplicity not preserved"
    return True


def test_bootstrap_metric_ci_keys():
    eng = np.repeat([1, 2, 3], 10)
    samples, _ = engine_bootstrap_indices(eng, B=20, seed=0)
    y = np.tile([0, 1], 15); p = np.linspace(0, 1, 30); d = (p >= 0.5).astype(int)
    r = bootstrap_metric_ci(samples, y, p, d, metric="AP")
    need = {"observed", "bootstrap_mean", "ci_low", "ci_high", "n_valid",
            "n_invalid", "n_no_predicted_positive"}
    assert need.issubset(r.keys()), f"missing keys: {need - set(r.keys())}"
    return True


def run_all_tests(verbose=True):
    # Explicit module-level registry (the SAME list run_all_tests iterates).
    CLASSIFICATION_TESTS = [
        ("platt output finite & in [0,1]",        test_platt_output_finite_bounded),
        ("platt rejects one-class target",         test_platt_rejects_one_class),
        ("cumulative-max monotone (+identity)",    test_cumulative_max_monotone),
        ("cost-threshold prefers fewer FN (10:1)", test_cost_threshold_prefers_fewer_fn),
        ("reliability bins counts",                test_reliability_bins_counts),
        ("to_jsonable conversion",                 test_to_jsonable),
        ("bootstrap reproducible indices",         test_bootstrap_reproducible_indices),
        ("engine bootstrap multiplicity",          test_engine_bootstrap_multiplicity),
        ("bootstrap_metric_ci flat keys",          test_bootstrap_metric_ci_keys),
        ("bootstrap helpers consistent",           test_bootstrap_helpers_consistent),  # NEW
    ]
    results, ok = {}, True
    for name, fn in CLASSIFICATION_TESTS:
        try:
            fn(); results[name] = "PASS"
        except AssertionError as e:
            results[name] = f"FAIL: {e}"; ok = False
        except Exception as e:
            results[name] = f"ERROR: {type(e).__name__}: {e}"; ok = False
    if verbose:
        print(" classification module smoke tests ")
        print(f"Registered tests = {len(CLASSIFICATION_TESTS)}")
        for k, v in results.items():
            print(f"  [{'OK' if v == 'PASS' else 'XX'}] {k}: {v}")
        print(f"PASS = {sum(1 for v in results.values() if v=='PASS')}/{len(CLASSIFICATION_TESTS)}")
    return results, ok


if __name__ == "__main__":
    print(f"Predeclared row-level threshold cost FN:FP = {FN_COST}:{FP_COST} "
          f"(sensitivity ratios: {SENSITIVITY_RATIOS})")
    print("NOTE: this is a ROW-LEVEL threshold cost, NOT a universal maintenance "
          "cost or the final engine-level alert policy (persistence, lead time, "
          "miss rate, late-warning delay, early-warning burden added later).\n")
    _, ok = run_all_tests(verbose=True)
    print("\nALL PASSED" if ok else "\nSOME TESTS FAILED")

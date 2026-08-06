"""
Jet Engine Hospital - Anomaly Detection Module
Unsupervised health-manifold detectors with DEPLOYMENT-SAFE score normalization.

Four detectors (all oriented so LARGER = MORE ABNORMAL):
  * IsolationForest        raw = -score_samples(X)
  * LOF (novelty=True)     raw = -score_samples(X)
  * OneClassSVM (RBF)      raw = -score_samples(X)   [fixed convention]
  * PCA reconstruction     raw = mean squared reconstruction error (scaled space)

Fixed TRAIN-reference empirical CDF (right-sided):
  percentile(s) = searchsorted(sorted_ref, s, side="right") / n_ref_rows
  below min -> 0.0 ; at/above max -> 1.0. NOT a probability.

Near-constant filter statistic: POPULATION standard deviation (ddof=0) <= std_threshold.

Role integrity: anomaly-fit / anomaly-reference / VAL-TUNE / VAL-CALIB all share
the SAME official-train numeric namespace, so numeric-ID disjointness is valid
among them. Official TEST lives in a DIFFERENT source file; it is excluded by
SOURCE-ROLE assertion, never by comparing its numeric IDs against official-train.

Module tests are SYNTHETIC implementation tests only.
"""
from __future__ import annotations
import math
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PCTL_BELOW_MIN = 0.0
PCTL_AT_OR_ABOVE_MAX = 1.0
ALLOWED_REFERENCE_ROLE = "anomaly_reference_train"
ALLOWED_FIT_ROLE = "anomaly_fit_train"
OFFICIAL_TRAIN_SOURCE = "train"

def _finite_2d(X, name="X"):
    X = np.asarray(X, dtype=float)
    assert X.ndim == 2, f"{name} must be 2-D, got {X.shape}"
    assert X.shape[0] > 0 and X.shape[1] > 0, f"{name} empty dim {X.shape}"
    assert np.all(np.isfinite(X)), f"{name} contains non-finite values"
    return X


def _finite_1d(a, name="a"):
    a = np.asarray(a, dtype=float).ravel()
    assert a.size > 0, f"{name} is empty"
    assert np.all(np.isfinite(a)), f"{name} contains non-finite values"
    return a

def make_anomaly_train_subdivision(train_engine_ids, seed, reference_fraction=0.25):
    ids = np.array(sorted(int(e) for e in train_engine_ids))
    n = ids.size
    assert n >= 2, "need >=2 train engines"
    assert np.unique(ids).size == n, "duplicate train engine ids"
    assert 0.0 < reference_fraction < 1.0, f"reference_fraction in (0,1): {reference_fraction}"
    n_ref = int(math.ceil(reference_fraction * n))
    assert 1 <= n_ref < n, f"degenerate split n_ref={n_ref}, n={n}"
    rng = np.random.RandomState(seed)
    perm = rng.permutation(ids)
    ref = sorted(int(e) for e in perm[:n_ref])
    fit = sorted(int(e) for e in perm[n_ref:])
    assert set(fit).isdisjoint(ref)
    assert set(fit) | set(ref) == set(int(e) for e in ids)
    return {"seed": int(seed), "reference_fraction": float(reference_fraction),
            "rounding_rule": "n_reference = ceil(reference_fraction * n_train_engines)",
            "n_train_engines": int(n), "n_reference": int(len(ref)),
            "n_fit": int(len(fit)), "anomaly_fit_ids": fit,
            "anomaly_reference_ids": ref}


def validate_same_source_roles(fit_ids, reference_ids, allowed_train_ids,
                               val_tune_ids=(), val_calib_ids=(),
                               source_role=OFFICIAL_TRAIN_SOURCE,
                               declared_source=OFFICIAL_TRAIN_SOURCE):
    """
    Validate the four official-train-sourced roles by NUMERIC disjointness
    (valid because they share ONE source file / namespace).

      * declared_source must equal source_role (the official-train namespace);
      * fit ∩ reference == ∅;
      * fit ∪ reference == allowed TRAIN set;
      * val-tune / val-calib IDs absent from fit and reference.

    Official TEST is NOT passed here; it is excluded by data-source role elsewhere,
    never by numeric-ID comparison against official-train.
    """
    assert declared_source == source_role, (
        f"source-role mismatch: declared={declared_source!r} != {source_role!r}")
    fit = set(int(e) for e in fit_ids)
    ref = set(int(e) for e in reference_ids)
    allowed = set(int(e) for e in allowed_train_ids)
    vt = set(int(e) for e in val_tune_ids)
    vc = set(int(e) for e in val_calib_ids)
    assert fit.isdisjoint(ref), "fit/reference overlap"
    assert (fit | ref) == allowed, "fit ∪ reference != allowed TRAIN set"
    assert fit.isdisjoint(vt), "fit contains a VAL-TUNE engine"
    assert ref.isdisjoint(vt), "reference contains a VAL-TUNE engine"
    assert fit.isdisjoint(vc), "fit contains a VAL-CALIB engine"
    assert ref.isdisjoint(vc), "reference contains a VAL-CALIB engine"
    return True


def assert_test_excluded_by_source(candidate_source_role):
    """Official TEST must be excluded by SOURCE ROLE, not numeric IDs."""
    assert candidate_source_role == OFFICIAL_TRAIN_SOURCE, (
        f"anomaly roles must originate from official-train source "
        f"{OFFICIAL_TRAIN_SOURCE!r}; got {candidate_source_role!r} "
        f"(official-test excluded by source role)")
    return True


# ==
# 3 · Fixed empirical reference CDF
# ==
def build_reference_cdf(ref_scores, provenance=None):
    s = _finite_1d(ref_scores, "ref_scores")
    if provenance is not None:
        role = provenance.get("source_role")
        assert role == ALLOWED_REFERENCE_ROLE, (
            f"reference CDF must be built from {ALLOWED_REFERENCE_ROLE!r}, got {role!r}")
    sorted_scores = np.sort(s)
    return {"sorted_scores": sorted_scores, "n_ref_rows": int(sorted_scores.size),
            "tie_rule": "searchsorted(side='right')",
            "orientation": "larger=more_abnormal", "provenance": provenance}


def reference_percentile(cdf, scores):
    sorted_scores = cdf["sorted_scores"]; n = cdf["n_ref_rows"]
    assert n > 0, "empty reference CDF"
    s = np.asarray(scores, dtype=float).ravel()
    assert s.size > 0, "empty score input"
    assert np.all(np.isfinite(s)), "reference_percentile rejects non-finite scores"
    return (np.searchsorted(sorted_scores, s, side="right") / n).astype(float)


# ==
# 4 · Engine-balanced reference sample (sensitivity CDF)
# ==
def engine_balanced_reference_indices(engine_ids, seed):
    eng = np.asarray(engine_ids)
    assert eng.size > 0, "empty engine_ids"
    uniq = np.unique(eng)
    avail = {int(e): int(np.sum(eng == e)) for e in uniq}
    k = min(avail.values())
    assert k >= 1, "some reference engine contributes zero rows"
    rng = np.random.RandomState(seed)
    picked, sel_counts = [], {}
    for e in uniq:
        pos = np.where(eng == e)[0]
        sel = rng.choice(pos, size=k, replace=False)
        picked.append(np.sort(sel)); sel_counts[int(e)] = int(k)
    selected = np.sort(np.concatenate(picked))
    return {"positions": selected, "k": int(k), "available_counts": avail,
            "selected_counts": sel_counts, "seed": int(seed),
            "reference_engine_ids": [int(e) for e in uniq]}


# ==
# 5 · OC-SVM capped engine-balanced subsample (quota redistribution)
# ==
def ocsvm_balanced_subsample(engine_ids, cap, seed):
    eng = np.asarray(engine_ids)
    n_rows = eng.size
    assert cap >= 1, "cap must be >= 1"
    uniq = np.unique(eng)
    avail = {int(e): int(np.sum(eng == e)) for e in uniq}
    total = int(sum(avail.values()))
    if n_rows <= cap:
        return (np.arange(n_rows),
                {"used_all": True, "n_selected": int(n_rows), "cap": int(cap),
                 "n_engines": int(uniq.size), "per_engine": dict(avail)})
    target = min(cap, total)
    alloc = {int(e): 0 for e in uniq}
    remaining = {int(e): avail[int(e)] for e in uniq}
    to_assign = target
    while to_assign > 0:
        active = [int(e) for e in uniq if remaining[int(e)] > 0]
        assert active, "capacity exhausted before cap"
        base = to_assign // len(active)
        rem = to_assign - base * len(active)
        assigned = 0
        for i, e in enumerate(active):
            give = min(base + (1 if i < rem else 0), remaining[e])
            alloc[e] += give; remaining[e] -= give; assigned += give
        to_assign -= assigned
        if base == 0 and assigned == 0:
            break
    assert sum(alloc.values()) == target, (sum(alloc.values()), target)
    rng = np.random.RandomState(seed)
    picked = []
    for e in uniq:
        take = alloc[int(e)]
        if take == 0:
            continue
        pos = np.where(eng == int(e))[0]
        assert take <= pos.size
        sel = rng.choice(pos, size=take, replace=False) if take < pos.size else pos
        picked.append(np.sort(sel))
    selected = np.sort(np.concatenate(picked))
    assert selected.size == target
    assert np.unique(selected).size == selected.size
    return selected, {"used_all": False, "n_selected": int(selected.size),
                      "cap": int(cap), "n_engines": int(uniq.size),
                      "per_engine": {int(e): int(alloc[int(e)]) for e in uniq}}


def _validate_max_samples(ms):
    assert not isinstance(ms, bool), "max_samples must not be bool"
    if isinstance(ms, str):
        assert ms == "auto", f"bad max_samples str {ms!r}"
    elif isinstance(ms, (int, np.integer)) and not isinstance(ms, bool):
        assert int(ms) >= 1, f"integer max_samples must be >=1, got {ms}"
    elif isinstance(ms, (float, np.floating)):
        assert 0.0 < float(ms) <= 1.0, f"float max_samples must be in (0,1], got {ms}"
    else:
        raise AssertionError(f"unsupported max_samples type {type(ms)}")


def make_detector(name, seed, n_healthy_fit_rows=None, **kw):
    if name == "iforest":
        ms = kw.get("max_samples", "auto"); _validate_max_samples(ms)
        n_estimators = int(kw.get("n_estimators", 200)); assert n_estimators > 0
        model = IsolationForest(n_estimators=n_estimators, max_samples=ms,
                                contamination="auto", random_state=seed, n_jobs=-1)
        meta = {"sklearn_class": "IsolationForest", "score_method": "score_samples",
                "orientation_sign": -1,
                "params": {"n_estimators": n_estimators, "max_samples": ms,
                           "contamination": "auto"}}
        return model, meta
    if name == "lof":
        n_neighbors = int(kw.get("n_neighbors", 20)); assert n_neighbors >= 1
        assert n_healthy_fit_rows is not None, "LOF requires n_healthy_fit_rows"
        assert n_neighbors < n_healthy_fit_rows, (
            f"LOF n_neighbors={n_neighbors} must be < n_healthy_fit_rows={n_healthy_fit_rows}")
        model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, n_jobs=-1)
        meta = {"sklearn_class": "LocalOutlierFactor", "score_method": "score_samples",
                "orientation_sign": -1,
                "params": {"n_neighbors": n_neighbors, "novelty": True}}
        return model, meta
    if name == "ocsvm":
        nu = float(kw.get("nu", 0.05)); assert 0.0 < nu <= 1.0, f"nu out of range {nu}"
        gamma = kw.get("gamma", "scale")
        assert gamma in ("scale", "auto") or (isinstance(gamma, (int, float))
               and not isinstance(gamma, bool) and gamma > 0), f"bad gamma {gamma}"
        model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
        meta = {"sklearn_class": "OneClassSVM", "score_method": "score_samples",
                "orientation_sign": -1, "params": {"kernel": "rbf", "nu": nu, "gamma": gamma}}
        return model, meta
    raise ValueError(f"unknown detector {name}")


def oriented_raw_score(name, model, X):
    X = _finite_2d(X, "X")
    if name in ("iforest", "lof", "ocsvm"):
        s = -np.asarray(model.score_samples(X), dtype=float)
    else:
        raise ValueError(f"unknown detector {name}")
    assert s.shape == (X.shape[0],)
    assert np.all(np.isfinite(s)), f"{name} produced non-finite score"
    return s


def fit_pca_evr(X_fit, evr_target, seed):
    X_fit = _finite_2d(X_fit, "X_fit")
    assert 0.0 < evr_target <= 1.0, f"evr_target in (0,1]: {evr_target}"
    n_samples, n_features = X_fit.shape
    max_k = min(n_samples, n_features)
    full = PCA(svd_solver="full", random_state=seed).fit(X_fit)
    cum = np.cumsum(full.explained_variance_ratio_)     # length == max_k
    k = int(np.searchsorted(cum, evr_target) + 1)
    k = int(min(max(k, 1), max_k))
    pk = PCA(n_components=k, svd_solver="full", random_state=seed).fit(X_fit)
    achieved = float(cum[k - 1])
    assert 1 <= k <= max_k, (k, max_k)
    return pk, k, achieved, cum.astype(float)


def pca_reconstruction_score(pca, X):
    X = _finite_2d(X, "X")
    Xhat = pca.inverse_transform(pca.transform(X))
    mse = np.mean((X - Xhat) ** 2, axis=1)
    assert mse.shape == (X.shape[0],)
    assert np.all(mse >= -1e-12) and np.all(np.isfinite(mse))
    return np.clip(mse, 0.0, None)


def pca_feature_contributions(pca, X):
    X = _finite_2d(X, "X")
    Xhat = pca.inverse_transform(pca.transform(X))
    contrib = np.mean((X - Xhat) ** 2, axis=0)
    assert contrib.shape == (X.shape[1],)
    assert np.all(np.isfinite(contrib))
    return contrib

class AnomalyPreprocessor:
    """Fitted on HEALTHY anomaly-fit TRAIN rows. Enforceable transform schema."""
    def __init__(self, std_threshold=1e-6):
        self.std_threshold = float(std_threshold)
        self.ddof = 0                        # POPULATION std
        self.fitted_ = False

    def fit(self, X_healthy_fit, feature_names, fit_engine_ids,
            healthy_boundary, source_role=ALLOWED_FIT_ROLE):
        assert source_role == ALLOWED_FIT_ROLE, (
            f"preprocessor must be fit on {ALLOWED_FIT_ROLE!r}, got {source_role!r}")
        X = _finite_2d(X_healthy_fit, "X_healthy_fit")
        feature_names = list(feature_names)
        assert X.shape[1] == len(feature_names), "X cols != feature_names"
        assert len(set(feature_names)) == len(feature_names), "duplicate feature names"
        fit_ids = [int(e) for e in fit_engine_ids]
        assert len(fit_ids) >= 1 and len(set(fit_ids)) == len(fit_ids), \
            "fit_engine_ids must be non-empty and unique"
        assert np.isfinite(float(healthy_boundary)), "healthy_boundary must be finite"
        std = X.std(axis=0, ddof=self.ddof)
        keep_mask = std > self.std_threshold
        self.input_feature_order_ = feature_names          # exact fitted schema
        self.expected_input_set_ = set(feature_names)
        self.keep_ = [feature_names[i] for i in range(len(feature_names)) if keep_mask[i]]
        self.drop_ = [feature_names[i] for i in range(len(feature_names)) if not keep_mask[i]]
        assert len(self.keep_) >= 1, "all features near-constant; nothing retained"
        self.std_by_feature_ = {feature_names[i]: float(std[i])
                                for i in range(len(feature_names))}
        keep_idx = [i for i in range(len(feature_names)) if keep_mask[i]]
        self.scaler_ = StandardScaler().fit(X[:, keep_idx])
        self.n_fit_rows_ = int(X.shape[0])
        self.n_input_features_ = int(len(feature_names))
        self.fit_engine_ids_ = fit_ids
        self.healthy_boundary_ = float(healthy_boundary)
        self.source_role_ = source_role
        self.stat_ = "population_std(ddof=0) <= std_threshold => dropped"
        self.fitted_ = True
        return self

    def transform(self, X, feature_names):
        assert self.fitted_, "not fitted"
        feature_names = list(feature_names)
        X_arr = np.asarray(X, dtype=float)
        assert X_arr.ndim == 2, f"X must be 2-D, got {X_arr.shape}"
        assert X_arr.shape[0] > 0, "X has zero rows"
        assert X_arr.shape[1] == len(feature_names), (
            f"X cols ({X_arr.shape[1]}) != len(feature_names) ({len(feature_names)})")
        assert len(set(feature_names)) == len(feature_names), "duplicate names in input"
        got = set(feature_names)
        missing = self.expected_input_set_ - got
        extra = got - self.expected_input_set_
        assert not missing, f"transform missing expected fitted columns: {sorted(missing)}"
        assert not extra, f"transform has unexpected columns: {sorted(extra)}"
        assert np.all(np.isfinite(X_arr)), "transform input contains non-finite values"
        order_idx = [feature_names.index(c) for c in self.input_feature_order_]
        X_ord = X_arr[:, order_idx]
        keep_idx = [self.input_feature_order_.index(c) for c in self.keep_]
        Xk = _finite_2d(X_ord[:, keep_idx], "X[keep]")
        return self.scaler_.transform(Xk)

    def provenance(self):
        assert self.fitted_
        return {"source_role": self.source_role_,
                "healthy_boundary": self.healthy_boundary_,
                "std_threshold": self.std_threshold, "ddof": self.ddof,
                "stat_rule": self.stat_,
                "input_feature_order": self.input_feature_order_,
                "n_input_features": self.n_input_features_,
                "retained": self.keep_, "dropped": self.drop_,
                "std_by_feature": self.std_by_feature_,
                "n_fit_rows": self.n_fit_rows_,
                "fit_engine_ids": self.fit_engine_ids_,
                "scaler": {"class": "StandardScaler",
                           "mean_": [float(v) for v in self.scaler_.mean_],
                           "scale_": [float(v) for v in self.scaler_.scale_]}}

def _roc_auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); s = _finite_1d(s, "score")
    assert y.shape[0] == s.shape[0], "y/score length mismatch"
    assert y.min() != y.max(), "ROC-AUC requires both classes"
    return float(roc_auc_score(y, s))


def per_engine_auc(engine_ids, y, s):
    eng = np.asarray(engine_ids); y = np.asarray(y); s = _finite_1d(s, "score")
    assert eng.shape[0] == y.shape[0] == s.shape[0]
    auc, sup, unsup = {}, [], []
    for e in np.unique(eng):
        m = eng == e; ye = y[m]
        if ye.min() != ye.max():
            auc[int(e)] = _roc_auc(ye, s[m]); sup.append(int(e))
        else:
            unsup.append(int(e))
    return auc, sorted(sup), sorted(unsup)


def anomaly_engine_bootstrap(engine_ids, y, s, draws):
    """Pooled-row ROC-AUC under shared engine draws; invalid replicates excluded (not 0)."""
    y = np.asarray(y); s = _finite_1d(s, "score")
    assert len(engine_ids) == y.shape[0] == s.shape[0], "length mismatch"
    values, valid_flags = [], []
    for dr in draws:
        pos = dr["row_positions"]; yy = y[pos]
        if yy.min() == yy.max():
            values.append(np.nan); valid_flags.append(False); continue
        values.append(_roc_auc(yy, s[pos])); valid_flags.append(True)
    values = np.asarray(values, float); valid_flags = np.asarray(valid_flags, bool)
    vv = values[valid_flags]
    ci = ((float(np.quantile(vv, 0.025)), float(np.quantile(vv, 0.975)))
          if vv.size else (float("nan"), float("nan")))
    return {"boot_mean": float(vv.mean()) if vv.size else float("nan"),
            "ci_low": ci[0], "ci_high": ci[1],
            "n_valid": int(vv.size), "n_invalid": int((~valid_flags).sum()),
            "B": int(len(draws)),
            "replicate_values": values, "valid_flags": valid_flags}

def to_jsonable(obj):
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
    return obj

def test_subdivision_ceil_and_integrity():
    sub = make_anomaly_train_subdivision(range(1, 71), seed=42, reference_fraction=0.25)
    assert sub["n_reference"] == math.ceil(0.25 * 70) == 18
    assert sub["n_fit"] == 52
    assert set(sub["anomaly_fit_ids"]).isdisjoint(sub["anomaly_reference_ids"])
    assert set(sub["anomaly_fit_ids"]) | set(sub["anomaly_reference_ids"]) == set(range(1, 71))
    s2 = make_anomaly_train_subdivision(range(1, 11), seed=1, reference_fraction=0.25)
    assert s2["n_reference"] == 3 and s2["n_fit"] == 7
    # degenerate fraction rejected
    try:
        make_anomaly_train_subdivision(range(1, 5), seed=0, reference_fraction=1.0)
        raise RuntimeError("degenerate fraction not rejected")
    except AssertionError:
        pass
    return True


def test_subdivision_reproducible():
    a = make_anomaly_train_subdivision(range(1, 71), seed=42)
    b = make_anomaly_train_subdivision(range(1, 71), seed=42)
    assert a["anomaly_fit_ids"] == b["anomaly_fit_ids"]
    assert a["anomaly_reference_ids"] == b["anomaly_reference_ids"]
    c = make_anomaly_train_subdivision(range(1, 71), seed=43)
    assert c["anomaly_reference_ids"] != a["anomaly_reference_ids"]
    return True


def test_same_source_role_validation():
    allowed = list(range(1, 71))
    sub = make_anomaly_train_subdivision(allowed, seed=42)
    fit, ref = sub["anomaly_fit_ids"], sub["anomaly_reference_ids"]
    # partition allowed into pretend val roles for the test by removing some?
    # Here val ids are DISTINCT numeric ids in the same namespace (e.g., held out).
    # Use a scenario: allowed excludes 71..90 which are val engines.
    allowed2 = list(range(1, 51))
    sub2 = make_anomaly_train_subdivision(allowed2, seed=42)
    f2, r2 = sub2["anomaly_fit_ids"], sub2["anomaly_reference_ids"]
    vt = list(range(51, 71)); vc = list(range(71, 81))
    assert validate_same_source_roles(f2, r2, allowed2, vt, vc)
    # overlap fit/reference
    try:
        validate_same_source_roles(f2 + [r2[0]], r2, allowed2, vt, vc)
        raise RuntimeError("overlap not caught")
    except AssertionError: pass
    # missing a TRAIN engine
    try:
        validate_same_source_roles(f2[:-1], r2, allowed2, vt, vc)
        raise RuntimeError("missing not caught")
    except AssertionError: pass
    # a VAL-TUNE engine present in fit
    try:
        validate_same_source_roles(f2 + [vt[0]], r2, allowed2 + [vt[0]], vt, vc)
        raise RuntimeError("val-tune leak not caught")
    except AssertionError: pass
    # a VAL-CALIB engine present in reference
    try:
        validate_same_source_roles(f2, r2 + [vc[0]], allowed2 + [vc[0]], vt, vc)
        raise RuntimeError("val-calib leak not caught")
    except AssertionError: pass
    # source-role mismatch
    try:
        validate_same_source_roles(f2, r2, allowed2, vt, vc,
                                   source_role="train", declared_source="test")
        raise RuntimeError("source mismatch not caught")
    except AssertionError: pass
    return True


def test_test_excluded_by_source():
    assert assert_test_excluded_by_source("train")
    try:
        assert_test_excluded_by_source("test")
        raise RuntimeError("test source not rejected")
    except AssertionError:
        return True


def test_cdf_boundary_and_ties():
    cdf = build_reference_cdf([1.0, 2.0, 2.0, 3.0, 4.0])
    assert reference_percentile(cdf, [0.5])[0] == 0.0
    assert reference_percentile(cdf, [4.0])[0] == 1.0
    assert reference_percentile(cdf, [99.0])[0] == 1.0
    assert abs(reference_percentile(cdf, [2.0])[0] - 0.6) < 1e-12
    assert abs(reference_percentile(cdf, [2.5])[0] - 0.6) < 1e-12
    return True


def test_cdf_rejects_empty_and_nonfinite():
    try:
        build_reference_cdf([]); raise RuntimeError("empty not rejected")
    except AssertionError: pass
    cdf = build_reference_cdf([1.0, 2.0, 3.0])
    try:
        reference_percentile(cdf, [np.nan]); raise RuntimeError("NaN not rejected")
    except AssertionError: pass
    try:
        reference_percentile(cdf, [np.inf]); raise RuntimeError("Inf not rejected")
    except AssertionError: pass
    return True


def test_cdf_provenance_role_enforced():
    build_reference_cdf([1.0, 2.0, 3.0],
                        provenance={"source_role": ALLOWED_REFERENCE_ROLE})
    try:
        build_reference_cdf([1.0, 2.0, 3.0], provenance={"source_role": "val_tune"})
        raise RuntimeError("forbidden role not rejected")
    except AssertionError:
        return True


def test_percentile_bounds_finite():
    rng = np.random.RandomState(0)
    cdf = build_reference_cdf(rng.randn(500))
    p = reference_percentile(cdf, rng.randn(300) * 5)
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
    return True


def test_batch_composition_invariance():
    rng = np.random.RandomState(1)
    cdf = build_reference_cdf(rng.randn(400))
    fixed = np.array([0.37])
    p_alone = reference_percentile(cdf, fixed)[0]
    batch = np.concatenate([fixed, rng.randn(250) * 3])
    assert p_alone == reference_percentile(cdf, batch)[0]
    return True


def test_pooled_cdf_reproducible():
    rng = np.random.RandomState(11)
    v = rng.randn(600)
    c1 = build_reference_cdf(v); c2 = build_reference_cdf(v.copy())
    assert np.array_equal(c1["sorted_scores"], c2["sorted_scores"])
    q = rng.randn(50) * 4
    assert np.array_equal(reference_percentile(c1, q), reference_percentile(c2, q))
    return True


def test_engine_balanced_reference_reproducible():
    eng = np.repeat([1, 2, 3], [5, 8, 6])
    r1 = engine_balanced_reference_indices(eng, seed=7)
    r2 = engine_balanced_reference_indices(eng, seed=7)
    assert r1["k"] == 5 and np.array_equal(r1["positions"], r2["positions"])
    assert r1["selected_counts"] == {1: 5, 2: 5, 3: 5}
    assert r1["available_counts"] == {1: 5, 2: 8, 3: 6}
    assert r1["positions"].size == 15
    return True


def test_ocsvm_subsample_even_and_reproducible():
    eng = np.repeat([1, 2, 3, 4], 6000)
    sel, info = ocsvm_balanced_subsample(eng, cap=20000, seed=0)
    assert not info["used_all"] and sel.size == 20000
    assert all(v == 5000 for v in info["per_engine"].values())
    sel2, _ = ocsvm_balanced_subsample(eng, cap=20000, seed=0)
    assert np.array_equal(sel, sel2)
    assert np.unique(sel).size == sel.size
    eng_small = np.repeat([1, 2], 100)
    sel3, info3 = ocsvm_balanced_subsample(eng_small, cap=20000, seed=0)
    assert info3["used_all"] and sel3.size == 200
    return True


def test_ocsvm_subsample_short_engine_redistribution():
    eng = np.concatenate([np.repeat(1, 100),
                          np.repeat(2, 10000), np.repeat(3, 10000), np.repeat(4, 10000)])
    cap = 20000
    sel, info = ocsvm_balanced_subsample(eng, cap=cap, seed=0)
    assert sel.size == cap
    pe = info["per_engine"]
    assert pe[1] <= 100
    for e in (2, 3, 4):
        assert pe[e] <= 10000
    assert sum(pe.values()) == cap
    assert np.unique(sel).size == sel.size
    assert sel.min() >= 0 and sel.max() < eng.size
    sel2, info2 = ocsvm_balanced_subsample(eng, cap=cap, seed=0)
    assert np.array_equal(sel, sel2) and info["per_engine"] == info2["per_engine"]
    return True


def test_preprocessor_contract_schema():
    rng = np.random.RandomState(2)
    X = np.column_stack([rng.randn(200), np.full(200, 3.0), rng.randn(200) * 2])
    names = ["a", "const", "c"]
    pp = AnomalyPreprocessor(std_threshold=1e-6).fit(
        X, names, fit_engine_ids=[1, 2, 3], healthy_boundary=125)
    assert pp.keep_ == ["a", "c"] and pp.drop_ == ["const"]
    # (1) reordered EXACT schema succeeds
    Xt = pp.transform(X[:, [2, 1, 0]], ["c", "const", "a"])
    assert Xt.shape == (200, 2)
    # (2) missing retained column fails
    try:
        pp.transform(X[:, [1, 2]], ["const", "c"]); raise RuntimeError("missing retained not caught")
    except AssertionError: pass
    # (3) missing dropped-but-expected column fails
    try:
        pp.transform(X[:, [0, 2]], ["a", "c"]); raise RuntimeError("missing dropped-expected not caught")
    except AssertionError: pass
    # (4) unexpected extra column fails
    Xe = np.column_stack([X, rng.randn(200)])
    try:
        pp.transform(Xe, names + ["extra"]); raise RuntimeError("extra not caught")
    except AssertionError: pass
    # (5) duplicate names fail
    try:
        pp.transform(X[:, [0, 0, 2]], ["a", "a", "c"]); raise RuntimeError("dup not caught")
    except AssertionError: pass
    # (6) non-finite fails
    Xbad = X.copy(); Xbad[0, 0] = np.inf
    try:
        pp.transform(Xbad, names); raise RuntimeError("nonfinite not caught")
    except AssertionError: pass
    return True


def test_preprocessor_fit_provenance_guards():
    rng = np.random.RandomState(21)
    X = rng.randn(30, 2)
    # wrong source role
    try:
        AnomalyPreprocessor().fit(X, ["a", "b"], [1], 125, source_role="val_tune")
        raise RuntimeError("bad source role not caught")
    except AssertionError: pass
    # empty fit ids
    try:
        AnomalyPreprocessor().fit(X, ["a", "b"], [], 125)
        raise RuntimeError("empty ids not caught")
    except AssertionError: pass
    # duplicate fit ids
    try:
        AnomalyPreprocessor().fit(X, ["a", "b"], [1, 1], 125)
        raise RuntimeError("dup ids not caught")
    except AssertionError: pass
    # non-finite boundary
    try:
        AnomalyPreprocessor().fit(X, ["a", "b"], [1], float("inf"))
        raise RuntimeError("bad boundary not caught")
    except AssertionError: pass
    ok = AnomalyPreprocessor().fit(X, ["a", "b"], [1, 2], 125)
    prov = ok.provenance()
    assert prov["ddof"] == 0 and prov["n_input_features"] == 2
    assert "mean_" in prov["scaler"] and "scale_" in prov["scaler"]
    return True


def test_pca_recon_nonneg_monotone_and_contrib():
    rng = np.random.RandomState(3)
    Xf = rng.randn(300, 6) @ rng.randn(6, 6)
    errs = []
    for k in (1, 2, 3, 4, 5, 6):
        p = PCA(n_components=k, svd_solver="full", random_state=0).fit(Xf)
        errs.append(float(pca_reconstruction_score(p, Xf).mean()))
    e = np.array(errs)
    assert np.all(e >= -1e-12)
    assert np.all(np.diff(e) <= 1e-9)
    p = PCA(n_components=3, svd_solver="full", random_state=0).fit(Xf)
    assert abs(float(pca_reconstruction_score(p, Xf).mean())
               - float(pca_feature_contributions(p, Xf).mean())) < 1e-9
    return True


def test_fit_pca_evr_selection():
    rng = np.random.RandomState(9)
    Xf = rng.randn(400, 6) @ rng.randn(6, 6)
    p, k, achieved, cum = fit_pca_evr(Xf, evr_target=0.90, seed=0)
    assert 1 <= k <= 6 and achieved >= 0.90 - 1e-9
    assert cum.shape == (6,) and abs(cum[-1] - 1.0) < 1e-6
    return True


def test_fit_pca_evr_wide_matrix():
    # n_samples < n_features: max_k must be n_samples
    rng = np.random.RandomState(13)
    Xf = rng.randn(5, 12)                    # 5 samples, 12 features
    p, k, achieved, cum = fit_pca_evr(Xf, evr_target=0.95, seed=0)
    assert k <= min(5, 12) == 5, (k,)
    assert cum.shape[0] == min(5, 12)
    assert pca_reconstruction_score(p, Xf).shape == (5,)
    return True


def _orientation_case(name, seed, **kw):
    rng = np.random.RandomState(seed)
    healthy_fit = rng.randn(500, 4)
    healthy_holdout = rng.randn(200, 4)
    abn_holdout = rng.randn(120, 4) * 0.5 + np.array([8, 8, 8, 8])
    model, meta = make_detector(name, seed=0, n_healthy_fit_rows=healthy_fit.shape[0], **kw)
    model.fit(healthy_fit)
    s_h = oriented_raw_score(name, model, healthy_holdout)
    s_a = oriented_raw_score(name, model, abn_holdout)
    assert meta["orientation_sign"] == -1
    assert np.median(s_a) > np.median(s_h) + 1e-3, \
        f"{name}: median_abn={np.median(s_a):.4f} !> median_healthy_holdout={np.median(s_h):.4f}"
    return True


def test_orientation_iforest(): return _orientation_case("iforest", 4, n_estimators=200)
def test_orientation_lof():     return _orientation_case("lof", 5, n_neighbors=20)
def test_orientation_ocsvm():   return _orientation_case("ocsvm", 6, nu=0.05, gamma="scale")


def test_orientation_pca_holdout():
    rng = np.random.RandomState(7)
    W = rng.randn(6, 6)
    healthy_fit = rng.randn(500, 6) @ W
    healthy_holdout = rng.randn(200, 6) @ W
    abn_holdout = rng.randn(120, 6) * 6.0
    pca, k, evr, _ = fit_pca_evr(healthy_fit, evr_target=0.90, seed=0)
    assert np.median(pca_reconstruction_score(pca, abn_holdout)) > \
           np.median(pca_reconstruction_score(pca, healthy_holdout)) + 1e-6
    return True


def test_detector_param_guards():
    # IF float max_samples out of range
    try:
        make_detector("iforest", 0, max_samples=1.5); raise RuntimeError("ms>1 not caught")
    except AssertionError: pass
    # IF int max_samples < 1
    try:
        make_detector("iforest", 0, max_samples=0); raise RuntimeError("ms=0 not caught")
    except AssertionError: pass
    # IF bool max_samples rejected
    try:
        make_detector("iforest", 0, max_samples=True); raise RuntimeError("bool ms not caught")
    except AssertionError: pass
    # LOF requires n_healthy_fit_rows
    try:
        make_detector("lof", 0, n_neighbors=20); raise RuntimeError("missing n_healthy not caught")
    except AssertionError: pass
    # LOF n_neighbors >= n_healthy_fit_rows
    try:
        make_detector("lof", 0, n_neighbors=50, n_healthy_fit_rows=40)
        raise RuntimeError("n_neighbors>=n_fit not caught")
    except AssertionError: pass
    # OCSVM nu out of range
    try:
        make_detector("ocsvm", 0, nu=0.0); raise RuntimeError("nu=0 not caught")
    except AssertionError: pass
    try:
        make_detector("ocsvm", 0, nu=1.5); raise RuntimeError("nu>1 not caught")
    except AssertionError: pass
    # PCA invalid evr target
    rng = np.random.RandomState(0); Xf = rng.randn(20, 4)
    try:
        fit_pca_evr(Xf, evr_target=0.0, seed=0); raise RuntimeError("evr=0 not caught")
    except AssertionError: pass
    try:
        fit_pca_evr(Xf, evr_target=1.5, seed=0); raise RuntimeError("evr>1 not caught")
    except AssertionError: pass
    # valid constructions return metadata
    _, m = make_detector("lof", 0, n_neighbors=20, n_healthy_fit_rows=100)
    assert m["orientation_sign"] == -1 and m["score_method"] == "score_samples"
    return True


def test_per_engine_auc_both_class_aware():
    eng = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    y = np.array([0, 0, 1, 1, 0, 0, 0, 0])
    s = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.4, 0.2, 0.1])
    auc, sup, unsup = per_engine_auc(eng, y, s)
    assert sup == [1] and unsup == [2] and 2 not in auc and 0.0 <= auc[1] <= 1.0
    return True


def test_anomaly_bootstrap_invalid_marking():
    eng = np.repeat([1, 2, 3], 6)
    y = np.concatenate([np.zeros(6), np.zeros(6), np.ones(6)]).astype(int)
    s = np.linspace(0, 1, 18)
    pos_by = {e: np.where(eng == e)[0] for e in np.unique(eng)}
    drawn_sets = [[1, 2, 2], [3, 3, 3], [1, 3, 2], [2, 3, 1], [1, 1, 2], [3, 1, 3]]
    draws = [{"row_positions": np.concatenate([pos_by[e] for e in ds])} for ds in drawn_sets]
    r = anomaly_engine_bootstrap(eng, y, s, draws)
    assert r["B"] == 6 and r["n_valid"] == 3 and r["n_invalid"] == 3
    assert r["n_valid"] + r["n_invalid"] == 6
    assert np.isnan(r["replicate_values"][~r["valid_flags"]]).all()
    assert np.isfinite(r["replicate_values"][r["valid_flags"]]).all()
    assert 0.0 <= r["ci_low"] <= r["ci_high"] <= 1.0
    return True


def test_preprocessor_matrix_shape_order():
    rng = np.random.RandomState(12)
    X = np.column_stack([rng.randn(50), rng.randn(50)])
    pp = AnomalyPreprocessor().fit(X, ["f1", "f2"], fit_engine_ids=[1],
                                   healthy_boundary=125)
    Xt = pp.transform(X, ["f1", "f2"])
    assert Xt.shape == (50, 2)
    return True

def test_transform_rejects_shape_mismatch():
    # feature_names has FULL expected schema, but X has fewer columns -> reject.
    rng = np.random.RandomState(31)
    X = np.column_stack([rng.randn(40), np.full(40, 5.0), rng.randn(40) * 2])
    names = ["a", "const", "c"]
    pp = AnomalyPreprocessor().fit(X, names, fit_engine_ids=[1, 2], healthy_boundary=125)
    # X has 2 cols, feature_names claims 3 -> shape assert must fire
    X_bad = X[:, [0, 2]]                       # 2 columns
    try:
        pp.transform(X_bad, names)             # names length 3
        raise RuntimeError("shape mismatch not rejected")
    except AssertionError:
        return True


def test_to_jsonable():
    import json as _json
    _json.dumps(to_jsonable({"a": np.int64(3), "b": np.float64(1.5),
                             "c": np.bool_(True), "d": None,
                             "e": np.array([1.0, 2.0]), "f": "x"}))
    return True

# ---------------------------------------------------------------------------
# Engine-level bootstrap draws (shared by Task 5 and Task 6).
# Deterministic; multiplicity preserved; NO isin() row construction.
# ---------------------------------------------------------------------------
def engine_bootstrap_draws(engine_ids, B, seed):
    """
    Resample UNIQUE engines with replacement B times; build row positions by
    concatenating positions of every drawn engine occurrence (multiplicity kept).
    Returns (draws, n_unique).
    """
    import numpy as _np
    # --- (A1) explicit input validation ---
    engine_ids = _np.asarray(engine_ids)
    assert engine_ids.ndim == 1, "engine_ids must be one-dimensional"
    assert engine_ids.size > 0, "engine_ids must be non-empty"
    if _np.issubdtype(engine_ids.dtype, _np.floating):
        assert _np.all(_np.isfinite(engine_ids)), "engine_ids contains NaN/Inf"
    assert isinstance(B, (int, _np.integer)) and not isinstance(B, bool), "B must be a non-bool integer"
    assert int(B) > 0, "B must be a positive integer"
    assert isinstance(seed, (int, _np.integer)) and not isinstance(seed, bool), "seed must be integer-compatible"

    uniq = _np.unique(engine_ids)                       # sorted, deterministic
    n_unique = int(uniq.size)
    pos_by_engine = {int(e): _np.flatnonzero(engine_ids == e) for e in uniq}
    for e, pos in pos_by_engine.items():
        assert pos.size > 0, f"engine {e} has no rows"   # unique() guarantees this
    rng = _np.random.RandomState(int(seed))
    draws = []
    for _ in range(int(B)):
        drawn = uniq[rng.randint(0, n_unique, size=n_unique)]
        row_positions = _np.concatenate([pos_by_engine[int(e)] for e in drawn])
        assert row_positions.size > 0, "empty row_positions"
        assert row_positions.min() >= 0 and row_positions.max() < engine_ids.size, "row position out of bounds"
        draws.append({"drawn_engine_ids": drawn.astype(_np.int64),
                      "row_positions": row_positions.astype(_np.int64)})
    return draws, n_unique


def _test_engine_bootstrap_draws():
    """Reproducibility, multiplicity, bounds, and invalid-input rejection."""
    import numpy as _np
    eng = _np.array([10, 10, 10, 20, 20, 30])   # 3 unique; counts 3,2,1
    B, seed = 50, 123
    d1, n1 = engine_bootstrap_draws(eng, B=B, seed=seed)
    d2, n2 = engine_bootstrap_draws(eng, B=B, seed=seed)
    assert n1 == n2 == 3 and len(d1) == len(d2) == B
    for a, b in zip(d1, d2):
        assert _np.array_equal(a["drawn_engine_ids"], b["drawn_engine_ids"])
        assert _np.array_equal(a["row_positions"], b["row_positions"])
    d3, _ = engine_bootstrap_draws(eng, B=B, seed=seed + 1)
    assert any(not _np.array_equal(d1[i]["drawn_engine_ids"], d3[i]["drawn_engine_ids"]) for i in range(B))
    counts = {10: 3, 20: 2, 30: 1}
    for dr in d1:
        drawn, pos = dr["drawn_engine_ids"], dr["row_positions"]
        assert drawn.size == 3
        assert pos.size == int(sum(counts[int(e)] for e in drawn))
        assert pos.min() >= 0 and pos.max() < eng.size
        for e in _np.unique(drawn):
            k = int((drawn == e).sum())
            assert int((eng[pos] == e).sum()) == k * counts[int(e)]
    # --- (A2) invalid-input rejection ---
    for bad in ([], _np.array([])):
        try: engine_bootstrap_draws(bad, B=10, seed=1); assert False, "empty engine_ids not rejected"
        except AssertionError as ex:
            if "not rejected" in str(ex): raise
    for badB in (0, -5):
        try: engine_bootstrap_draws(eng, B=badB, seed=1); assert False, "bad B not rejected"
        except AssertionError as ex:
            if "not rejected" in str(ex): raise
    try: engine_bootstrap_draws(eng, B=True, seed=1); assert False, "bool B not rejected"
    except AssertionError as ex:
        if "not rejected" in str(ex): raise
    return True


# ==
# 12 · Module-level registry (built AFTER all test functions are defined)
# ==
ANOMALY_TESTS = [
    ("train subdivision ceil + integrity",         test_subdivision_ceil_and_integrity),
    ("train subdivision reproducible",             test_subdivision_reproducible),
    ("same-source role validation",                test_same_source_role_validation),
    ("official-test excluded by source",           test_test_excluded_by_source),
    ("empirical CDF boundary + right-tie",         test_cdf_boundary_and_ties),
    ("CDF rejects empty + non-finite scores",      test_cdf_rejects_empty_and_nonfinite),
    ("CDF provenance role enforced",               test_cdf_provenance_role_enforced),
    ("percentile bounds finite",                   test_percentile_bounds_finite),
    ("batch-composition invariance",               test_batch_composition_invariance),
    ("pooled CDF reproducible",                     test_pooled_cdf_reproducible),
    ("engine-balanced reference reproducible",     test_engine_balanced_reference_reproducible),
    ("OC-SVM subsample even + reproducible",       test_ocsvm_subsample_even_and_reproducible),
    ("OC-SVM subsample short-engine redistrib",    test_ocsvm_subsample_short_engine_redistribution),
    ("preprocessing contract schema (6 cases)",    test_preprocessor_contract_schema),
    ("preprocessor fit provenance guards",         test_preprocessor_fit_provenance_guards),
    ("PCA recon non-neg/monotone/contrib",         test_pca_recon_nonneg_monotone_and_contrib),
    ("fit_pca_evr selection",                       test_fit_pca_evr_selection),
    ("fit_pca_evr wide matrix (n<p)",              test_fit_pca_evr_wide_matrix),
    ("orientation IForest (holdout, distrib)",     test_orientation_iforest),
    ("orientation LOF (holdout, distrib)",         test_orientation_lof),
    ("orientation OC-SVM (holdout, distrib)",      test_orientation_ocsvm),
    ("orientation PCA (holdout, distrib)",         test_orientation_pca_holdout),
    ("detector param guards",                       test_detector_param_guards),
    ("per-engine AUC both-class-aware",            test_per_engine_auc_both_class_aware),
    ("anomaly bootstrap invalid-marking",          test_anomaly_bootstrap_invalid_marking),
    ("preprocessor matrix shape/order",            test_preprocessor_matrix_shape_order),
    ("to_jsonable conversion",                     test_to_jsonable),
    ("transform rejects X/names shape mismatch", test_transform_rejects_shape_mismatch),
    ("engine_bootstrap_draws", _test_engine_bootstrap_draws)
]


def run_all_tests(verbose=True):
    results, ok = {}, True
    for name, fn in ANOMALY_TESTS:
        try:
            fn(); results[name] = "PASS"
        except AssertionError as e:
            results[name] = f"FAIL: {e}"; ok = False
        except Exception as e:
            results[name] = f"ERROR: {type(e).__name__}: {e}"; ok = False
    if verbose:
        print(" anomaly module smoke tests ")
        print(f"Registered tests = {len(ANOMALY_TESTS)}")
        for k, v in results.items():
            print(f"  [{'OK' if v == 'PASS' else 'XX'}] {k}: {v}")
        print(f"PASS = {sum(1 for v in results.values() if v=='PASS')}/{len(ANOMALY_TESTS)}")
    return results, ok


if __name__ == "__main__":
    _, ok = run_all_tests(verbose=True)
    print("\nALL PASSED" if ok else "\nSOME TESTS FAILED")
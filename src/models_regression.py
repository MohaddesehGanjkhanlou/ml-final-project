"""
Jet Engine Hospital - RUL Regression Module
==
Reusable model factory, metrics, PHM/NASA asymmetric score, and split-conformal
prediction-interval utilities.

Selection discipline (enforced by the notebook, not here):
  * Cap, hyperparameters, representation, window, settings chosen on TRAIN/VAL only.
  * Primary validation metric: PHM asymmetric score (LOWER is better).
  * Secondary diagnostics: MAE and near-failure MAE. Metrics are NOT combined.
  * Capped-target vs uncapped-RUL metrics are reported separately.

Error/sign convention:
  d = prediction - truth
  Overestimation (d > 0, "engine looks healthier than it is") is penalized MORE.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

import config


# --------------------------------------------------------------------------
# PHM/NASA asymmetric score  (d = prediction - truth ; LOWER is better)
# --------------------------------------------------------------------------
def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Standard PHM'08 asymmetric scoring function.
      d = y_pred - y_true
      d <  0 (early / underestimate): s = exp(-d / 13) - 1  = expm1(-d/13)
      d >= 0 (late  / overestimate) : s = exp( d / 10) - 1  = expm1( d/10)
    Overestimation penalized more (10 < 13). LOWER is better.

    Uses boolean masks + np.expm1 so each branch is evaluated only on its own
    elements (no wasted exp on the unused branch; better numerical behavior).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true

    s = np.empty_like(d)
    late = d >= 0                     # overestimate / late
    early = ~late                     # underestimate / early
    s[late] = np.expm1(d[late] / 10.0)
    s[early] = np.expm1(-d[early] / 13.0)
    return float(np.sum(s))


def phm_score_mean(y_true, y_pred) -> float:
    """Per-sample mean PHM score (comparable across different-sized splits)."""
    n = len(y_true)
    return phm_score(y_true, y_pred) / n if n else float("nan")


# --------------------------------------------------------------------------
# Standard regression metrics
# --------------------------------------------------------------------------
def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sqrt(np.mean(d ** 2)))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def near_failure_mae(y_true_uncapped, y_pred, threshold: int = 30) -> float:
    """
    MAE restricted to near-failure rows (UNCAPPED RUL <= threshold).
    y_pred is compared on the same (capped or uncapped) scale as provided;
    the notebook passes matching arrays and documents the scale.
    """
    y_true_uncapped = np.asarray(y_true_uncapped, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true_uncapped <= threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_pred[mask] - y_true_uncapped[mask])))


# --------------------------------------------------------------------------
# Model factory
# --------------------------------------------------------------------------
def make_model(name: str, **kwargs):
    """
    Return an unfitted regressor.
      'ridge'      -> Ridge (use SCALED features)
      'poly2'      -> PolynomialFeatures(deg=2) + Ridge (use SCALED, reduced feats)
      'rf'         -> RandomForestRegressor (use UNSCALED features)
      'gbr'        -> GradientBoostingRegressor (use UNSCALED features)
    """
    seed = config.RANDOM_SEED
    if name == "ridge":
        return Ridge(alpha=kwargs.get("alpha", 1.0), random_state=seed)
    if name == "poly2":
        return Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=kwargs.get("alpha", 1.0), random_state=seed)),
        ])
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", None),
            min_samples_leaf=kwargs.get("min_samples_leaf", 5),
            n_jobs=-1, random_state=seed)
    if name == "gbr":
        return GradientBoostingRegressor(
            n_estimators=kwargs.get("n_estimators", 300),
            max_depth=kwargs.get("max_depth", 3),
            learning_rate=kwargs.get("learning_rate", 0.05),
            random_state=seed)
    raise ValueError(f"Unknown model name: {name}")


def apply_cap(y: np.ndarray, cap):
    """Clip RUL to [0, cap]. cap=None -> uncapped."""
    y = np.asarray(y, dtype=float)
    return y if cap is None else np.clip(y, 0, cap)


# --------------------------------------------------------------------------
# Sign-convention verification (worked numerical example)
# --------------------------------------------------------------------------
def verify_sign_convention(verbose=True):
    """
    Worked example: d=+10 (overestimate/late) must be penalized MORE than d=-10.
    """
    late  = phm_score([100.0], [110.0])   # d = +10
    early = phm_score([100.0], [ 90.0])   # d = -10
    if verbose:
        print("PHM sign-convention check (d = prediction - truth):")
        print(f"  d = +10 (overestimate/LATE) : score = {late:.4f}")
        print(f"  d = -10 (underestimate/EARLY): score = {early:.4f}")
        print(f"  late > early ? {late > early}  (must be True: LATE penalized more)")
    assert late > early, "Sign convention broken: late must be penalized more."
    # exact expected values
    assert np.isclose(late,  np.exp(10/10) - 1)   # e^1 - 1
    assert np.isclose(early, np.exp(10/13) - 1)
    return {"d_plus10": late, "d_minus10": early}


# --------------------------------------------------------------------------
# Split-conformal prediction intervals (pointwise, absolute-residual score)
# --------------------------------------------------------------------------
def conformal_quantile(residuals_abs: np.ndarray, coverage: float) -> float:
    """
    Split-conformal quantile of |residual| for target coverage (e.g., 0.90).
    Uses the finite-sample-adjusted rank: ceil((n+1)*coverage)/n quantile.
    """
    r = np.sort(np.asarray(residuals_abs, dtype=float))
    n = len(r)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * coverage))
    k = min(max(k, 1), n)                 # clamp to [1, n]
    return float(r[k - 1])                # k-th smallest (1-indexed)


def conformal_interval(pred: np.ndarray, q: float, upper_bound=None):
    """
    Pointwise interval [pred - q, pred + q], lower bound clipped at 0.
    If upper_bound is not None, the upper bound is also clipped to it
    (e.g., upper_bound=100 for a capped-RUL target with support [0, 100]).
    """
    pred = np.asarray(pred, dtype=float)
    lower = np.clip(pred - q, 0, None)
    upper = pred + q
    if upper_bound is not None:
        upper = np.minimum(upper, upper_bound)
    return lower, upper


def coverage_and_width(y_true, lower, upper):
    """Empirical coverage and average width."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered)), float(np.mean(upper - lower))


def test_conformal_coverage_synthetic(verbose=True):
    """
    IMPLEMENTATION SMOKE TEST only (not proof of real-data coverage): on
    exchangeable synthetic data, split-conformal should approximately achieve the
    target coverage. The synthetic target is NOT capped, so upper_bound=None.
    Prints actual empirical coverage, quantile, and average width.
    """
    rng = np.random.RandomState(0)
    n = 4000
    y = rng.uniform(0, 100, n)
    pred = y + rng.normal(0, 8, n)                 # homoscedastic noise
    resid = np.abs(pred - y)
    calib, test = resid[:2000], resid[2000:]
    q = conformal_quantile(calib, 0.90)
    yt, pt = y[2000:], pred[2000:]
    lo, hi = conformal_interval(pt, q, upper_bound=None)   # generic, uncapped
    cov, width = coverage_and_width(yt, lo, hi)
    if verbose:
        print("  [smoke test] synthetic split-conformal @ target=0.90 (upper_bound=None):")
        print(f"    quantile q          = {q:.4f}")
        print(f"    empirical coverage  = {cov:.4f}")
        print(f"    average width       = {width:.4f}")
    assert 0.86 <= cov <= 0.94, f"synthetic coverage {cov:.3f} off target 0.90"
    return True


if __name__ == "__main__":
    verify_sign_convention()
    print("\nConformal implementation smoke test (synthetic, exchangeable):")
    ok = test_conformal_coverage_synthetic(verbose=True)
    print("  smoke test:", "PASS" if ok else "FAIL")
    print("\nNOTE: synthetic smoke test verifies the IMPLEMENTATION only; it is "
          "NOT evidence of coverage on real, temporally-dependent engine data.")
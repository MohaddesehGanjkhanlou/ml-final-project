"""Production, leakage-safe C-MAPSS training and evaluation pipeline.

This module completes the capstone contract for FD001 and FD003.  Selection is
performed on validation-tune engines, calibration and policy selection on
validation-calibration engines, and the official test set is evaluated once by
the frozen system.  All resampling is by engine, never by row.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score, r2_score)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

import config
import data_loader as dl
import models_classification as mc
import models_regression as mr
import splits as split_module
from features import FeaturePipeline

HORIZONS = (10, 20, 30)
POLICY_COST = {"miss": 100.0, "late": 5.0, "early": 1.0}


def _json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def audit_frame(df: pd.DataFrame, subset: str, source: str) -> dict:
    lengths = df.groupby("engine_id")["cycle"].max()
    discontinuous = []
    for engine_id, group in df.groupby("engine_id"):
        cycles = np.sort(group["cycle"].to_numpy())
        if not np.array_equal(cycles, np.arange(1, len(cycles) + 1)):
            discontinuous.append(int(engine_id))
    numeric = df.select_dtypes(include=[np.number]).to_numpy(float)
    return {
        "subset": subset, "source": source, "rows": int(len(df)),
        "engines": int(df.engine_id.nunique()),
        "length_min": int(lengths.min()), "length_median": float(lengths.median()),
        "length_max": int(lengths.max()), "missing": int(df.isna().sum().sum()),
        "nonfinite": int((~np.isfinite(numeric)).sum()),
        "duplicate_keys": int(df.duplicated(["engine_id", "cycle"]).sum()),
        "discontinuous_engines": discontinuous,
    }


def engine_subdivision(engine_ids, seed: int):
    ids = np.array(sorted(engine_ids), dtype=int)
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    cut = len(ids) // 2
    return sorted(ids[:cut].tolist()), sorted(ids[cut:].tolist())


def prepare_data(subset: str):
    raw_train = dl.load_raw(subset, "train")
    official_test_raw = dl.load_raw(subset, "test")
    split = split_module.make_engine_split(subset)
    val_tune_ids, val_calib_ids = engine_subdivision(split["val"], config.RANDOM_SEED + 17)

    labeled = dl.add_classification_labels(dl.add_train_rul(raw_train, cap=None))
    test = dl.add_classification_labels(dl.load_test_with_rul(subset, cap=None))
    train = labeled[labeled.engine_id.isin(split["train"])].copy().reset_index(drop=True)
    val_tune = labeled[labeled.engine_id.isin(val_tune_ids)].copy().reset_index(drop=True)
    val_calib = labeled[labeled.engine_id.isin(val_calib_ids)].copy().reset_index(drop=True)
    test = test.copy().reset_index(drop=True)
    return train, val_tune, val_calib, test, split, {
        "official_train": audit_frame(raw_train, subset, "official_train"),
        "official_test_integrity": audit_frame(official_test_raw, subset, "official_test"),
        "roles": {"train": split["train"], "val_tune": val_tune_ids,
                  "val_calib": val_calib_ids, "official_test": split["test"]},
    }


def select_columns(train: pd.DataFrame):
    variances = train[config.SENSOR_COLS].var()
    keep_sensors = variances[variances > 1e-10].index.tolist()
    removed = variances[variances <= 1e-10].index.tolist()
    setting_variances = train[config.SETTING_COLS].var()
    keep_settings = setting_variances[setting_variances > 1e-10].index.tolist()
    return keep_sensors, keep_settings, removed, variances.to_dict()


def fit_feature_pipeline(train, sensors, settings, representation, window, include_settings):
    fp = FeaturePipeline(sensors, settings, representation=representation,
                         window=window, include_settings=include_settings,
                         require_continuity=True)
    fp.fit(train)
    return fp


def regression_metrics(y, pred, uncapped):
    y = np.asarray(y, float); pred = np.asarray(pred, float); uncapped = np.asarray(uncapped, float)
    out = {"MAE": mr.mae(y, pred), "RMSE": mr.rmse(y, pred), "R2": mr.r2(y, pred),
           "PHM_mean": mr.phm_score_mean(y, pred)}
    regions = {"early": uncapped > 125, "mid": (uncapped > 30) & (uncapped <= 125),
               "near_failure": uncapped <= 30}
    out["regions"] = {k: {"n": int(m.sum()), "MAE": mr.mae(y[m], pred[m]),
                           "RMSE": mr.rmse(y[m], pred[m])} for k, m in regions.items() if m.any()}
    return out


def engine_bootstrap_regression(engine_ids, y, pred, draws=200, seed=42):
    ids = np.asarray(engine_ids); unique = np.unique(ids); rng = np.random.RandomState(seed)
    positions = {e: np.flatnonzero(ids == e) for e in unique}
    values = {"MAE": [], "RMSE": [], "PHM_mean": []}
    for _ in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        pos = np.concatenate([positions[e] for e in sampled])
        values["MAE"].append(mr.mae(y[pos], pred[pos]))
        values["RMSE"].append(mr.rmse(y[pos], pred[pos]))
        values["PHM_mean"].append(mr.phm_score_mean(y[pos], pred[pos]))
    return {name: {"low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5))}
            for name, v in values.items()}


def engine_bootstrap_classification(engine_ids, y, p, decision, draws=200, seed=42):
    ids=np.asarray(engine_ids); unique=np.unique(ids); rng=np.random.RandomState(seed)
    positions={e:np.flatnonzero(ids==e) for e in unique}
    values={k:[] for k in ("PR_AUC","precision","recall","F1","Brier")}
    for _ in range(draws):
        sampled=rng.choice(unique,len(unique),replace=True); pos=np.concatenate([positions[e] for e in sampled])
        yy=y[pos]; pp=p[pos]; dd=decision[pos]
        if yy.sum()>0: values["PR_AUC"].append(average_precision_score(yy,pp))
        values["precision"].append(precision_score(yy,dd,zero_division=0))
        values["recall"].append(recall_score(yy,dd,zero_division=0))
        values["F1"].append(f1_score(yy,dd,zero_division=0)); values["Brier"].append(brier_score_loss(yy,pp))
    return {k:{"low":float(np.percentile(v,2.5)),"high":float(np.percentile(v,97.5))} for k,v in values.items()}


def train_regression(train, tune, calib, test, sensors, settings):
    cache = {}
    def matrices(rep, window, incl):
        key = (rep, window, incl)
        if key not in cache:
            fp = fit_feature_pipeline(train, sensors, settings, rep, window, incl)
            cache[key] = (fp,) + tuple(fp.transform(d, scaled=True)[0] for d in (train, tune, calib, test))
        return cache[key]

    cap_rows = []
    fp, xtr, xva, _, _ = matrices("window", 15, False)
    for cap in (100, 125, 150):
        model = mr.make_model("ridge", alpha=10.0).fit(xtr, np.minimum(train.RUL, cap))
        pred = np.clip(model.predict(xva), 0, cap)
        cap_rows.append({"cap": cap, **regression_metrics(np.minimum(tune.RUL, cap), pred, tune.RUL)})
    cap = min(cap_rows, key=lambda r: r["PHM_mean"])["cap"]

    ablations = []
    ablation_configs = [(window, False) for window in config.FEATURE_WINDOWS]
    if settings:
        ablation_configs.append((15, True))
    for window, include_settings in ablation_configs:
        fp, xtr, xva, _, _ = matrices("window", window, include_settings)
        m = mr.make_model("ridge", alpha=10.0).fit(xtr, np.minimum(train.RUL, cap))
        p = np.clip(m.predict(xva), 0, cap)
        ablations.append({"window": window, "include_settings": include_settings,
                          **regression_metrics(np.minimum(tune.RUL, cap), p, tune.RUL)})
    best_ablation = min(ablations, key=lambda r: r["PHM_mean"])
    best_window = int(best_ablation["window"]); best_settings = bool(best_ablation["include_settings"])

    candidates = [
        ("current_ridge", "baseline", 1, False, "ridge", True, {"alpha": 10.0}),
        ("current_poly2", "baseline", 1, False, "poly2", True, {"alpha": 10.0}),
        ("current_rf", "baseline", 1, False, "rf", False, {"n_estimators": 120, "min_samples_leaf": 5}),
        ("current_gbr", "baseline", 1, False, "gbr", False, {"n_estimators": 150}),
        ("window_ridge", "window", best_window, best_settings, "ridge", True, {"alpha": 10.0}),
        ("window_rf", "window", best_window, best_settings, "rf", False, {"n_estimators": 120, "min_samples_leaf": 5}),
        ("window_gbr", "window", best_window, best_settings, "gbr", False, {"n_estimators": 150}),
    ]
    rows, fitted = [], {}
    candidate_features = {}
    for _, rep, win, incl, _, _, _ in candidates:
        key=(rep,win,incl)
        if key not in candidate_features:
            fp=fit_feature_pipeline(train,sensors,settings,rep,win,incl)
            candidate_features[key]={"fp":fp,
                "train_scaled":fp.transform(train,scaled=True)[0],"tune_scaled":fp.transform(tune,scaled=True)[0],
                "train_raw":fp.transform(train,scaled=False)[0],"tune_raw":fp.transform(tune,scaled=False)[0]}
    for label, rep, win, incl, family, scaled, params in candidates:
        entry=candidate_features[(rep,win,incl)]; fp=entry["fp"]
        suffix="scaled" if scaled else "raw"; xtr=entry[f"train_{suffix}"]; xva=entry[f"tune_{suffix}"]
        started = time.perf_counter(); model = mr.make_model(family, **params).fit(xtr, np.minimum(train.RUL, cap))
        latency_fit = time.perf_counter() - started
        started = time.perf_counter(); pred = np.clip(model.predict(xva), 0, cap)
        latency = (time.perf_counter() - started) * 1000 / len(xva)
        row = {"model": label, "representation": rep, "window": win,
               "include_settings": incl, "fit_seconds": latency_fit,
               "latency_ms_per_row": latency, "model_size_bytes":len(pickle.dumps(model)),
               **regression_metrics(np.minimum(tune.RUL, cap), pred, tune.RUL)}
        rows.append(row); fitted[label] = (fp, model, scaled)
    selected_row = min(rows, key=lambda r: r["PHM_mean"])
    fp, model, scaled = fitted[selected_row["model"]]
    xcal = fp.transform(calib, scaled=scaled)[0]; xtest = fp.transform(test, scaled=scaled)[0]
    pcal = np.clip(model.predict(xcal), 0, cap); ptest = np.clip(model.predict(xtest), 0, cap)
    q = mr.conformal_quantile(np.abs(np.minimum(calib.RUL, cap) - pcal), config.CONFORMAL_COVERAGE)
    lower, upper = mr.conformal_interval(ptest, q, upper_bound=cap)
    test_metrics = regression_metrics(np.minimum(test.RUL, cap), ptest, test.RUL)
    test_metrics["interval_coverage"], test_metrics["interval_average_width"] = mr.coverage_and_width(
        np.minimum(test.RUL, cap), lower, upper)
    test_metrics["ci95"] = engine_bootstrap_regression(test.engine_id, np.minimum(test.RUL.to_numpy(), cap),
                                                        ptest, config.BOOTSTRAP_DRAWS, config.RANDOM_SEED)
    return {"cap": cap, "cap_comparison": cap_rows, "ablations": ablations,
            "comparison": rows, "selected": selected_row, "test_metrics": test_metrics,
            "conformal_q": q}, {"fp": fp, "model": model, "scaled": scaled,
                                "cap": cap, "q": q}, ptest, lower, upper


def train_classification(train, tune, calib, test, sensors, settings, window):
    fp = fit_feature_pipeline(train, sensors, settings, "window", window, bool(settings))
    scaled = {role: fp.transform(frame, scaled=True)[0] for role, frame in
              (("train", train), ("tune", tune), ("calib", calib), ("test", test))}
    raw = {role: fp.transform(frame, scaled=False)[0] for role, frame in
           (("train", train), ("tune", tune), ("calib", calib), ("test", test))}
    rows, bundles, probs, thresholds, test_rows = [], {}, {}, {}, []
    calib_probs, raw_test_probs = {}, {}
    for h in HORIZONS:
        ytr = train[f"label_h{h}"].to_numpy(); ytu = tune[f"label_h{h}"].to_numpy()
        choices = []
        for name, is_scaled, params in (("logistic", True, {"class_weight": "balanced"}),
                                         ("rf", False, {"n_estimators": 160, "min_samples_leaf": 5,
                                                        "class_weight": "balanced"})):
            matrices = scaled if is_scaled else raw
            started = time.perf_counter(); model = mc.make_classifier(name, **params).fit(matrices["train"], ytr)
            fit_seconds = time.perf_counter() - started
            p = model.predict_proba(matrices["tune"])[:, 1]
            started=time.perf_counter(); model.predict_proba(matrices["tune"]); latency=(time.perf_counter()-started)*1000/len(tune)
            row = {"horizon": h, "model": name, "fit_seconds": fit_seconds,
                   "latency_ms_per_row":latency,"model_size_bytes":len(pickle.dumps(model)),
                   "PR_AUC": average_precision_score(ytu, p), "Brier": brier_score_loss(ytu, p)}
            rows.append(row); choices.append((row, model, is_scaled))
        choice, model, is_scaled = max(choices, key=lambda item: item[0]["PR_AUC"])
        matrices = scaled if is_scaled else raw
        scores_cal = mc.calibration_scores(model, matrices["calib"])
        calibrator = mc.fit_platt(scores_cal, calib[f"label_h{h}"].to_numpy())
        pcal = mc.apply_platt(calibrator, scores_cal)
        raw_test_probs[h]=model.predict_proba(matrices["test"])[:,1]
        ptest = mc.apply_platt(calibrator, mc.calibration_scores(model, matrices["test"]))
        bundles[h] = {"fp": fp, "clf": model, "platt": calibrator, "scaled": is_scaled}
        probs[h] = ptest; calib_probs[h]=pcal
    c10,c20,c30=mc.enforce_monotone(calib_probs[10],calib_probs[20],calib_probs[30])
    calib_probs={10:c10,20:c20,30:c30}
    for h in HORIZONS:
        thresholds[h],_=mc.best_cost_threshold(calib[f"label_h{h}"],calib_probs[h])
    p10, p20, p30 = mc.enforce_monotone(probs[10], probs[20], probs[30]); probs = {10:p10,20:p20,30:p30}
    for h in HORIZONS:
        y = test[f"label_h{h}"].to_numpy(); p = probs[h]; d = (p >= thresholds[h]).astype(int)
        metric_row={"horizon": h, "threshold": thresholds[h],
                          "precision": precision_score(y,d,zero_division=0),
                          "recall": recall_score(y,d,zero_division=0),
                          "F1": f1_score(y,d,zero_division=0),
                          "PR_AUC": average_precision_score(y,p),
                          "Brier_before_calibration":brier_score_loss(y,raw_test_probs[h]),
                          "Brier": brier_score_loss(y,p),
                          "tn": int(((y==0)&(d==0)).sum()), "fp": int(((y==0)&(d==1)).sum()),
                          "fn": int(((y==1)&(d==0)).sum()), "tp": int(((y==1)&(d==1)).sum())}
        metric_row["ci95"]=engine_bootstrap_classification(test.engine_id,y,p,d,config.BOOTSTRAP_DRAWS,
                                                            config.RANDOM_SEED+h)
        test_rows.append(metric_row)
    return {"comparison": rows, "test_metrics": test_rows, "thresholds": thresholds}, \
           {"bundles": bundles, "thresholds": thresholds}, probs


def anomaly_score(name, model, X):
    if name == "pca":
        return np.mean((X - model.inverse_transform(model.transform(X))) ** 2, axis=1)
    return -model.score_samples(X)


def percentile_from_reference(reference, scores):
    ref = np.sort(np.asarray(reference, float))
    return np.searchsorted(ref, scores, side="right") / len(ref)


def persistent_flags(alerts, m=2, n=3):
    return pd.Series(alerts).rolling(n, min_periods=n).sum().ge(m).to_numpy()


def warning_cost(frame, flags, horizon=30):
    records = []
    for engine_id, pos in frame.groupby("engine_id", sort=False).indices.items():
        pos = np.asarray(pos); local = np.flatnonzero(flags[pos])
        if len(local) == 0:
            records.append({"engine_id": int(engine_id), "miss": 1, "lead_time": np.nan,
                            "late_delay": horizon, "early_burden": 0})
        else:
            lead = float(frame.iloc[pos[local[0]]].RUL)
            records.append({"engine_id": int(engine_id), "miss": 0, "lead_time": lead,
                            "late_delay": max(0.0, horizon-lead), "early_burden": max(0.0, lead-horizon)})
    out = pd.DataFrame(records)
    out["cost"] = (POLICY_COST["miss"]*out.miss + POLICY_COST["late"]*out.late_delay +
                   POLICY_COST["early"]*out.early_burden)
    return out


def engine_table_ci(table, columns, draws=1000, seed=42):
    rng=np.random.RandomState(seed); n=len(table); result={}
    for col in columns:
        vals=[]
        for _ in range(draws): vals.append(table.iloc[rng.randint(0,n,n)][col].mean())
        result[col]={"low":float(np.percentile(vals,2.5)),"high":float(np.percentile(vals,97.5))}
    return result


def train_anomaly(train, tune, calib, test, sensors, settings, window):
    fp = fit_feature_pipeline(train, sensors, settings, "window", window, bool(settings))
    X = {k: fp.transform(v, scaled=True)[0].to_numpy(float) for k,v in
         (("train",train),("tune",tune),("calib",calib),("test",test))}
    # Label-free healthy proxy: the first 30 observed cycles of every training
    # engine.  Detector fitting never reads RUL or a failure-horizon label.
    healthy = train.cycle.to_numpy() <= 30
    Xh = X["train"][healthy]
    candidates = {
        "iforest": IsolationForest(n_estimators=200, contamination="auto", random_state=config.RANDOM_SEED, n_jobs=-1),
        "lof": LocalOutlierFactor(n_neighbors=35, novelty=True, contamination="auto", n_jobs=-1),
        "ocsvm": OneClassSVM(kernel="rbf", nu=0.05, gamma="scale"),
        "pca": PCA(n_components=0.95, svd_solver="full"),
    }
    validation, fitted = [], {}
    for name, model in candidates.items():
        fit_matrix=Xh
        if name=="ocsvm" and len(Xh)>3000:
            rng=np.random.RandomState(config.RANDOM_SEED)
            fit_matrix=Xh[np.sort(rng.choice(len(Xh),3000,replace=False))]
        started=time.perf_counter(); model.fit(fit_matrix); fit_seconds=time.perf_counter()-started
        ref=anomaly_score(name, model, Xh); tune_score=anomaly_score(name, model, X["tune"])
        tune_pct=percentile_from_reference(ref,tune_score); proxy=(tune.RUL.to_numpy()<=30).astype(int)
        validation.append({"model":name,"fit_seconds":fit_seconds,"PR_AUC_proxy":average_precision_score(proxy,tune_pct)})
        fitted[name]=(model,ref)
    selected=max(validation,key=lambda r:r["PR_AUC_proxy"])["model"]
    model,ref=fitted[selected]; calib_pct=percentile_from_reference(ref,anomaly_score(selected,model,X["calib"]))
    policy_rows=[]
    for threshold in (0.90,0.95,0.975,0.99):
        for m,n in ((2,3),(3,5)):
            flags=np.zeros(len(calib),bool)
            for _,pos in calib.groupby("engine_id",sort=False).indices.items():
                pos=np.asarray(pos); flags[pos]=persistent_flags(calib_pct[pos]>=threshold,m,n)
            costs=warning_cost(calib,flags)
            policy_rows.append({"threshold":threshold,"m":m,"n":n,"average_cost":costs.cost.mean(),
                                "miss_rate":costs.miss.mean(),"late_delay":costs.late_delay.mean(),
                                "early_burden":costs.early_burden.mean()})
    policy=min(policy_rows,key=lambda r:r["average_cost"])
    comparisons=[]; selected_pct=None; selected_flags=None; selected_lead=None
    for name,(candidate,reference) in fitted.items():
        pct=percentile_from_reference(reference,anomaly_score(name,candidate,X["test"]))
        flags=np.zeros(len(test),bool)
        for _,pos in test.groupby("engine_id",sort=False).indices.items():
            pos=np.asarray(pos); flags[pos]=persistent_flags(pct[pos]>=policy["threshold"],policy["m"],policy["n"])
        lead=warning_cost(test,flags); proxy=(test.RUL.to_numpy()<=30).astype(int)
        median_lead=(float(lead.lead_time.dropna().median()) if lead.lead_time.notna().any() else None)
        comparisons.append({"model":name,"PR_AUC_proxy":average_precision_score(proxy,pct),
                            "false_alert_rate_healthy":float(flags[test.RUL.to_numpy()>30].mean()),
                            "miss_rate":lead.miss.mean(),"average_cost":lead.cost.mean(),
                            "median_lead_time":median_lead})
        if name==selected: selected_pct,selected_flags,selected_lead=pct,flags,lead
    pca_model,pca_ref=fitted["pca"]; recon=pca_model.inverse_transform(pca_model.transform(X["test"]))
    contrib=np.mean((X["test"]-recon)**2,axis=0)
    pca_contributions=sorted([{"feature":name,"mean_squared_contribution":float(value)}
                              for name,value in zip(fp.feature_names_,contrib)],
                             key=lambda row:row["mean_squared_contribution"],reverse=True)[:20]
    return {"healthy_definition":"training engines, observed cycle <= 30 (label-free)",
            "healthy_fit_rows":int(healthy.sum()),
            "validation":validation,"policy_search":policy_rows,"selected":selected,
            "policy":policy,"test_comparison":comparisons,
            "pca_top_contributions":pca_contributions,
            "lead_time_summary":{"median":float(selected_lead.lead_time.median()),
                                 "miss_rate":float(selected_lead.miss.mean()),
                                 "average_cost":float(selected_lead.cost.mean()),
                                 "late_delay":float(selected_lead.late_delay.mean()),
                                 "early_burden":float(selected_lead.early_burden.mean()),
                                 "ci95":engine_table_ci(selected_lead,["miss","cost","late_delay","early_burden"]) }}, \
           {"fp":fp,"model":model,"model_name":selected,"reference_scores":ref,"policy":policy}, \
           selected_pct,selected_flags,selected_lead


def make_recommendations(test, rul_pred, lower, upper, probabilities, thresholds, anomaly_pct, anomaly_flags):
    """Combine three traceable evidence-family actions without fake confidence probabilities."""
    rec=[]; explanation=[]; rul_actions=[]; classification_actions=[]; anomaly_actions=[]
    agreements=[]; confidence=[]
    for i in range(len(test)):
        rul_action = "STOP" if lower[i] <= 10 else ("INSPECT" if lower[i] <= 30 else "CONTINUE")
        if probabilities[10][i] >= thresholds[10]:
            classification_action = "STOP"
        elif (probabilities[20][i] >= thresholds[20]
              or probabilities[30][i] >= thresholds[30]):
            classification_action = "INSPECT"
        else:
            classification_action = "CONTINUE"
        anomaly_action = "INSPECT" if anomaly_flags[i] else "CONTINUE"
        family_actions = (rul_action, classification_action, anomaly_action)
        final_action = max(family_actions, key={"CONTINUE": 0, "INSPECT": 1, "STOP": 2}.get)
        agreement = sum(action == final_action for action in family_actions)
        reasons=[]
        if rul_action != "CONTINUE": reasons.append(f"RUL lower bound -> {rul_action}")
        if classification_action != "CONTINUE": reasons.append(f"calibrated failure risk -> {classification_action}")
        if anomaly_action != "CONTINUE": reasons.append("persistent anomaly -> INSPECT")
        if not reasons: reasons.append("all three evidence families support CONTINUE")
        rul_actions.append(rul_action); classification_actions.append(classification_action)
        anomaly_actions.append(anomaly_action); rec.append(final_action)
        agreements.append(agreement); confidence.append({3:"HIGH",2:"MODERATE",1:"LOW"}[agreement])
        explanation.append("; ".join(reasons))
    return tuple(np.asarray(values) for values in (
        rec, explanation, rul_actions, classification_actions,
        anomaly_actions, agreements, confidence,
    ))


def plot_outputs(subset, test, rul_pred, lower, upper, probabilities, anomaly_pct):
    out=config.FIGURES_DIR; out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(11,4)); lengths=test.groupby("engine_id").size()
    ax[0].hist(lengths,bins=20,color="#286090"); ax[0].set(title=f"{subset} test sequence lengths",xlabel="cycles")
    ax[1].bar([str(h) for h in HORIZONS],[test[f"label_h{h}"].mean() for h in HORIZONS],color="#d98c10")
    ax[1].set(title="Failure-horizon class balance",xlabel="horizon",ylabel="positive fraction")
    fig.tight_layout(); fig.savefig(out/f"{subset}_production_eda.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,e in zip(axes,test.engine_id.unique()[:3]):
        g=test[test.engine_id==e]; ax.plot(g.cycle,g.sensor_2); ax.set(title=f"Engine {e}",xlabel="cycle",ylabel="sensor_2")
    fig.tight_layout(); fig.savefig(out/f"{subset}_production_sensor_traces.png",dpi=160); plt.close(fig)
    engines=test.engine_id.unique()[:3]; fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,e in zip(axes,engines):
        m=test.engine_id.to_numpy()==e; ax.plot(test.loc[m,"cycle"],test.loc[m,"RUL"],label="truth")
        ax.plot(test.loc[m,"cycle"],rul_pred[m],label="prediction"); ax.fill_between(test.loc[m,"cycle"],lower[m],upper[m],alpha=.2)
        ax.set_title(f"Engine {e}"); ax.set_xlabel("cycle")
    axes[0].set_ylabel("RUL cycles"); axes[-1].legend(); fig.tight_layout(); fig.savefig(out/f"{subset}_production_rul_traces.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); residual=rul_pred-np.minimum(test.RUL.to_numpy(),np.nanmax(upper))
    ax.scatter(np.minimum(test.RUL,np.nanmax(upper)),residual,s=5,alpha=.2); ax.axhline(0,color="black")
    ax.set(xlabel="true capped RUL",ylabel="prediction - truth",title=f"{subset} residual diagnostics")
    fig.tight_layout(); fig.savefig(out/f"{subset}_production_residuals.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,h in zip(axes,HORIZONS):
        y=test[f"label_h{h}"]; p=probabilities[h]; pr,rc,_=precision_recall_curve(y,p); ax.plot(rc,pr)
        ax.set(title=f"h={h}, AP={average_precision_score(y,p):.3f}",xlabel="recall",ylabel="precision")
    fig.tight_layout(); fig.savefig(out/f"{subset}_production_pr.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,h in zip(axes,HORIZONS):
        frac,mean=calibration_curve(test[f"label_h{h}"],probabilities[h],n_bins=8,strategy="quantile")
        ax.plot([0,1],[0,1],"--",color="gray"); ax.plot(mean,frac,"o-"); ax.set(title=f"h={h}",xlabel="mean probability",ylabel="observed frequency")
    fig.tight_layout(); fig.savefig(out/f"{subset}_production_reliability.png",dpi=160); plt.close(fig)
    aligned=pd.DataFrame({"RUL":test.RUL,"pct":anomaly_pct}); mean=aligned.assign(bin=(aligned.RUL//10)*10).groupby("bin").pct.mean()
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(mean.index,mean.values); ax.invert_xaxis(); ax.set(xlabel="cycles to failure",ylabel="anomaly percentile",title=f"{subset} anomaly trajectory")
    fig.tight_layout(); fig.savefig(out/f"{subset}_production_anomaly.png",dpi=160); plt.close(fig)


def run_subset(subset: str):
    print(f"\n {subset}: data and audit ", flush=True)
    train,tune,calib,test,split,audit=prepare_data(subset)
    sensors,settings,removed,variances=select_columns(train)
    audit["feature_filter"]={"kept_sensors":sensors,"removed_near_constant":removed,
                             "kept_settings":settings,"affected_engines":0,"affected_rows":0}
    split_module.save_split(subset,split)
    print(f" {subset}: regression ", flush=True)
    reg,reg_bundle,rul_pred,lower,upper=train_regression(train,tune,calib,test,sensors,settings)
    window=int(reg["selected"]["window"])
    print(f" {subset}: classification ", flush=True)
    clf,clf_bundle,probabilities=train_classification(train,tune,calib,test,sensors,settings,window)
    print(f" {subset}: anomaly and warning policy ", flush=True)
    anomaly,anomaly_bundle,anomaly_pct,anomaly_flags,lead=train_anomaly(train,tune,calib,test,sensors,settings,window)
    (rec,why,rul_action,classification_action,anomaly_action,
     signal_agreement,decision_confidence)=make_recommendations(
        test,rul_pred,lower,upper,probabilities,clf_bundle["thresholds"],anomaly_pct,anomaly_flags)
    plot_outputs(subset,test,rul_pred,lower,upper,probabilities,anomaly_pct)
    truth_action=np.where(test.RUL.to_numpy()<=10,"STOP",np.where(test.RUL.to_numpy()<=30,"INSPECT","CONTINUE"))
    final_pos=test.groupby("engine_id").tail(1).index.to_numpy()
    action_confusion=pd.crosstab(pd.Series(truth_action[final_pos],name="truth"),
                                 pd.Series(rec[final_pos],name="recommendation"),dropna=False).to_dict()
    summary={"subset":subset,"seed":config.RANDOM_SEED,"audit":audit,"regression":reg,
             "classification":clf,"anomaly":anomaly,"decision_counts":pd.Series(rec).value_counts().to_dict(),
             "decision_confusion_by_engine":action_confusion,
             "model_metadata":{"version":"1.0.0","trained_utc":pd.Timestamp.utcnow().isoformat(),
                               "feature_window":window,"sensors":sensors,"settings":settings}}
    config.RESULTS_DIR.mkdir(parents=True,exist_ok=True); write_json(config.RESULTS_DIR/f"{subset}_summary.json",summary)
    system={"metadata":summary["model_metadata"],"regression":reg_bundle,
            "classification":clf_bundle,"anomaly":anomaly_bundle}
    config.SYSTEMS_DIR.mkdir(parents=True,exist_ok=True); joblib.dump(system,config.SYSTEMS_DIR/f"{subset}_system.joblib",compress=3)
    demo=test[["engine_id","cycle","RUL"]+config.SETTING_COLS+config.SENSOR_COLS].copy()
    demo["rul_prediction"]=rul_pred; demo["rul_lower"]=lower; demo["rul_upper"]=upper
    for h in HORIZONS: demo[f"risk_h{h}"]=probabilities[h]; demo[f"threshold_h{h}"]=clf_bundle["thresholds"][h]
    demo["anomaly_normalized_score"]=anomaly_pct
    demo["anomaly_percentile"]=anomaly_pct
    demo["anomaly_threshold"]=anomaly_bundle["policy"]["threshold"]
    demo["anomaly_threshold_margin"]=demo["anomaly_normalized_score"]-demo["anomaly_threshold"]
    demo["anomaly_persistent"]=anomaly_flags
    demo["rul_evidence_action"]=rul_action
    demo["classification_evidence_action"]=classification_action
    demo["anomaly_evidence_action"]=anomaly_action
    demo["signal_agreement_count"]=signal_agreement.astype(int)
    demo["decision_confidence"]=decision_confidence
    demo["recommendation"]=rec; demo["explanation"]=why
    config.DEMO_DIR.mkdir(parents=True,exist_ok=True); demo.to_csv(config.DEMO_DIR/f"{subset}_predictions.csv.gz",index=False,compression="gzip")
    lead.to_csv(config.RESULTS_DIR/f"{subset}_lead_times.csv",index=False)
    return summary


def build_master(stage1,stage2):
    rows=[]
    for s in (stage1,stage2):
        rows.append({"subset":s["subset"],"task":"RUL","model":s["regression"]["selected"]["model"],
                     **{k:s["regression"]["test_metrics"][k] for k in ("MAE","RMSE","R2","PHM_mean")}})
        for row in s["classification"]["test_metrics"]:
            rows.append({"subset":s["subset"],"task":f"classification_h{row['horizon']}","model":"selected+Platt",
                         "PR_AUC":row["PR_AUC"],"F1":row["F1"],"Brier":row["Brier"]})
        for row in s["anomaly"]["test_comparison"]:
            rows.append({"subset":s["subset"],"task":"anomaly","model":row["model"],
                         "PR_AUC":row["PR_AUC_proxy"],"miss_rate":row["miss_rate"],"cost":row["average_cost"]})
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR/"master_results.csv",index=False)
    write_json(config.RESULTS_DIR/"project_summary.json",{"stage1":stage1,"stage2":stage2})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--subsets",nargs="+",default=[config.STAGE1_SUBSET,config.STAGE2_SUBSET])
    args=parser.parse_args(); results=[run_subset(s) for s in args.subsets]
    if len(results)>=2: build_master(results[0],results[1])
    print("\nProduction pipeline completed.")


if __name__=="__main__": main()

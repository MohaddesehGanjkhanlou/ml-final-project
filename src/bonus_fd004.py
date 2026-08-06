"""FD004 combined-condition/multi-fault bonus experiment."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import config
import models_regression as mr
import production_pipeline as pp
from condition_features import ConditionAwareFeaturePipeline
from features import FeaturePipeline


def fit_predict(fp, family, train, eval_frame, cap=125, **params):
    scaled=family in ("ridge","poly2")
    Xtr=fp.transform(train,scaled=scaled)[0]; Xe=fp.transform(eval_frame,scaled=scaled)[0]
    model=mr.make_model(family,**params).fit(Xtr,np.minimum(train.RUL,cap))
    return model,np.clip(model.predict(Xe),0,cap),scaled


def source_transfer(subset, target_test, cap=125, window=15):
    train,_,_,_,_,_=pp.prepare_data(subset)
    sensors,settings,_,_=pp.select_columns(train)
    if subset=="FD002":
        fp=ConditionAwareFeaturePipeline(sensors,config.SETTING_COLS,window=window,
                                         include_settings=True,n_regimes=6).fit(train)
    else:
        fp=FeaturePipeline(sensors,settings,representation="window",window=window,
                           include_settings=bool(settings),require_continuity=True).fit(train)
    model,pred,scaled=fit_predict(fp,"ridge",train,target_test,cap,alpha=10.0)
    metrics=pp.regression_metrics(np.minimum(target_test.RUL,cap),pred,target_test.RUL)
    return {"source":subset,"training":"source train engines, Ridge/window-15",
            **{k:metrics[k] for k in ("MAE","RMSE","R2","PHM_mean")}}, \
           {"fp":fp,"model":model,"scaled":scaled,"cap":cap}


def run():
    subset="FD004"
    print("FD004 dedicated global pipeline",flush=True)
    global_summary=pp.run_subset(subset)
    train,tune,calib,test,split,audit=pp.prepare_data(subset)
    sensors,settings,removed,variances=pp.select_columns(train)
    cap=int(global_summary["regression"]["cap"]); window=int(global_summary["regression"]["selected"]["window"])

    print("FD004 global vs condition-aware normalization",flush=True)
    global_fp=FeaturePipeline(sensors,settings,representation="window",window=window,
                              include_settings=True,require_continuity=True).fit(train)
    condition_fp=ConditionAwareFeaturePipeline(sensors,config.SETTING_COLS,
        representation="window",window=window,include_settings=True,n_regimes=6).fit(train)
    normalization=[]; fitted={}
    for name,fp in (("global",global_fp),("condition_aware",condition_fp)):
        model,pred,scaled=fit_predict(fp,"ridge",train,tune,cap,alpha=10.0)
        metrics=pp.regression_metrics(np.minimum(tune.RUL,cap),pred,tune.RUL)
        normalization.append({"normalization":name,"model":"ridge",**{k:metrics[k] for k in ("MAE","RMSE","R2","PHM_mean")}})
        fitted[(name,"ridge")]=(fp,model,scaled)
    for family,params in (("gbr",{"n_estimators":150}),("rf",{"n_estimators":120,"min_samples_leaf":5})):
        model,pred,scaled=fit_predict(condition_fp,family,train,tune,cap,**params)
        metrics=pp.regression_metrics(np.minimum(tune.RUL,cap),pred,tune.RUL)
        normalization.append({"normalization":"condition_aware","model":family,
                              **{k:metrics[k] for k in ("MAE","RMSE","R2","PHM_mean")}})
        fitted[("condition_aware",family)]=(condition_fp,model,scaled)
    selected=min(normalization,key=lambda row:row["PHM_mean"])
    selected_fp,selected_model,selected_scaled=fitted[(selected["normalization"],selected["model"])]
    Xcal=selected_fp.transform(calib,scaled=selected_scaled)[0]
    pcal=np.clip(selected_model.predict(Xcal),0,cap)
    q=mr.conformal_quantile(np.abs(np.minimum(calib.RUL,cap)-pcal),config.CONFORMAL_COVERAGE)
    Xtest=selected_fp.transform(test,scaled=selected_scaled)[0]
    pred=np.clip(selected_model.predict(Xtest),0,cap); lower,upper=mr.conformal_interval(pred,q,upper_bound=cap)
    dedicated=pp.regression_metrics(np.minimum(test.RUL,cap),pred,test.RUL)
    dedicated["interval_coverage"],dedicated["interval_average_width"]=mr.coverage_and_width(
        np.minimum(test.RUL,cap),lower,upper)
    dedicated["ci95"]=pp.engine_bootstrap_regression(test.engine_id,np.minimum(test.RUL.to_numpy(),cap),pred,
                                                       config.BOOTSTRAP_DRAWS,config.RANDOM_SEED)

    print("FD004 transfer baselines",flush=True)
    transfer=[]; transfer_bundles={}
    for source in ("FD001","FD002","FD003"):
        row,bundle=source_transfer(source,test,cap=cap,window=window); transfer.append(row); transfer_bundles[source]=bundle
    transfer.append({"source":"FD004_dedicated","training":f"{selected['normalization']} {selected['model']}",
                     **{k:dedicated[k] for k in ("MAE","RMSE","R2","PHM_mean")}})

    regimes=condition_fp.assign_regimes(test)
    regime_rows=[]
    for regime in range(6):
        mask=regimes==regime
        metrics=pp.regression_metrics(np.minimum(test.RUL.to_numpy()[mask],cap),pred[mask],test.RUL.to_numpy()[mask])
        regime_rows.append({"regime":regime,"rows":int(mask.sum()),"engines":int(test.loc[mask,"engine_id"].nunique()),
                            **{k:metrics[k] for k in ("MAE","RMSE","R2","PHM_mean")}})

    lengths=test.groupby("engine_id").cycle.max(); length_band=pd.qcut(lengths.rank(method="first"),3,
        labels=["short","medium","long"]); row_band=test.engine_id.map(length_band)
    length_rows=[]
    for band in ("short","medium","long"):
        mask=row_band.to_numpy()==band; metrics=pp.regression_metrics(np.minimum(test.RUL.to_numpy()[mask],cap),pred[mask],test.RUL.to_numpy()[mask])
        length_rows.append({"length_band":band,"rows":int(mask.sum()),"engines":int(test.loc[mask,"engine_id"].nunique()),
                            **{k:metrics[k] for k in ("MAE","RMSE","R2","PHM_mean")}})

    train_final=train.sort_values(["engine_id","cycle"]).groupby("engine_id").tail(1)
    test_final=test.sort_values(["engine_id","cycle"]).groupby("engine_id").tail(1)
    proxy_scaler=StandardScaler().fit(train_final[sensors]); proxy_model=KMeans(n_clusters=2,n_init=20,
        random_state=config.RANDOM_SEED).fit(proxy_scaler.transform(train_final[sensors]))
    proxy_map=dict(zip(test_final.engine_id,proxy_model.predict(proxy_scaler.transform(test_final[sensors]))))
    proxy=test.engine_id.map(proxy_map).to_numpy(); proxy_rows=[]
    for cluster in (0,1):
        mask=proxy==cluster; metrics=pp.regression_metrics(np.minimum(test.RUL.to_numpy()[mask],cap),pred[mask],test.RUL.to_numpy()[mask])
        proxy_rows.append({"behavior_cluster":cluster,"rows":int(mask.sum()),"engines":int(test.loc[mask,"engine_id"].nunique()),
                           **{k:metrics[k] for k in ("MAE","RMSE","R2","PHM_mean")}})

    demo=pd.read_csv(config.DEMO_DIR/f"{subset}_predictions.csv.gz")
    stability=[]
    for regime in range(6):
        mask=regimes==regime; healthy=test.RUL.to_numpy()>30
        stability.append({"regime":regime,"rows":int(mask.sum()),
            "mean_anomaly_percentile":float(demo.loc[mask,"anomaly_percentile"].mean()),
            "crossing_rate":float((demo.loc[mask,"anomaly_percentile"]>=demo.loc[mask,"anomaly_threshold"]).mean()),
            "healthy_false_alert_rate":float(demo.loc[mask & healthy,"anomaly_persistent"].mean())})

    bonus={"subset":subset,"challenge":"six conditions + two fault modes",
           "protocol":"same engine split, metrics, horizons, uncertainty, and policy contract",
           "condition_normalization":{"n_regimes":6,"selection_role":"validation-tune",
                                      "comparison":normalization,"selected":selected},
           "dedicated_test_metrics":dedicated,"transfer_comparison":transfer,
           "performance_by_regime":regime_rows,"performance_by_sequence_length":length_rows,
           "performance_by_fault_behavior_proxy":proxy_rows,
           "fault_proxy_caution":"clusters are training-derived behavior proxies; official per-engine fault labels are unavailable",
           "anomaly_threshold_stability":stability,
           "dashboard_threshold_policy":"global validation-selected thresholds; regime results are diagnostic, not post-test retuning"}
    pp.write_json(config.RESULTS_DIR/"FD004_bonus_summary.json",bonus)
    pd.DataFrame(normalization).to_csv(config.RESULTS_DIR/"FD004_normalization_ablation.csv",index=False)
    pd.DataFrame(transfer).to_csv(config.RESULTS_DIR/"FD004_transfer_comparison.csv",index=False)
    pd.DataFrame(regime_rows).to_csv(config.RESULTS_DIR/"FD004_regime_metrics.csv",index=False)
    pd.DataFrame(stability).to_csv(config.RESULTS_DIR/"FD004_anomaly_regime_stability.csv",index=False)
    joblib.dump({"metadata":{"subset":"FD004","normalization":selected["normalization"],"window":window},
                 "fp":selected_fp,"model":selected_model,"scaled":selected_scaled,"cap":cap,"q":q},
                config.SYSTEMS_DIR/"FD004_condition_aware_regression.joblib",compress=3)

    fig,axes=plt.subplots(1,3,figsize=(15,4))
    norm=pd.DataFrame(normalization); axes[0].bar(norm.normalization+"\n"+norm.model,norm.RMSE,color="#2f6f8f"); axes[0].set(title="Validation normalization/model ablation",ylabel="RMSE")
    reg=pd.DataFrame(regime_rows); axes[1].bar(reg.regime.astype(str),reg.RMSE,color="#d28b26"); axes[1].set(title="FD004 test RMSE by regime",xlabel="regime")
    tr=pd.DataFrame(transfer); axes[2].bar(tr.source,tr.RMSE,color="#4b9b66"); axes[2].tick_params(axis="x",rotation=35); axes[2].set(title="Transfer vs dedicated",ylabel="RMSE")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR/"FD004_bonus_comparison.png",dpi=180); plt.close(fig)

    project_path=config.RESULTS_DIR/"project_summary.json"
    project=json.loads(project_path.read_text(encoding="utf-8")); project["bonus"]=bonus; pp.write_json(project_path,project)
    master=pd.read_csv(config.RESULTS_DIR/"master_results.csv")
    extra=pd.DataFrame([{"subset":"FD004","task":"RUL_bonus","model":selected["normalization"]+"_"+selected["model"],
                         "MAE":dedicated["MAE"],"RMSE":dedicated["RMSE"],"R2":dedicated["R2"],"PHM_mean":dedicated["PHM_mean"]}])
    pd.concat([master[master.subset!="FD004"],extra],ignore_index=True).to_csv(config.RESULTS_DIR/"master_results.csv",index=False)
    print("FD004 bonus completed.")
    return bonus


if __name__=="__main__": run()

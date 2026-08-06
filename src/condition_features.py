"""Operating-regime-aware causal feature pipeline for FD002/FD004."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import config
from features import FeaturePipeline


class ConditionAwareFeaturePipeline:
    """Normalize sensors within learned operating regimes before causal windows.

    Regime clustering, setting scaling, and per-regime sensor statistics are fitted on
    training engines only. KMeans assignment at inference uses the current operating
    settings and does not use future observations.
    """
    def __init__(self, keep_sensors, keep_settings=None, representation="window",
                 window=15, include_settings=True, n_regimes=6,
                 random_state=config.RANDOM_SEED):
        self.keep_sensors=list(keep_sensors)
        self.keep_settings=list(keep_settings or config.SETTING_COLS)
        self.representation=representation; self.window=int(window)
        self.include_settings=bool(include_settings); self.n_regimes=int(n_regimes)
        self.random_state=int(random_state); self._fitted=False

    def fit(self, df):
        settings=df[self.keep_settings].to_numpy(float)
        self.setting_scaler_=StandardScaler().fit(settings)
        self.regime_model_=KMeans(n_clusters=self.n_regimes,n_init=20,
                                  random_state=self.random_state).fit(
                                      self.setting_scaler_.transform(settings))
        regimes=self.regime_model_.labels_
        self.global_mean_=df[self.keep_sensors].mean().to_numpy(float)
        self.global_scale_=df[self.keep_sensors].std(ddof=0).replace(0,1).to_numpy(float)
        self.regime_stats_={}
        for regime in range(self.n_regimes):
            part=df.loc[regimes==regime,self.keep_sensors]
            mean=part.mean().to_numpy(float); scale=part.std(ddof=0).replace(0,1).fillna(1).to_numpy(float)
            self.regime_stats_[regime]=(mean,scale)
        normalized=self._normalize(df,regimes)
        self.inner_=FeaturePipeline(self.keep_sensors,self.keep_settings,
            representation=self.representation,window=self.window,
            include_settings=self.include_settings,require_continuity=True).fit(normalized)
        self.feature_names_=list(self.inner_.feature_names_); self._fitted=True
        return self

    def assign_regimes(self,df):
        if not hasattr(self,"regime_model_"): raise RuntimeError("fit before assign_regimes")
        return self.regime_model_.predict(self.setting_scaler_.transform(
            df[self.keep_settings].to_numpy(float)))

    def _normalize(self,df,regimes):
        out=df.copy()
        out[self.keep_sensors]=out[self.keep_sensors].astype(float)
        values=out[self.keep_sensors].to_numpy(float); normalized=np.empty_like(values)
        for regime in np.unique(regimes):
            mask=regimes==regime; mean,scale=self.regime_stats_.get(int(regime),(self.global_mean_,self.global_scale_))
            normalized[mask]=(values[mask]-mean)/scale
        out.loc[:,self.keep_sensors]=normalized
        return out

    def transform(self,df,scaled=True):
        if not self._fitted: raise RuntimeError("fit before transform")
        regimes=self.assign_regimes(df); return self.inner_.transform(self._normalize(df,regimes),scaled=scaled)

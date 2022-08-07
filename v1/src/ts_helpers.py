#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from collections import namedtuple

import numpy as np
import pandas as pd
import statsmodels.tsa.api as smt


def check_stationarity(
    ts: pd.Series, ts_name: str, adf_alpha: float = 0.05
) -> pd.DataFrame:
    ADF = namedtuple("ADF", "adf pvalue usedlag nobs critical icbest")
    adf_dict = ADF(*smt.adfuller(ts.dropna()))._asdict()
    df_adf = pd.DataFrame.from_dict(adf_dict, orient="index")
    df_adf = df_adf[~df_adf.index.isin(["critical"])]
    df_adf.index = [
        "ADF_Statistic",
        "p-value",
        "num_lags_used",
        "n_observations_used",
        "IC_for_best",
    ]
    pvalue_comparison = (df_adf.loc["p-value"] < adf_alpha).iloc[0]
    d_crit_vals = {k: v for k, v in adf_dict["critical"].items()}
    df_adf = pd.concat(
        [
            pd.DataFrame.from_dict({"name": ts_name}, orient="index"),
            df_adf,
            pd.DataFrame.from_dict(d_crit_vals, orient="index"),
            pd.DataFrame.from_dict(
                {"likely stationary": pvalue_comparison}, orient="index"
            ),
        ],
        axis=0,
    )
    df_adf.columns = ["Value"]
    df_adf["module"] = "adfuller"
    df_adf.loc[
        "likely stationary", "module"
    ] = f"p-value {'<' if pvalue_comparison else '>'} {adf_alpha}"
    df_adf.loc["name", "module"] = np.nan
    return df_adf

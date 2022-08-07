#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def median_filter_outliers(df_train, df_val, window=24 * 1, std=3):
    # train
    medianv = df_train["y"].rolling(window, center=True).median()
    stdv = df_train["y"].rolling(window, center=True).std()

    # transform train
    mask = (df_train.loc[:, "y"] >= medianv + (std * stdv)) | (
        df_train.loc[:, "y"] <= medianv - (std * stdv)
    )
    df_train.loc[mask, "y"] = np.nan

    # transform val
    num_obs_from_end = len(df_val)
    medianv_val = medianv.iloc[-num_obs_from_end:]
    stdv_val = stdv.iloc[-num_obs_from_end:]
    medianv_val.index = df_val.index
    stdv_val.index = df_val.index
    mask = (df_val.loc[:, "y"] >= medianv_val + (std * stdv_val)) | (
        df_val.loc[:, "y"] <= medianv_val - (std * stdv_val)
    )
    df_val.loc[mask, "y"] = np.nan
    return [df_train, df_val]

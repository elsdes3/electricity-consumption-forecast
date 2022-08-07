#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import sklearn.metrics as skm


def rmspe_error(y_true, y_pred):
    rmspe_val = np.sqrt(
        np.mean(np.square(((y_true - y_pred) / y_true)), axis=0)
    )
    return rmspe_val * 100


def smape_error(y_true, y_pred):
    n = len(y_pred)
    num = np.abs(y_pred - y_true)
    denom = np.abs(y_pred) + np.abs(y_true)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def score_predictions(y_true, y_pred, prediction_type="pred", get_r2=False):
    rmse_score = skm.mean_squared_error(y_true, y_pred, squared=False)
    mse_score = skm.mean_squared_error(y_true, y_pred)
    # mape_score = skm.mean_absolute_percentage_error(y_true, y_pred) * 100
    mae_score = skm.mean_absolute_error(y_true, y_pred)
    smape_score = smape_error(y_true, y_pred)
    scores = {
        "rmse": rmse_score,
        "mae": mae_score,
        # "mape(%)": mape_score,
        "smape(%)": smape_score,
        "mse": mse_score,
        "type": prediction_type,
    }
    if get_r2:
        scores.update({"r2": skm.r2_score(y_true, y_pred)})
    if y_true.min() > 0:
        scores.update({"rmspe(%)": rmspe_error(y_true, y_pred)})
    return scores


def groupwise_score_predictions(df_group, true="y", pred="ypred"):
    scores = score_predictions(df_group[true], df_group[pred])
    return pd.Series(scores)

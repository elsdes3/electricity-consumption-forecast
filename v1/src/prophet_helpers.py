#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from prophet import Prophet

from src.custom_estimators import CustomNaiveRegressor
from src.metrics_helpers import score_predictions
from src.processing_helpers import median_filter_outliers
from src.prophet_utils import suppress_stdout_stderr
from src.visualization_helpers import generate_plots


def analyze(
    m,
    train,
    test,
    horizon,
    test_date_start,
    test_date_end,
    nums_fcast_params,
    categoricals,
    primary_metric="rmse",
    show_plots=True,
):
    with suppress_stdout_stderr():
        m.fit(train)

    # Predict
    future = m.make_future_dataframe(
        periods=((horizon - 1) * 24) + 23 + 1, freq="H", include_history=False
    )
    # print(list(future))
    # display(future.head(2))
    future_nums = CustomNaiveRegressor(**nums_fcast_params).fit_predict(train)
    # print(list(future_nums))
    # display(future_nums.head(2).append(future_nums.tail(2)))
    future = future_nums.merge(future, on=["ds"]).merge(
        test[["ds"] + categoricals], on="ds"
    )
    # print(list(future))
    # display(future.head(2).append(future.tail(2)))
    forecast = m.predict(future)
    # display(forecast)
    # display(
    #     forecast[
    #         (forecast["ds"] >= test_date_start)
    #         & (forecast["ds"] <= test_date_end)
    #     ]
    # )
    # display(test[["ds", "y"]].head(2).append(test[["ds", "y"]].tail(2)))
    low_mask = forecast["ds"] >= test_date_start
    high_mask = forecast["ds"] <= test_date_end
    future_forecast = forecast[low_mask & high_mask].merge(
        test[["ds", "y"]], on="ds", how="left"
    )
    # display(future_forecast.head(2).append(future_forecast.tail(2)))

    # Calculate residual
    future_no_nan = future_forecast.dropna(how="any", subset=["y", "yhat"])
    residual = future_no_nan["y"] - future_no_nan["yhat"]
    # display(residual)

    # Score predictions
    # print(len(future_forecast["y"]), len(future_forecast["yhat"]))
    scores_dict = score_predictions(
        future_no_nan["y"], future_no_nan["yhat"], get_r2=True
    )

    # Plot Prophet model components
    if show_plots:
        generate_plots(m, residual, scores_dict, forecast, primary_metric)
    return [forecast, future_forecast, scores_dict]


def train_score_model(
    df_train_val_test,
    params,
    country,
    seasonalities,
    custom_seasonalities,
    horizon,
    test_start,
    test_end,
    nums_fcast_params,
    categoricals,
    primary_metric="rmse",
):
    m = Prophet(**params).add_country_holidays(country_name=country)
    weather_attrs_to_forecast = nums_fcast_params["fcast_attrs"]
    if weather_attrs_to_forecast:
        for regressor in nums_fcast_params["fcast_attrs"]:
            m.add_regressor(regressor)
    if seasonalities:
        for sparams in seasonalities:
            m.add_seasonality(**sparams)
    if custom_seasonalities:
        for custom_seasonality_params in custom_seasonalities:
            m.add_seasonality(**custom_seasonality_params)

    df_train_val, df_test = median_filter_outliers(
        df_train_val_test[0].copy(), df_train_val_test[1].copy(), 24, std=8
    )

    _, future_forecast, scores_dict = analyze(
        m,
        df_train_val,
        df_test,
        horizon,
        test_start,
        test_end,
        nums_fcast_params,
        categoricals,
        primary_metric,
        False,
    )
    df_scores = (
        pd.DataFrame.from_dict(scores_dict, orient="index")
        .T.assign(country=country)
        .assign(test_start=test_start)
        .assign(test_end=test_end)
        .assign(params=str(params))
        .assign(weather_attrs_to_forecast=",".join(weather_attrs_to_forecast))
        .assign(seasonalities=",".join([s["name"] for s in seasonalities]))
        .assign(
            custom_seasonalities=",".join(
                [cs["name"] for cs in custom_seasonalities]
            )
        )
    )
    return [future_forecast.assign(country=country), df_scores]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_array, check_X_y


class CustomNaiveRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        naive_strategy="slice",
        lookback=366 * 24,
        horizon=91 * 24,
        fcast_attrs=["temperature"],
        datetime_colname="date",
        freq="H",
    ):
        self.naive_strategy = naive_strategy
        self.lookback = lookback
        self.horizon = horizon
        self.fcast_attrs = fcast_attrs
        self.datetime_colname = datetime_colname
        self.freq = freq

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        # make forecast
        if self.naive_strategy == "slice":
            lb = self.lookback
            X_lookback = X[-lb:]
            X_pred = X_lookback[: (self.horizon + 1)].reset_index(drop=True)

        # Increment forecast datetimes forward in time by <horizon> time steps
        fstart = X[self.datetime_colname].max() + pd.to_timedelta(
            1, unit=self.freq
        )
        fend = fstart + pd.to_timedelta(self.horizon, unit=self.freq)
        X_pred[self.datetime_colname] = pd.date_range(
            fstart, fend, freq=self.freq
        )
        return X_pred[[self.datetime_colname] + self.fcast_attrs]

    def fit_predict(self, X, y=None, **kwargs):
        self = self.fit(X)
        return self.predict(X)

    def get_params(self, deep=True):
        return {
            "naive_strategy": self.naive_strategy,
            "horizon": self.horizon,
            "fcast_attrs": self.fcast_attrs,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class DFColumnRenamer(BaseEstimator, TransformerMixin):
    def __init__(self, renaming_dict):
        self.renaming_dict = renaming_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        X_trans = X.rename(columns=self.renaming_dict)
        self.feature_names_ = X_trans.columns.tolist()
        return X_trans

    def fit_transform(self, X, y=None, **kwargs):
        self = self.fit(X, y)
        return self.transform(X)

    def get_features(self):
        return self.feature_names_


class MultiTSCustomNaiveRegressor(BaseEstimator, RegressorMixin):
    """
    Notes
    -----
    1. The length of the datetimes covered by the start and end of each
       naive cutoff pair must match the unique dates in the
       - forecast horizon
       - out-of-sample data, on which .predict() will be called
    2. .predict() can be called on the out-of-sample data only

    Usage
    -----
    # Inputs
    train_val_start = "1800-01-01"
    train_val_end = "1802-09-30"
    test_start = "1802-10-01"
    test_end = "1802-12-31"
    naive_cutoffs = [
        ["1800-10-01", "1800-12-31"], ["1801-10-01", "1801-12-31"]
    ]
    index_name = "ds"
    countries = ["BE", "CZ"]
    renamer = {"load": "y"}

    # Generate data
    indx = pd.concat(
        [
            pd.date_range("1800", "1803", name="ds").to_frame()
            for c in countries
        ]
    )
    df = pd.DataFrame(index=indx).reset_index().rename(
        columns={"index": "ds"}
    )
    cnt_list = [[c] * int(len(indx) / len(countries)) for c in countries]
    df["country"] = np.array(cnt_list).flatten()
    df["load"] = np.random.rand(len(df), 1)
    df = df[["country", "ds", "load"]]

    # Split
    df_train_val_naive = df[
        (df[index_name] >= train_val_start)
        & (df[index_name] <= train_val_end)
    ].reset_index(drop=True)[['country', index_name, 'load']]
    df_test_naive = df[
        (df[index_name] >= test_start) & (df[index_name] <= test_end)
    ].reset_index(drop=True)[['country', index_name, 'load']]

    # Train
    reg = MultiTSCustomNaiveRegressor(naive_cutoffs, 'ds', 'country')
    est = Pipeline(
        [('rename', DFColumnRenamer(renamer)), ('reg', reg)]
    )
    est.fit(df_train_val_naive)

    # Predict
    df_naive_pred = est.predict(df_test_naive)
    """

    def __init__(self, naive_cutoffs, index_name="ds", ts_name_col="country"):
        self.naive_cutoffs = naive_cutoffs
        self.index_name = index_name
        self.ts_name_col = ts_name_col

    def fit(self, X, y=None, **fit_kws):
        X = X.set_index(self.ts_name_col, append=True)
        Xc, yc = check_X_y(X[[self.index_name]], X["y"].fillna(-999))
        X = (
            pd.DataFrame(Xc, index=X.index, columns=[self.index_name])
            .assign(y=yc)
            .replace(-999, np.nan)
        ).reset_index(level=1)

        # Assemble list of historical data before each specified cutoff
        dfs_pred_ = []
        for naive_cutoff_ in self.naive_cutoffs:
            low = X[self.index_name] >= naive_cutoff_[0]
            high = X[self.index_name] <= naive_cutoff_[1]
            df_pred = X[low & high][[self.ts_name_col, "ds", "y"]]
            dfs_pred_.append(df_pred)

        # Concatenate all data before cutoffs
        self.df_pred_ = pd.concat(dfs_pred_, ignore_index=True)
        return self

    def predict(self, X):
        X = X.set_index(self.ts_name_col, append=True)[[self.index_name]]
        X = pd.DataFrame(
            check_array(X), index=X.index, columns=list(X)
        ).reset_index(1)

        # Assemble out-of-sample dates for each timeseries being forecast
        nrows = X[self.ts_name_col].nunique() * len(self.naive_cutoffs)

        # Append column to concatenated data with future dates for which a
        # forecast is required
        df_pred_ = self.df_pred_.assign(
            ds_true=list(X[self.index_name].unique()) * nrows
        )

        # GROUP BY over concatenated data by country and future date columns
        # and compute the mean (aggregation) of historical y values
        groupby_cols = [self.ts_name_col, f"{self.index_name}_true"]
        df_pred_ = df_pred_.groupby(groupby_cols, as_index=False)["y"].mean()

        # Rename columns in prediction
        renaming_dict = {
            "y": "yhat",
            f"{self.index_name}_true": self.index_name,
        }
        df_pred_ = df_pred_.rename(columns=renaming_dict)
        return df_pred_

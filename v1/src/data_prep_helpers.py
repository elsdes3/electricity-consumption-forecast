#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd


def add_corona_dates(df, index_name, strategy=["during_corona", "no_corona"]):
    """
    Inputs
    ------
    strategy : List
        division of datetimes based on stages of corona; acceptable strategies
        are one of the following (order in list does not matter)
        - ['during_corona', 'no_corona']
        - ['pre_corona', 'during_corona', 'post_corona']

    SOURCE
    ------
    https://github.com/facebook/prophet/issues/1416#issuecomment-618553502
    """
    d_corona = {
        "BE": [
            pd.to_datetime("2020-03-07 00:00:00"),
            pd.to_datetime("2020-04-12 23:00:00"),
        ],
        "CH": [
            pd.to_datetime("2020-03-07 00:00:00"),
            pd.to_datetime("2020-04-12 23:00:00"),
        ],
        "CZ": [
            pd.to_datetime("2020-03-14 00:00:00"),
            pd.to_datetime("2020-04-12 23:00:00"),
        ],
        "DE": [
            pd.to_datetime("2020-03-14 00:00:00"),
            pd.to_datetime("2020-04-12 23:00:00"),
        ],
        "ES": [
            pd.to_datetime("2020-03-14 00:00:00"),
            pd.to_datetime("2020-04-12 23:00:00"),
        ],
        "FR": [
            pd.to_datetime("2020-03-07 00:00:00"),
            pd.to_datetime("2020-04-12 23:00:00"),
        ],
        "HR": [
            pd.to_datetime("2020-03-21 00:00:00"),
            pd.to_datetime("2020-04-12 23:00:00"),
        ],
        "IT": [
            pd.to_datetime("2020-03-14 00:00:00"),
            pd.to_datetime("2020-04-12 23:00:00"),
        ],
        "NL": [
            pd.to_datetime("2020-03-14 00:00:00"),
            pd.to_datetime("2020-04-12 23:00:00"),
        ],
        "PL": [
            pd.to_datetime("2020-03-14 00:00:00"),
            pd.to_datetime("2020-04-12 23:00:00"),
        ],
    }
    df_corona = (
        pd.DataFrame.from_dict(d_corona, orient="index")
        .reset_index()
        .rename(
            columns={0: "corona_start", 1: "corona_end", "index": "country"}
        )
    )
    df = df.merge(df_corona, on="country", how="left")

    # Add corona periods based on specified strategy
    strategies_dict = {
        "dn": ["during_corona", "no_corona"],
        "pdp": ["pre_corona", "during_corona", "post_corona"],
    }
    if set(strategy) == set(strategies_dict["dn"]):
        df["no_corona"] = (df[index_name] < df["corona_start"]) | (
            df[index_name] > df["corona_end"]
        )
    elif set(strategy) == set(strategies_dict["pdp"]):
        df["pre_corona"] = df[index_name] < df["corona_start"]
        df["post_corona"] = df[index_name] > df["corona_end"]
    else:
        strategies = ""
        for _, v in strategies_dict.items():
            strategies += "['" + "', '".join(map(str, v)) + "'], "
        strategies = strategies.rstrip(", ")
        raise Exception(
            f"Unsupported corona strategy. Expected one of: {strategies}"
        )
    df["during_corona"] = (df[index_name] >= df["corona_start"]) & (
        df[index_name] <= df["corona_end"]
    )
    return df

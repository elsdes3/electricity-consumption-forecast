#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from IPython.display import display


def show_df(df, nrows=5):
    display(
        df.head(nrows)
        .append(df.tail(nrows))
        .style.set_caption(f"First & Last {nrows} rows")
    )


def show_df_dtypes_nans(df):
    display(
        df.isna()
        .sum()
        .rename("num_missing")
        .to_frame()
        .merge(
            df.dtypes.rename("dtypes").to_frame(),
            left_index=True,
            right_index=True,
            how="left",
        )
        .style.set_caption("Column Datatypes and Missing Values")
    )

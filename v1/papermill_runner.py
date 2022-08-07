#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import shlex
import subprocess
from datetime import datetime
from typing import Dict, List

import papermill as pm

PROJ_ROOT_DIR = os.path.abspath(os.getcwd())
data_dir = os.path.join(PROJ_ROOT_DIR, "data")
output_notebook_dir = os.path.join(PROJ_ROOT_DIR, "executed_notebooks")

raw_data_path = os.path.join(data_dir, "raw")

zero_dict_nb_name = "0_get_data.ipynb"
one_dict_nb_name = "1_naive.ipynb"
two_dict_nb_name = "2_train.ipynb"
two_v2_dict_nb_name = "02_train_v2.ipynb"

zero_dict = dict(
    opsd_data_url=(
        "https://data.open-power-system-data.org/time_series/latest/"
        "time_series_60min_singleindex.csv"
    ),
    load_col_name_str="load_actual_entsoe_transparency",
    start_date="2015",
    end_date="2020",
    sstr=[
        "BE",
        "CH",
        "CZ",
        "DE",
        "FR",
        "ES",
        "HR",
        "IT",
        "NL",
        "PL",
        "AT",
        "DK",
        "SK",
        "GB_GBN",
        "RO",
    ],
    get_weather=True,
    use_demographics=False,
    comfort_threshold=20,
    indicators_wanted=[
        "Labor force, total",
        "Population, total",
        "Trademark applications, total",
        "Listed domestic companies, total",
    ],
    cntry_renaming_dict={
        "country": {
            "BEL": "BE",
            "HRV": "HR",
            "CZE": "CZ",
            "FRA": "FR",
            "DEU": "DE",
            "ITA": "IT",
            "NLD": "NL",
            "POL": "PL",
            "ESP": "ES",
            "CHE": "CH",
        }
    },
)
one_dict = dict(
    index_name="utc_timestamp",
    train_val_start="2015-01-01 00:00:00",
    train_val_end="2020-07-01 23:00:00",
    test_start="2020-07-02 00:00:00",
    test_end="2020-09-30 23:00:00",
    naive_cutoffs=[
        ["2016-06-30 00:00:00", "2016-09-28 23:00:00"],
        ["2017-06-29 00:00:00", "2017-09-27 23:00:00"],
        ["2018-07-05 00:00:00", "2018-10-03 23:00:00"],
        ["2019-07-04 00:00:00", "2019-10-02 23:00:00"],
    ],
    renamer={"utc_timestamp": "ds", "load": "y"},
)
two_dict = dict(
    index_name="utc_timestamp",
    horizon=91,
    freq="H",
    primary_metric="rmse",
    train_start="2017-01-01 00:00:00",
    train_end="2019-07-01 23:00:00",
    val_start="2019-07-02 00:00:00",
    val_end="2019-09-30 23:00:00",
    train_val_end="2020-07-01 23:00:00",
    test_start="2020-07-02 00:00:00",
    test_end="2020-09-30 23:00:00",
    country="FR",
    lookback=365,
    weather_attrs_to_forecast=["temp"],
)
two_dict_v2 = dict(
    start_date="2015",
    end_date="2020",
    opsd_data_url=(
        "https://data.open-power-system-data.org/time_series/"
        "latest/time_series_60min_singleindex.csv"
    ),
    load_col_name_str="load_actual_entsoe_transparency",
    seasons={
        "1": "winter",
        "2": "winter",
        "3": "winter",
        "4": "spring",
        "5": "spring",
        "6": "spring",
        "7": "summer",
        "8": "summer",
        "9": "summer",
    },
    country_name="DE",
)


def run_cmd(cmd: str) -> None:
    print(cmd)
    process = subprocess.Popen(
        shlex.split(cmd), shell=False, stdout=subprocess.PIPE
    )
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(str(output.strip(), "utf-8"))
    _ = process.poll()


def papermill_run_notebook(
    nb_dict: Dict, output_notebook_dir: str = "executed_notebooks"
) -> None:
    """Execute notebook with papermill"""
    for notebook, nb_params in nb_dict.items():
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_nb = os.path.basename(notebook).replace(
            ".ipynb", f"-{now}.ipynb"
        )
        print(
            f"\nInput notebook path: {notebook}",
            f"Output notebook path: {output_notebook_dir}/{output_nb} ",
            sep="\n",
        )
        for key, val in nb_params.items():
            print(key, val, sep=": ")
        pm.execute_notebook(
            input_path=notebook,
            output_path=f"{output_notebook_dir}/{output_nb}",
            parameters=nb_params,
        )


def run_notebooks(
    notebook_list: List, output_notebook_dir: str = "executed_notebooks"
) -> None:
    """Execute notebooks from CLI.
    Parameters
    ----------
    nb_dict : List
        list of notebooks to be executed
    Usage
    -----
    > import os
    > PROJ_ROOT_DIR = os.path.abspath(os.getcwd())
    > one_dict_nb_name = "a.ipynb
    > one_dict = {"a": 1}
    > run_notebook(
          notebook_list=[
              {os.path.join(PROJ_ROOT_DIR, one_dict_nb_name): one_dict}
          ]
      )
    """
    for nb in notebook_list:
        papermill_run_notebook(
            nb_dict=nb, output_notebook_dir=output_notebook_dir
        )


if __name__ == "__main__":
    PROJ_ROOT_DIR = os.getcwd()
    nb_dict_list = [
        # zero_dict,
        # one_dict,
        # two_dict,
        two_dict_v2,
    ]
    nb_name_list = [
        # zero_dict_nb_name,
        # one_dict_nb_name,
        # two_dict_nb_name,
        two_v2_dict_nb_name,
    ]
    notebook_list = [
        {os.path.join(PROJ_ROOT_DIR, nb_name): nb_dict}
        for nb_dict, nb_name in zip(nb_dict_list, nb_name_list)
    ]
    run_notebooks(
        notebook_list=notebook_list, output_notebook_dir=output_notebook_dir
    )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Programmatic execution of notebooks."""


import os
import shlex
import subprocess
from datetime import datetime
from typing import Dict, List

import papermill as pm

# pylint: disable=invalid-name,dangerous-default-value, redefined-outer-name

PROJ_ROOT_DIR = os.path.abspath(os.getcwd())
output_notebook_dir = os.path.join(PROJ_ROOT_DIR, "executed_notebooks")

zero_dict_nb_name = "01_get_data_analyze.ipynb"

zero_dict = dict(
    start_date="2015",
    end_date="2020",
    opsd_data_url=(
        "https://data.open-power-system-data.org/time_series/latest/"
        "time_series_60min_singleindex.csv"
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
    country_names=[
        "DE",
        "CZ",
        "BE",
        "HR",
        "PL",
        "IT",
        "SK",
        "RO",
        "ES",
        "AT",
        "DK",
        "GB_GBN",
    ],
)


def run_cmd(cmd: str) -> None:
    """."""
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
    nb_dict_list = [zero_dict]
    nb_name_list = [zero_dict_nb_name]
    notebook_list = [
        {os.path.join(PROJ_ROOT_DIR, nb_name): nb_dict}
        for nb_dict, nb_name in zip(nb_dict_list, nb_name_list)
    ]
    run_notebooks(
        notebook_list=notebook_list, output_notebook_dir=output_notebook_dir
    )

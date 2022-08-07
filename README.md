# [Forecasting Electricity Consumption](#forecasting-electricity-consumption)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/elsdes3/electricity-consumption-forecast)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elsdes3/electricity-consumption-forecast/master/01_get_data_analyze.ipynb)
![CI](https://github.com/elsdes3/electricity-consumption-forecast/workflows/CI/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/mit)
![OpenSource](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![prs-welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)

## [Table of Contents](#table-of-contents)
0. [About](#about)
   - [Objective](#objective)
   - [Business Use-Case](#business-use-case)
   - [Implementation Details](#implementation-details)
   - [Data](#data)
   - [Model Requirements](#model-requirements)
1. [Notebooks](#notebooks)
2. [Project Organization](#project-organization)
3. [References](#references)

## [About](#about)
### [Objective](#objective)
This project assesses the feasibility of using a ML model to forecast total electricity consumption data per country in 12 countries across Europe.

### [Business Use-Case](#business-use-case)
All countries in Europe have one TSO, except for [Germany which has four](https://www.next-kraftwerke.com/knowledge/european-tsos-list#:~:text=%20%20%20%20Country%20%20%20,%20%2072%20%2043%20more%20rows%20). A forecast of electricity usage over the medium term is useful for TSOs when setting up of a schedule for grid maintenance on a national scale. This forecast horizon helps TSOs plan their nation-wide [maintenance schedules](https://www.sciencedirect.com/science/article/abs/pii/S0378779620304855) by anticipating electricity demand during planned maintenance periods.

### [Implementation Details](#implementation-details)
Train ML model to make and evaluate a medium-term aggregated (total) electricity demand forecast per country, for 12 countries in Europe.

The feasibility of adopting an ML-based forecast will be assessed during a pilot study covering 2019-09-01 to 2019-09-30 in all 12 countries, during which nationwide plant maintenance is planned.

When maintenance is completed, the grid should be brought back to its initial state as soon as possible. For this reason, the data frequency should be hourly.

### [Data](#data)
Hourly electricity consumption data is available from 2015 to 2020. ML model development should be limited to using the bare minimum data for model training. So, the ML model will be trained using data from 2018-01-01 to 2019-08-31. This will capture any annual, and higher frequency, patterns in electricity usage.

### [Model Requirements](#model-requirements)
The simplest ML model should be used in order to aid in
- model explainability
- prospective model deployment (should this be warranted based on performance of the forecast)

## [Notebooks](#notebooks)
1. `01_get_data_analyze.ipynb` ([view](https://nbviewer.jupyter.org/github/elsdes3/electricity-consumption-forecast/blob/main/01_get_data_analyze.ipynb))
   - retrieving
     - raw electricity consumption data from open data portal
     - demographics data from World Bank
     - weather data from Meteostat
   - exploratory timeseries analysis of electricity load and weather data
   - feature engineering
   - feature selection
   - generate and score a naive forecast for each country
   - split data into four splits
   - manually tune hyperparameters using coarse grid search with train and validation splits
   - with model trained using best hyperparameters on the train+validation split, generate forecasts of test split separately for each country
   - evaluate forecasts of the test split

## [Project Organization](#project-organization)

    ├── LICENSE
    ├── .env                          <- environment variables (verify this is in .gitignore)
    ├── .gitignore                    <- files and folders to be ignored by version control system
    ├── .pre-commit-config.yaml       <- configuration file for pre-commit hooks
    ├── .github
    │   ├── workflows
    │       └── integrate.yml         <- configuration file for CI build on Github Actions
    │       └── codeql-analysis.yml   <- configuration file for security scanning on Github Actions
    ├── Makefile                      <- Makefile with commands like `make lint` or `make build`
    ├── README.md                     <- The top-level README for developers using this project.
    ├── environment.yml               <- configuration file to create environment to run project on Binder
    ├── data
    │   ├── raw                       <- Scripts to download or generate data
    |   └── processed                 <- merged and filtered data, sampled at daily frequency
    ├── *.ipynb                       <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                    and a short `-` delimited description, e.g. `1.0-jqp-initial-data-exploration`.
    ├── requirements.txt              <- base packages required to execute all Jupyter notebooks (incl. jupyter)
    ├── src                           <- Source code for use in this project.
    │   ├── __init__.py               <- Makes tostreetcar a Python module
    │   └── *.py                      <- Scripts to use in analysis for pre-processing, visualization, training, etc.
    ├── papermill_runner.py           <- Python functions that execute system shell commands.
    └── tox.ini                       <- tox file with settings for running tox; see https://tox.readthedocs.io/en/latest/

## [References](#references)
1. Open Power System Data. 2020. Data Package Time series. Version 2020-10-06. https://doi.org/10.25832/time_series/2020-10-06. (Primary data from various sources, for a complete list see URL).
2.  Sean J. Taylor & Benjamin Letham (2018) Forecasting at Scale, The American Statistician, 72:1, 37-45, DOI: 10.1080/00031305.2017.1380080

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import time
from multiprocessing import cpu_count

import pandas as pd
from joblib import Parallel, delayed
from meteostat import Daily, Hourly, Point, Stations


def get_single_station_weather(
    k,
    row,
    num_stations,
    ks_wanted,
    weather_data_dir,
    location_type="from_station_name",
    time_colname="startdatehour",
    get_hourly=True,
    verbose=False,
):
    start, end, lat, lon, from_station_id, c, tz = [row[k] for k in ks_wanted]
    # print(start, end)
    if verbose:
        print(f"({k+1}/{num_stations}) {from_station_id}...", end="")
    start_time = time.time()
    if get_hourly:
        data = Hourly(Point(lat, lon), start, end, timezone=tz)
    else:
        data = Daily(Point(lat, lon), start, end)
    data = data.normalize()
    data = data.interpolate()
    data = data.fetch().reset_index().rename(columns={"time": time_colname})
    end_time = time.time()
    if get_hourly:
        # Convert time column in weather data to GMT / UTC timezone
        data[time_colname] = pd.to_datetime(data[time_colname], utc=True)
        # Convert time column in weather data to station's timezone
        data[time_colname] = data[time_colname].dt.tz_convert(tz)
        # Remove timezone from time column
        data[time_colname] = data[time_colname].dt.tz_localize(None)
        # DST 1/2 - Drop second duplicated entry in each Nov
        data = data.drop_duplicates(subset=[time_colname], keep="first")
        # DST 2/2 - Insert missing entry in each Mar with 'time' interpolation
        data = (
            data.set_index(time_colname)
            .resample("H")
            .mean()
            .interpolate(method="time")
            .reset_index()
        )
    if verbose:
        print(f"done ({(end_time - start_time):.2f}s)...", end="")

    data[location_type] = from_station_id
    data = (
        data.assign(country=c).assign(year=data[time_colname].dt.year)
        # .drop(columns=["hour"])
    )
    if get_hourly:
        data["timezone"] = tz
    parquet_filepath = os.path.join(
        weather_data_dir, f"{from_station_id.replace(' ', '_')}.parquet"
    )
    try:
        if verbose:
            print(
                f"saving to {os.path.basename(parquet_filepath+'.gzip')}",
                end="...",
            )
        data.to_parquet(
            parquet_filepath + ".gzip",
            engine="auto",
            index=False,
            compression="gzip",
        )
        if verbose:
            print("done.")
    except Exception as e:
        if verbose:
            print(str(e))

    data[time_colname] = pd.to_datetime(data[time_colname])
    # for k, v in zip(
    #     keys_wanted[:-1],
    #     [start, end, lat, lon],
    # ):
    #     data[k] = v
    return data


def get_weather_by_station(
    df_stations,
    keys_wanted,
    weather_data_dir,
    location_type="from_station_name",
    time_colname="startdatehour",
    get_hourly=True,
):
    num_stations = df_stations[location_type].nunique()
    executor = Parallel(n_jobs=cpu_count(), backend="multiprocessing")
    tasks = (
        delayed(get_single_station_weather)(
            k,
            row,
            num_stations,
            keys_wanted,
            weather_data_dir,
            location_type,
            time_colname,
            get_hourly,
        )
        for k, (_, row) in enumerate(df_stations.iterrows())
    )
    dfs_weather = executor(tasks)
    dfs_weather = pd.concat(dfs_weather, ignore_index=True)
    return dfs_weather


def get_airport_weather_station_metadata(
    station_lookup_ref_point_by_country_dict,
    station_cols_wanted,
    radius=None,
):
    dfs_stations = []
    for k, v in station_lookup_ref_point_by_country_dict.items():
        stations = Stations()
        stations = stations.nearby(v[0], v[1], radius=radius)
        station = stations.fetch()
        station = station[station["country"] == k]
        station = station[station["name"].str.contains(v[2])]
        station = station[station["hourly_end"].dt.year >= 2020]
        # display(station)
        dfs_stations.append(station[station_cols_wanted])
    df_stations = pd.concat(dfs_stations, ignore_index=True)
    df_stations = df_stations.rename(columns={"name": "from_station_name"})
    df_stations["from_station_name"] = df_stations[
        "from_station_name"
    ].str.replace(" / ", "_")
    return df_stations

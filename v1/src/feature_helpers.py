#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def calculate_daylight(day: int, latitude: float = 53.551086) -> float:
    """
    Calculate number of hours of daylight in a day
    Parameters
    ----------
    day : integer (required)
        day of the week by number, starting at 0 (Monday)
    latitude : float (required)
        latitude at which number of daylight hours in a day is required
        (default is taken as latitude of Hamburg since weather data is also
        taken for this city:
        https://www.latlong.net/place/hamburg-germany-8766.html)
    Returns
    -------
    daylightamount : float
       number of hours of daylight in a day
    SOURCE: http://mathforum.org/library/drmath/view/56478.html
    """
    ast = 0.9671396
    abc = 0.2163108
    const = 0.39795
    d_shift = day - 186
    P = np.arcsin(
        const * np.cos(abc + 2 * np.arctan(ast * np.tan(0.00860 * d_shift)))
    )
    pi = np.pi
    numerator = (0.8333 * pi / 180) + np.sin(latitude * pi / 180) * np.sin(P)
    denominator = np.cos(latitude * pi / 180) * np.cos(P)
    daylightamount = 24 - (24 / pi) * np.arccos(
        (np.sin(numerator) / denominator)
    )
    return daylightamount


def get_too_hot_cold(data, threshold=20):
    # METHOD 1 - use GROUP BY
    temps = data["temp"] - threshold
    data["too_hot"] = temps.mask(temps < 0, other=0).abs()
    data["too_cold"] = temps.mask(temps > 0, other=0).abs()
    # # METHOD 2 - do not use GROUP BY
    # hcs = []
    # for c in data["country"].unique():
    #     temps = data[data["country"] == c]["temp"] - threshold
    #     too_hot = temps.mask(temps < 0, other=0).abs()
    #     too_cold = temps.mask(temps > 0, other=0).abs()
    #     hc = (
    #         too_hot.rename("too_hot")
    #         .to_frame()
    #         .merge(
    #             too_cold.rename("too_cold").to_frame(),
    #             left_index=True,
    #             right_index=True,
    #             how="left",
    #         )
    #         .merge(
    #             data[data["country"] == c][["ds"]],
    #             left_index=True,
    #             right_index=True,
    #             how="left",
    #         )
    #         .assign(country=c)
    #     )
    #     hc = hc.sort_values(["ds"])
    #     hcs.append(hc)
    # data = data.merge(
    #     pd.concat(hcs, ignore_index=True), on=["country", "ds"], how="left"
    # )
    return data

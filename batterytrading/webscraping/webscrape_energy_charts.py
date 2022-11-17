"""
Webscraping of the data from the website energy-charts.info
Both the old and the new API are supported.
The old API is valid until 2019-12-31 and the new API is valid from 2020-01-01
"""

import requests
import json
import pandas as pd
from time import sleep
import numpy as np

one_week = 1421016300000 - 1420412400000
quarter_hour = 900000
time_step = one_week + quarter_hour


def get_energy_chart_data(year, week, frequency="15min"):
    """
    Get raw 15min data from energy-charts.info for a given year and month.
    Unpack the data and preprocess it to a pandas DataFrame.
    Depending on the year and month, the data is either in the old or the new API format, both are supported.
    Switch from old to new API is in 2020, CW 40.
    This data includes intra-day data.
    Link to original dashboard:
    https://energy-charts.info/charts/price_spot_market/chart.htm?l=de&c=DE&minuteInterval=15min

    Args:
        year: year to get data for
        week: calendar week to get data for (0-52)

    Returns:
        pandas DataFrame with data
    """
    if frequency == "hour":
        url = f"https://energy-charts.info/charts/price_spot_market/data/de/week_{year}_{week}.json"
    elif (year == 2020 and int(week) > 40) or (year > 2020):
        url = f"https://energy-charts.info/charts/price_spot_market/data/de/week_15min_{year}_{week}.json"
    else:
        url = f"https://energy-charts.info/charts/price_spot_market/raw_data/de/week_15min_{year}_{week}.json"

    response = requests.get(url, timeout=10)
    if response.ok:
        json_data = json.loads(response.content)
    else:
        print(f"{year}_{week}_Response not ok: {response.status_code}")
        return None
    d = {}
    print(year, week)
    # If clauses for the different data formats depending on the year and week
    # Different years have different data formats
    if ((year == 2020 and int(week) > 40) or (year > 2020)) or frequency == "hour":
        for i in range(len(json_data)):
            try:
                d[json_data[i]["name"]["en"]] = json_data[i]["data"]
            except TypeError:
                if json_data[i]["name"][0]["en"] == "Load":
                    d[json_data[i]["name"][0]["en"]] = json_data[i]["data"]
                else:
                    d[json_data[i]["name"][0]["en"]] = json_data[i]["data"]
        d["date"] = json_data[0]["xAxisValues"]

    else:
        for i in range(len(json_data)):
            try:
                sublist = list(zip(*json_data[i]["values"]))
                d[json_data[i]["key"][0]["en"]] = sublist[1]
            except TypeError:
                d[json_data[i]["key"]["en"]] = json_data[i]["values"][1]
        d["date"] = sublist[0]

    df = pd.DataFrame(d)
    df["date"] = pd.to_datetime(df["date"], unit="ms")
    return df
    # return webscraping



def get_yearly_energy_chart_data(year, frequency="15min"):
    """
    Get data from energy-charts.info for a given year and preprocess it to a pandas DataFrame
    Old API (Valid until 2019-12-31)
    Args:
        year: year to get data for

    Returns:
        list of pandas DataFrames with data for each api call
    """
    data = []
    for month_int in range(1, 53):
        if month_int < 10:
            month = f"0{month_int}"
        else:
            month = f"{month_int}"
        monthly_data = get_energy_chart_data(year=year, week=month, frequency = frequency)
        if isinstance(monthly_data, pd.DataFrame):
            data.append(monthly_data)
        sleep(np.random.uniform(0.0, 1.0))
    return data

root_dir = "../../data/energy_chart_data/"
start_year = 2020
end_year = 2023
frequency = "hour"


if __name__ == "__main__":
    """
    Get data from energy-charts.info for a given year and preprocess it to a pandas DataFrame
    """
    data = []
    for year in range(start_year, end_year):
        print(year)
        data += get_yearly_energy_chart_data(year, frequency)

    complete_data = pd.concat(data)
    complete_data.to_csv(
        f"{root_dir}energy_chart_data_test_{start_year}_{end_year}_{frequency}.csv"
    )

import requests
import json
import pandas as pd
from time import sleep

one_week = 1421016300000 - 1420412400000
quarter_hour = 900000
time_step = one_week + quarter_hour



def get_energy_chart_data(year, month):


    url = f"https://energy-charts.info/charts/price_spot_market/raw_data/de/month_{year}_{month}.json"
    url = f"https://energy-charts.info/charts/price_spot_market/raw_data/de/week_15min_{year}_{month}.json"
    response = requests.get(url)
    if response.ok:
        json_data = json.loads(response.content)
    else:
        print(f"{year}_{month}_Response not ok: {response.status_code}")
        return None
    d = {}
    for i in range(len(json_data)):
        try:
            sublist = list(zip(*json_data[i]["values"]))
            d[json_data[i]["key"][0]["en"]] = sublist[1]
            #d[json_data[i]["key"][0]["en"] + "_date"] = sublist[0]

        except:
            d[json_data[i]["key"]["en"]] = json_data[i]["values"][1]
            #d[json_data[i]["key"]["en"] + "_date"] = json_data[i]["values"][0]

    d["date"] = sublist[0]
        #d["date"] = json_data[0]["xAxisValues"]
    df = pd.DataFrame(d)

    df["date"] = pd.to_datetime(df["date"], unit='ms')
    return df
            #return webscraping

def get_energy_chart_data_new_api(year, month, frequency = "15min"):

    if frequency == "15min":
        url = f"https://energy-charts.info/charts/price_spot_market/data/de/week_15min_{year}_{month}.json"
    elif frequency ==  "hour":
        url = f"https://energy-charts.info/charts/price_spot_market/data/de/week_{year}_{month}.json"
    response = requests.get(url)
    if response.ok:
        json_data = json.loads(response.content)
    else:
        print(f"{year}_{month}_Response not ok: {response.status_code}")
        return None
    d = {}
    for i in range(len(json_data)):
        try:
            d[json_data[i]["name"]["en"]] = json_data[i]["webscraping"]

        except:
            d[json_data[i]["name"][0]["en"]] = json_data[i]["webscraping"]
            #d[json_data[i]["key"]["en"] + "_date"] = json_data[i]["values"][0]

    d["date"] = json_data[0]["xAxisValues"]
        #d["date"] = json_data[0]["xAxisValues"]
    df = pd.DataFrame(d)

    df["date"] = pd.to_datetime(df["date"], unit='ms')
    return df
            #return webscraping

def get_yearly_energy_chart_data(year ):
    data = []
    for month_int in range(1, 13):
        if month_int < 10:
            month = f"0{month_int}"
        else:
            month = f"{month_int}"
        monthly_data = get_energy_chart_data(year=year, month=month)
        if isinstance(monthly_data, pd.DataFrame):
            data.append(monthly_data)
        sleep(0.5)
    return data


def get_yearly_energy_chart_data_new_api(year, frequency = "15min" ):
    data = []
    for month_int in range(1, 13):
        if month_int < 10:
            month = f"0{month_int}"
        else:
            month = f"{month_int}"
        monthly_data = get_energy_chart_data_new_api(year=year, month=month, frequency=frequency)
        if isinstance(monthly_data, pd.DataFrame):
            data.append(monthly_data)
        sleep(0.5)
    return data

start_year = 2015
end_year = 2023
api_version = "new"
frequency = "hour"

if __name__ == '__main__':
    data = []
    for year in range(start_year,end_year):
        print(year)
        if api_version == "new":
            data += get_yearly_energy_chart_data_new_api(year, frequency=frequency)
        else:
            data += get_yearly_energy_chart_data(year, frequency = frequency)

    complete_data = pd.concat(data)
    complete_data.to_csv(f"../../webscraping/energy_chart_data_{start_year}_{end_year}_{api_version}-api_{frequency}.csv")
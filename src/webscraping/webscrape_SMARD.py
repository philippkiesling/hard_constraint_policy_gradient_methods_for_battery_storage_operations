import requests
import json
import pandas as pd
from time import sleep

one_week = 1421016300000 - 1420412400000
quarter_hour = 900000
time_step = one_week + quarter_hour



def get_smard_data(filter = "1223", region = "DE", resolution = "quarterhour" , timestamp= 1420412400000):
    filterCopy = filter
    regionCopy = region
    url = f"https://www.smard.de/app/table_data/{filter}/{region}/{filterCopy}_{regionCopy}_{resolution}_{timestamp}.json"
    #url = f"https://www.smard.de/app/table_data/1225/DE/1225_DE_quarterhour_1419807600000.json"
    data = requests.get(url)
    print(data)
    return data
    #return webscraping

def save_smard_data(filter = "1223", region = "DE", resolution = "quarterhour" , start_point= 1420412400000):
    data_list = []
    # 1420412400000 # 04.01.2015
    for i in range(0,100000):
        response = get_smard_data(filter = filter, timestamp = start_point + time_step * i, region = region, resolution=resolution )

        if response.ok:
            json_data = json.loads(response.content)
            json_data = json_data["series"][0]["values"]
            df = pd.DataFrame(json_data)
            df["timestamp_correct"] = pd.to_datetime(df["timestamp"], unit='ms')
            sleep(1)
            data_list.append(df)
        else:
            print(f"Break at index {i}")
            break

    if len(data_list)> 0:
        df_save = pd.concat(data_list)
        min_timestamp = df["timestamp"].min()
        df_save.to_csv(f"../../data/table_data_{filter}_{region}_{resolution}_{start_point}-{min_timestamp}.csv")


if __name__ == '__main__':
    filter = "4068" #"410"
    region = "DE"
    resolution = "quarterhour"
    start_point = 1419807600000

    save_smard_data(filter = "", region=region, resolution=resolution, start_point=start_point)

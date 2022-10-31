import pandas as pd
import os
def concat_grosshandel_SMARD():
    rootdir = "../../webscraping/prices-grosshandel"

    data = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            print(file)
            path = f"{rootdir}/{file}"
            print(path)
            data.append(pd.read_csv(path, sep = ";"))

    df = pd.concat(data)
    df.to_csv("../../webscraping/grosshandelpreise_201501010000_202210282359.csv")

def concat_energy_charts():
    rootdir = "../../webscraping/energy_chart_data"

    data = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            print(file)
            path = f"{rootdir}/{file}"
            print(path)
            data.append(pd.read_csv(path, sep=",", index_col=False).iloc[:, 1:])

    df = pd.concat(data)
    df.to_csv("../../webscraping/energy_chart_data_2010_2023.csv")
    return df
if __name__ == '__main__':
    df = concat_energy_charts()
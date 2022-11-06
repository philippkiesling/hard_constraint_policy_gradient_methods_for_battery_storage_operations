import numpy as np
import pandas as pd
import sys
def get_15min_data():
    """
    Get 15min data from csv. This includes Intraday Continuous 15 minutes ID1-Price.
    Data is concatenated from 2015 to 2021 (old api) and 2021 to 2023 (new api).
    Returns:
        df_15min: Dataframe with 15min data including Intraday Continuous 15 minutes ID1-Price
    """
    df_new_api = pd.read_csv("../data/energy_chart_data/energy_chart_data_2021_2023_new_api_15min.csv",parse_dates=["date"])
    df_new_api = df_new_api.iloc[:, 1:]
    df_old_api = pd.read_csv("../data/energy_chart_data/energy_chart_data_2015_2021_old_api_15min.csv", parse_dates=["date"])
    df_old_api = df_old_api.iloc[:, 1:]
    df_15min = pd.concat([df_old_api, df_new_api])
    return df_15min

def get_hourly_data():
    """
    Get hourly data from csv. This includes Day-Ahead Auction Prices.
    Day-Ahead Auction Prices are Merged with Day-Ahead Auction Prices DE-LU and DE-AT-LU.
    Returns:
        df_hourly: Dataframe with hourly data including Day-Ahead Auction Prices
    """
    df_hour = pd.read_csv("../data/energy_chart_data/energy_chart_data_2015_2023_new-api_hour.csv",
                          parse_dates=["date"])
    df_hour = df_hour.iloc[:, 1:]
    df_hour.loc[:, "Day-Ahead Auction"] = df_hour.loc[:, "Day Ahead Auction (DE-LU)"]
    df_hour.loc[df_hour["Day-Ahead Auction"].isna(), "Day-Ahead Auction"] = df_hour.loc[
        df_hour["Day-Ahead Auction"].isna(), "Day Ahead Auction (DE-AT-LU)"]
    df_hour = df_hour.sort_values(by="date")
    df_hour = df_hour.drop(["Day Ahead Auction (DE-LU)", "Day Ahead Auction (DE-AT-LU)"], axis=1)
    return df_hour

def get_data():
    """
    Get 15min and hourly data from csv.
    Interpolates hourly data to 15min data.
    Returns:
        df: Dataframe with 15min data including Intraday Continuous 15 minutes ID1-Price and Day-Ahead Auction Prices
    """
    df_15min = get_15min_data()
    df_hour = get_hourly_data()
    df = df_15min.set_index("date").join(df_hour[["Day-Ahead Auction", "date"]].set_index("date"), how="left")
    df = df[~df[["Day-Ahead Auction", "Intraday Continuous 15 minutes ID1-Price"]].isna()]
    df = df.rename(columns={"Intraday Continuous 15 minutes ID1-Price": "intraday_15min",
                            "Day-Ahead Auction": "day_ahead"})
    df["day_ahead"] = df["day_ahead"].resample("15min").interpolate()
    df = df.sort_index()
    df = df[df["day_ahead"].isna() == False]
    df = df[df["intraday_15min"].isna() == False]
    return df

class Data_Loader_np():
    def __init__(self):
        """
        Initialize the Data Loader
        """
        self.data = get_data()
        self.current_index = 0
        self.max_index = len(self.data) - 1

        # Initialize the day ahead price as NaN
        self.day_ahead_price = np.zeros((24+12)*4)
        self.day_ahead_price[:] = np.NAN
        # Initialize the intraday price as NaN
        self.intraday_price = np.zeros((24 + 12) * 4)
        self.intraday_price[:] = np.NAN

    def reset(self):
        """
        Reset the current index, day ahead price and intraday price
        """
        self.current_index = 0
        self.day_ahead_price = np.zeros((24+12)*4)
        self.day_ahead_price[:] = np.NAN
        self.intraday_price = np.zeros((24 + 12) * 4)
        self.intraday_price[:] = np.NAN
        return self#

    def get_next_day_ahead_price(self):
        """
        Get the next day ahead price
        Returns:
            day ahead price
        """
        if self.current_index+(24+12)*4 < self.max_index:
            # Rotate the day ahead price one step ahead and delete the last entry
            self.day_ahead_price = np.roll(self.day_ahead_price, -1)
            self.day_ahead_price[-1] = np.NAN
            if self.data.iloc[self.current_index].name.hour == 12 and self.data.iloc[self.current_index].name.minute == 0:
                self.day_ahead_price[(12)*4:] = self.data.iloc[self.current_index + 12*4: self.current_index+ (24+12)*4]["day_ahead"]
            return self.day_ahead_price, False
        else:
            return None, True

    def get_next_intraday_price(self):
        """
        Get the next intraday price
        Returns:
            intraday price
        """
        # Rotate the day ahead price one step ahead and delete the last entry
        self.intraday_price = np.roll(self.intraday_price, 1)
        self.intraday_price[0] = self.data.iloc[self.current_index]["intraday_15min"]
        return self.intraday_price, False

    def get_next_day_ahead_and_intraday_price(self):
        """
        Get the next day ahead and intraday price
        Returns:
            day ahead price and intraday price
        """
        self.current_index += 1
        day_ahead_price,done_day_ahead = self.get_next_day_ahead_price()
        intraday_price, done_intraday = self.get_next_intraday_price()

        return day_ahead_price, intraday_price, done_intraday + done_day_ahead


if __name__ == "__main__":
    data_loader = Data_Loader_np()
    while True:
        day_ahead_price, intraday_price, done = data_loader.get_next_day_ahead_and_intraday_price()
        if done:
            break
        #print(day_ahead_price, intraday_price)
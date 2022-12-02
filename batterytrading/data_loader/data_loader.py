import numpy as np
import pandas as pd
import math

def get_15min_data(root_path=""):
    """
    Get 15min data from csv. This includes Intraday Continuous 15 minutes ID1-Price.
    Data is concatenated from 2015 to 2021 (old api) and 2021 to 2023 (new api).
    Returns:
        df_15min: Dataframe with 15min data including Intraday Continuous 15 minutes ID1-Price
    """


    df_15min = pd.read_csv(
        f"{root_path}data/energy_chart_data/energy_chart_data_test_2015_2023_new-api_15min.csv",
        parse_dates=["date"],
    )
    df_15min = df_15min.iloc[:, 1:]


    return df_15min


def get_hourly_data(root_path=""):
    """
    Get hourly data from csv. This includes Day-Ahead Auction Prices.
    Day-Ahead Auction Prices are Merged with Day-Ahead Auction Prices DE-LU and DE-AT-LU.
    Returns:
        df_hourly: Dataframe with hourly data including Day-Ahead Auction Prices
    """
    df_hour = pd.read_csv(
        f"{root_path}data/energy_chart_data/energy_chart_data_2015_2023_new-api_hour.csv",
        parse_dates=["date"],
    )

    df_hour = pd.read_csv(
        f"{root_path}data/energy_chart_data/energy_chart_data_test_2015_2023_new-api_hour.csv",
        parse_dates=["date"],
    )
    df_hour = df_hour.iloc[:, 1:]
    df_hour.loc[:, "Day-Ahead Auction"] = df_hour.loc[:, "Day Ahead Auction (DE-LU)"]
    df_hour.loc[df_hour["Day-Ahead Auction"].isna(), "Day-Ahead Auction"] = df_hour.loc[
        df_hour["Day-Ahead Auction"].isna(), "Day Ahead Auction (DE-AT-LU)"
    ]
    df_hour.loc[df_hour["Day-Ahead Auction"].isna(), "Day-Ahead Auction"] = df_hour.loc[
        df_hour["Day-Ahead Auction"].isna(), "Day Ahead Auction"
    ]
    df_hour = df_hour.sort_values(by="date")
    df_hour = df_hour.drop(["Day Ahead Auction (DE-LU)", "Day Ahead Auction (DE-AT-LU)"], axis=1)
    return df_hour


def get_data(root_path="", time_interval = "15min")  -> pd.DataFrame:
    """
    Get 15min and hourly data from csv.
    Interpolates hourly data to 15min data.
    Returns:
        df: Dataframe with 15min data including Intraday Continuous 15 minutes ID1-Price and Day-Ahead Auction Prices
    """
    df_15min = get_15min_data(root_path)
    df_hour = get_hourly_data(root_path)
    df_hour = df_hour.drop_duplicates("date")
    df_hour = df_hour.set_index("date")
    #df_hour["Day-Ahead Auction"] = df_hour["Day-Ahead Auction"].resample("15min").interpolate()
    df = df_15min.set_index("date").join(df_hour[["Day-Ahead Auction"]], how="left")

    df.loc[df["Intraday Continuous 15 minutes ID3-Price"].isna(), "Intraday Continuous 15 minutes ID3-Price"] = df.loc[df["Intraday Continuous 15 minutes ID3-Price"].isna(), "Intraday Continuous 15 minutes Average Price"]
    df.loc[df["Intraday Continuous 15 minutes ID3-Price"].isna(), "Intraday Continuous 15 minutes ID3-Price"] = df.loc[df["Intraday Continuous 15 minutes ID3-Price"].isna(), "Intraday Continuous 15 minutes ID1-Price"]

    #df["Intraday Continuous 15 minutes ID3-Price"] = df["Intraday Continuous 15 minutes ID3-Price"].interpolate(method='linear')
    df = df[~df[["Day-Ahead Auction", "Intraday Continuous 15 minutes ID3-Price"]].isna()]
    df = df.rename(
        columns={
            "Intraday Continuous 15 minutes ID3-Price": "intraday_15min",
            "Day-Ahead Auction": "day_ahead",
        }
    )
    df = df.sort_index()
    #df = df[~ df["day_ahead"].isna()]
    min_index = df[~df["day_ahead"].isna()].index.min()
    max_index = df[~df["day_ahead"].isna()].index.max()
    df = df.loc[min_index:max_index]
    df["day_ahead"] = df["day_ahead"].ffill()

    min_index = df[~df["intraday_15min"].isna()].index.min()
    max_index = df[~df["intraday_15min"].isna()].index.max()
    df = df.loc[min_index:max_index]
    df["intraday_15min"] = df["intraday_15min"].interpolate("linear")
    if time_interval == "H":
        df = df.resample("H").mean()
        df["intraday_15min"] = df["intraday_15min"].interpolate("linear")
    return df


class Data_Loader_np:
    def __init__(self, price_time_horizon=1.5, data=None, root_path="", time_interval="15min", n_past_timesteps = 1, time_features = True):
        """
        Initialize the Data Loader
        The returned intraday prices are historic and can therefore vary.
        The day ahead prices are the official forecasts of the day ahead auction.
        As in the real world, the next days prices are published at 12:00 of the previous day.
        Args:
            price_time_horizon: Number of days to return intraday prices
        """
        self.data = get_data(root_path=root_path, time_interval=time_interval)
        self.current_index = 0
        self.max_index = len(self.data) - 1
        self.time_interval = time_interval
        # Initialize the day ahead price as NaN
        self.day_ahead_price = np.zeros((24 + 12) * 4)
        self.day_ahead_price[:] = np.NAN
        # Initialize the intraday price as NaN
        intraday_sequence_length = int(24 * price_time_horizon * 4)
        self.intraday_price = np.zeros(intraday_sequence_length)
        self.n_past_timesteps = n_past_timesteps
        self.time_features = time_features
        # self.intraday_price[:] = np.NAN

    def reset(self):
        """
        Reset the current index, day ahead price and intraday price
        """
        self.current_index = 0
        self.day_ahead_price = np.zeros((24 + 12) * 4)
        self.day_ahead_price[:] = np.NAN
        self.intraday_price = np.zeros((24 + 12) * 4)
        # self.intraday_price[:] = np.NAN
        return self  #

    def get_next_day_ahead_price(self):
        """
        Get the next day ahead price
        Returns:
            day ahead price
        """
        if self.current_index + (24 + 12) * 4 < self.max_index:
            # Rotate the day ahead price one step ahead and delete the last entry
            self.day_ahead_price = np.roll(self.day_ahead_price, -1)
            self.day_ahead_price[-1] = np.NAN
            if self.data.iloc[self.current_index].name.hour == 12 and self.data.iloc[self.current_index].name.minute == 0:
                self.day_ahead_price[(12) * 4 :] = self.data.iloc[self.current_index + 12 * 4 : self.current_index + (24 + 12) * 4]["day_ahead"]
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
    def _get_time_features(self):
        """
        Get the current time of day
        Returns:
            time of day
        """
        try:
            np.sin((1/4))
            current_tod = self.data.iloc[self.current_index].name.hour + self.data.iloc[self.current_index].name.minute / 60
            current_sin = np.sin((current_tod)/4)
            current_cos = np.cos((current_tod)/4)
            current_sin2 = np.sin((current_tod)/12)
            current_cos2 = np.cos((current_tod)/12)
            current_sin3 = np.sin((current_tod)/24)
            current_cos3 = np.cos((current_tod)/24)
            #return np.array([current_sin, current_cos, current_sin2, current_cos2, current_sin3, current_cos3]), False
            return current_sin / 60, False
        except(IndexError):
            return None, True

    def get_next_day_ahead_and_intraday_price(self):
        """
        Get the next day ahead and intraday price
        Returns:
            day ahead price and intraday price
        """
        self.current_index += 1
        day_ahead_price, done_day_ahead = self.get_next_day_ahead_price()
        intraday_price, done_intraday = self.get_next_intraday_price()
        time_features, done_time_features = self._get_time_features()
        #features = np.hstack([intraday_price[0], time_features])
        if self.time_features:
            features = np.hstack([intraday_price[0:self.n_past_timesteps], time_features ])
        else:
            features = np.hstack([intraday_price[0:self.n_past_timesteps]])
        price = intraday_price[0]
        done = done_day_ahead or done_intraday or done_time_features

        return features, price, done


if __name__ == "__main__":
    data_loader = Data_Loader_np(time_interval="H")
    while True:
        (
            day_ahead_price,
            intraday_price,
            time_features,
            done,
        ) = data_loader.get_next_day_ahead_and_intraday_price()
        if done:
            break
        # print(day_ahead_price, intraday_price)

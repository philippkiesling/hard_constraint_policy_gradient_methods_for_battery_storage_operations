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

def get_gaussian_noise(mean, std, size):
    return np.random.normal(mean, std, size)
def get_solar_data(root_path=""):
    """
    Get solar data from csv.
    Returns:
        df_solar: Dataframe with solar data
    """
    df_solar = pd.read_csv(
        f"{root_path}data/solar_data.csv",
        parse_dates=["date"],
    )
    df_solar = df_solar.iloc[:, 1:]
    df_solar = df_solar.set_index("date")
    # For each year between 2015 duplicate the data and add one year to the date
    for year in range(2016, 2024):
        df_solar_year = df_solar.loc[df_solar.index.year == 2015]
        df_solar_year.index = df_solar_year.index + pd.DateOffset(years=year-2015)
        df_solar = pd.concat([df_solar,df_solar_year])
    df_solar = df_solar.sort_index()
    # Resample to 15 min with linear interpolation
    # Multiply Gaussian noise with std 0.1
    return df_solar
def get_data(root_path="", time_interval = "15min", gaussian_noise = False, noise_std = -1)  -> pd.DataFrame:
    """
    Get 15min and hourly data from csv.
    Interpolates hourly data to 15min data.
    Returns:
        df: Dataframe with 15min data including Intraday Continuous 15 minutes ID1-Price and Day-Ahead Auction Prices
    """
    df_15min = get_15min_data(root_path)
    df_hour = get_hourly_data(root_path)
    df_solar = get_solar_data(root_path)
    df_hour = df_hour.drop_duplicates("date")
    df_hour = df_hour.set_index("date")
    #df_hour["Day-Ahead Auction"] = df_hour["Day-Ahead Auction"].resample("15min").interpolate()
    df = df_15min.set_index("date").join(df_hour[["Day-Ahead Auction"]], how="left")
    df = df.join(df_solar[["electricity"]], how="left")
    df["solar"] = df["electricity"].interpolate(method='linear')
    df["solar"] = df["solar"] #* get_gaussian_noise(1, 0.05, df["solar"].shape[0])
    df["solar"] = df["solar"].fillna(0.0)
    df.loc[df["Intraday Continuous 15 minutes ID3-Price"].isna(), "Intraday Continuous 15 minutes ID3-Price"] = df.loc[df["Intraday Continuous 15 minutes ID3-Price"].isna(), "Intraday Continuous 15 minutes Average Price"]
    df.loc[df["Intraday Continuous 15 minutes ID3-Price"].isna(), "Intraday Continuous 15 minutes ID3-Price"] = df.loc[df["Intraday Continuous 15 minutes ID3-Price"].isna(), "Intraday Continuous 15 minutes ID1-Price"]

    #df["Intraday Continuous 15 minutes ID3-Price"] = df["Intraday Continuous 15 minutes ID3-Price"].interpolate(method='linear')
    #df = df[~df[["Day-Ahead Auction", "Intraday Continuous 15 minutes ID3-Price"]].isna()]
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
    # Add difference
    df["intraday_15min_diff"] = df["intraday_15min"].diff()
    df["intraday_15min_diff10"] = df["intraday_15min"].diff(10)

    df = df[~df["intraday_15min_diff10"].isna()]
    df = df[~df["intraday_15min_diff"].isna()]

    # merge solar data on index
    if time_interval == "H":
        df = df.resample("H").mean()
        df["intraday_15min"] = df["intraday_15min"].interpolate("linear")
    # Add gaussian noise, if specified
    if gaussian_noise:
        df["intraday_15min"] = df["intraday_15min"] + get_gaussian_noise(0, noise_std, df.shape[0])
        df["day_ahead"] = df["day_ahead"] + get_gaussian_noise(0, noise_std, df.shape[0])
    return df

class Data_Loader_np:
    def __init__(self, price_time_horizon=1.5,
                 data=None,
                 root_path="",
                 time_interval="15min",
                 n_past_timesteps = 1,
                 start_index = 0,
                 time_features=True,
                 gaussian_noise = False,
                 noise_std = -1):
        """
        Initialize the Data Loader
        The returned intraday prices are historic and can therefore vary.
        The day ahead prices are the official forecasts of the day ahead auction.
        As in the real world, the next days prices are published at 12:00 of the previous day.
        Args:
            price_time_horizon: Number of days to return intraday prices
            data: Dataframe with 15min data including Intraday Continuous 15 minutes ID1-Price and Day-Ahead Auction Prices
            root_path: Path to data folder
            time_interval: Time interval of the data. Either 15min or H
            n_past_timesteps: Number of past timesteps to return (intraday prices)
            time_features: If True, time features are returned
            start_index: Index of the first timestep to return
        """
        self.data = get_data(root_path=root_path, time_interval=time_interval,  gaussian_noise = gaussian_noise, noise_std = noise_std)
        self.current_index = start_index
        self.start_index = start_index
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
        self.intraday_price_scaled = np.zeros((24 + 12) * 4)

    def reset(self):
        """
        Reset the current index, day ahead price and intraday price
        """
        self.current_index = self.start_index
        self.day_ahead_price = np.zeros((24 + 12) * 4)
        self.day_ahead_price[:] = np.NAN
        self.intraday_price = np.zeros((24 + 12) * 4)
        self.intraday_price_scaled = np.zeros((24 + 12) * 4)
        # self.intraday_price[:] = np.NAN
        return self  #

    def get_next_day_ahead_price(self, set_current_index=False):
        """
        Get the next day ahead price
        Returns:
            day ahead price
        """
        if set_current_index:
            self.current_index += 1
        self.day_ahead_price = np.roll(self.day_ahead_price, -1)
        self.day_ahead_price[-1] = np.NAN

        if self.current_index + (24 + 12) * 4 < self.max_index:
            # Rotate the day ahead price one step ahead and delete the last entry

            if self.data.iloc[self.current_index].name.hour == 12 and self.data.iloc[self.current_index].name.minute == 0:
                self.day_ahead_price[(12) * 4 :] = self.data.iloc[self.current_index + 12 * 4 : self.current_index + (24 + 12) * 4]["day_ahead"]

            return self.day_ahead_price, False
        else:
            return None, True

    def get_current_timestamp(self):
        return self.data.iloc[self.current_index].name

    def get_next_intraday_price(self, set_current_index=False, n_past_timestep = 1):
        """
        Get the next intraday price
        Returns:
            intraday price
        """
        # Rotate the day ahead price one step ahead and delete the last entry
        self.intraday_price = np.roll(self.intraday_price, 1)
        self.intraday_price[0] = self.data.iloc[self.current_index]["intraday_15min"]
        return self.intraday_price, False
    def get_next_intraday_price_scaled(self, set_current_index=False, n_past_timestep = 1):
        """
        Get the next intraday price
        Returns:
            intraday price
        """
        # Rotate the day ahead price one step ahead and delete the last entry
        self.intraday_price_scaled = np.roll(self.intraday_price_scaled, 1)
        self.intraday_price_scaled[0] = self.data.iloc[self.current_index]["intraday_15min_diff10"]
        return self.intraday_price_scaled, False
    def _get_time_features(self):
        """
        Get the current time of day as sine and cosine waves with different frequencies (daily, weekly, monthly, yearly)
        Returns:
            time features as numpy array [daily_sine, daily_cosine, weekly_sine, weekly_cosine, monthly_sine, monthly_cosine, yearly_sine, yearly_cosine]

                    """
        try:
            time = self.current_index
            daily_frequency = 2 * np.pi / (24 * 4)  # 4 times per hour = 4 * 24 = 96 times per day
            weekly_frequency = 2 * np.pi / (168 * 4)  # 168 hours per week * 4 times per hour
            monthly_frequency = 2 * np.pi / (720 * 4)  # 720 hours per month * 4 times per hour
            yearly_frequency = 2 * np.pi / (8760 * 4)  # 8760 hours per year * 4 times per hour

            # Compute the sine and cosine waves for each frequency
            daily_sine = np.sin(daily_frequency * time)
            daily_cosine = np.cos(daily_frequency * time)

            weekly_sine = np.sin(weekly_frequency * time)
            weekly_cosine = np.cos(weekly_frequency * time)

            monthly_sine = np.sin(monthly_frequency * time)
            monthly_cosine = np.cos(monthly_frequency * time)

            yearly_sine = np.sin(yearly_frequency * time)
            yearly_cosine = np.cos(yearly_frequency * time)

            # Return the time features
            return np.array([daily_sine,
                             daily_cosine,
                             weekly_sine,
                             weekly_cosine,
                             monthly_sine,
                             monthly_cosine,
                             yearly_sine,
                             yearly_cosine]), False
            #return np.array([current_sin, current_cos, current_sin2, current_cos2, current_sin3, current_cos3]), False
        except(IndexError):
            return None, True

    def get_next_features(self):
        """
        Get the next day ahead and intraday price
        Returns:
            day ahead price and intraday price
        """
        #self.current_index += 1

        day_ahead_price, done_day_ahead = self.get_next_day_ahead_price(set_current_index=False)
        intraday_price, done_intraday = self.get_next_intraday_price(set_current_index=False)
        intraday_price_scaled, done_intraday_scaled = self.get_next_intraday_price_scaled(set_current_index=False)
        time_features, done_time_features = self._get_time_features()
        #features = np.hstack([intraday_price[0], time_features])
        day_ahead_price = day_ahead_price[[idx * 4 for idx in range(0,12)]]
        #day_ahead_price = day_ahead_price[46:47]
        day_ahead_price[np.isnan(day_ahead_price)] = 0.0
        if self.time_features:
            features = np.hstack([intraday_price[0:self.n_past_timesteps], intraday_price[0:self.n_past_timesteps], day_ahead_price, time_features ])
        else:
            features = np.hstack([intraday_price[0:self.n_past_timesteps], intraday_price[0:self.n_past_timesteps], day_ahead_price])
        price = intraday_price[0]
        done = done_day_ahead or done_intraday or done_time_features

        self.current_index += 1
        return features, price, done

class Data_Loader_np_solar(Data_Loader_np):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
    def get_next_features(self):
        solar_production = self.data.iloc[self.current_index]["solar"]/6
        features, price, done = super().get_next_features()
        features = np.hstack([features, solar_production])
        self.solar_production = solar_production
        return features, price,  done


class RandomSampleDataLoader(Data_Loader_np):
    def __init__(self, price_time_horizon=1.5, data=None, root_path="", time_interval="15min", n_past_timesteps = 1, time_features = True):
        """
        Initialize the Data Loader
        The returned intraday prices are historic and can therefore vary.
        The day ahead prices are the official forecasts of the day ahead auction.
        As in the real world, the next days prices are published at 12:00 of the previous day.
        Args:
            price_time_horizon: Number of days to return intraday prices
            data: Dataframe with 15min data including Intraday Continuous 15 minutes ID1-Price and Day-Ahead Auction Prices
            root_path: Path to data folder
            time_interval: Time interval of the data. Either 15min or H
            n_past_timesteps: Number of past timesteps to return (intraday prices)
            time_features: If True, time features are returned

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

    def get_next_day_ahead_price(self, set_current_index=False):
        """
        Get the next day ahead price
        Returns:
            day ahead price
        """
        random_index = np.random.randint(0,len(self.data))
        if set_current_index:
            self.current_index += 1
        if self.current_index + (24 + 12) * 4 < self.max_index:
            # Rotate the day ahead price one step ahead and delete the last entry
            self.day_ahead_price = np.roll(self.day_ahead_price, -1)
            self.day_ahead_price[-1] = np.NAN
            if self.data.iloc[self.current_index].name.hour == 12 and self.data.iloc[self.current_index].name.minute == 0:
                self.day_ahead_price[12 * 4:] = self.data.iloc[random_index + 12 * 4: random_index + (24+12) * 4]["day_ahead"]
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
        random_index = np.random.randint(0, len(self.data))
        self.intraday_price = np.roll(self.intraday_price, 1)
        self.intraday_price[0] = self.data.iloc[random_index]["intraday_15min"]
        return self.intraday_price, False


if __name__ == "__main__":
    data_loader = Data_Loader_np(time_interval="15min", root_path="../", gaussian_noise=True, noise_std=1)
    while True:
        (
            features, price, done
        ) = data_loader.get_next_features()
        if done:
            break
        # print(day_ahead_price, intraday_price)
# Check if cuda is available

from batterytrading.environment import Environment
import numpy as np


class Day_Ahead_Trader:
    def __init__(self, max_charge=0.15, total_storage_capacity=1, initial_charge=0.0, max_SOC=1):
        self.max_charge = max_charge
        self.max_SOC = max_SOC
        self.min_SOC = 0
        self.total_storage_capacity = total_storage_capacity
        self.initial_charge = initial_charge
        self.SOC = np.array([initial_charge])
        self.TOTAL_EARNINGS = 0
        self.action_valid = []
        self.day_ahead_price = 0
        self.trajectory = np.zeros((24 + 12) * 4)
        self.n_charges_per_day = self._get_max_times()

    def step(self, state, reward):
        """
            Perform a step
            If the last element in day_ahead_price is reached, not Nan, we calculate a new trajectory
            If the last element is nan, rotate the trajectory and return the next element
        Args:
            state:

        Returns:

        """
        self.SOC = state["SOC"]
        self.day_ahead_price = state["day-ahead-price"]
        if not np.isnan(self.day_ahead_price[-1]):
            self.trajectory = self._calculate_trajectory(self.day_ahead_price)
        action = self.trajectory[0]
        self.trajectory = np.roll(self.trajectory, -1)
        self.trajectory[-1] = np.nan
        return action

    def _calculate_trajectory(self, day_ahead_price):
        """
            Calculate the trajectory for the day
            Set Traaectory to -1 at discharge_index and 1 at charge_index
            Set Trajectory to 0 at all other timesteps

        Args:
            day_ahead_price:

        Returns:

        """
        discharge_index, charge_index = self._get_charging_timesteps()
        trajectory = np.zeros((24 + 12) * 4)
        trajectory[charge_index] = 1
        trajectory[discharge_index] = -1
        return trajectory

    def _get_max_times(self):
        """
            Calculate the maximum charge-instructions per day.
            This is based on the assumption that the battery is only charged once a day.
        Returns:
            maximum number of times the battery can be charged and discharged in a day
        """
        max_times = np.ceil(self.max_SOC / self.max_charge)
        return int(max_times)

    def _get_charging_timesteps(self):
        """
            Find the n_charges lowest prices in the day ahead price array
            Find the n_chargest highest prices in the day ahead price array
        Returns:
            index of the n_charges lowest and highest prices
        """
        n_charges = int(self.n_charges_per_day * 5)
        n_charges_lowest_prices = np.argpartition(self.day_ahead_price[12 * 4 :], n_charges)[:n_charges] + 12 * 4
        n_charges_highest_prices = np.argpartition(self.day_ahead_price[12 * 4 :], -n_charges)[-n_charges:] + 12 * 4
        return n_charges_lowest_prices, n_charges_highest_prices


if __name__ == "__main__":
    """
    Test the day ahead trader
    """
    env = Environment()
    trader = Day_Ahead_Trader()
    env.reset()
    action = 0
    sum_of_rewards = 0
    while True:
        next_state, reward, done, info = env.step(action)

        if done:
            print(sum_of_rewards)
            break
        action = trader.step(next_state, reward)
        sum_of_rewards += reward
        print(sum_of_rewards)

from gym import core
from gym import spaces
import numpy as np
from src.data_loader import Data_Loader_np

class Environment(core.Env):
    def __init__(self, max_charge = 0.15, total_storage_capacity = 1, initial_charge = 0.0, max_SOC = 1):
        """
        Initialize the Environment
        Args:
            max_charge: Maximum charge/discharge rate
            total_storage_capacity: Total Storage Capacity of the Battery
            initial_charge: Initial SOC of the Battery
            max_SOC: Maximum SOC of the Battery
        """
        self.SOC = initial_charge
        self.initial_charge = initial_charge
        self.TOTAL_EARNINGS = 0
        self.max_charge = max_charge
        self.max_SOC = max_SOC
        self.min_SOC = 0

        self.TOTAL_STORAGE_CAPACITY = total_storage_capacity
        # Three actions: Hold, Charge, Discharge
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.state = NotImplementedError
        # Save which action is valid
        self.action_valid = []
        self.data_loader = Data_Loader_np()

    def reset(self):
        """
        Reset the Environment
        Returns:
            Environment (reseted)
        """
        self.SOC = self.initial_charge
        self.TOTAL_EARNINGS = 0
        self.action_valid = []
        self.data_loader.reset()
        #super().reset()
        return self

    def _get_next_state(self, action):
        """
        Get the next state of the Environment consisting of the SOC, day_ahead-action and the price
        Args:
            action: action to perform (hold, charge, discharge)
        Returns:
            next state, price, done
        """
        self.SOC = self.charge(action)
        day_ahead_price, current_price, done = self.data_loader.get_next_day_ahead_and_intraday_price()
        return {"SOC": self.SOC, "day-ahead-price":day_ahead_price}, current_price[0], done


    def step(self, action):
        """
        Perform a (15min) step in the Environment
        Args:
            action: action to perform (hold, charge, discharge)

        Returns:
            next state, reward, done, info
        """
        err_msg = f"{action!r} ({type(action)}) invalid"
        #assert self.action_space.contains(action), err_msg
        #assert self.state is not None, "Call reset before using step method."

        # Clip action into a valid space
        self.action_valid.append(np.abs(action) <= self.max_charge)

        action = np.clip(action, -self.max_charge, self.max_charge)
        # Calculate earnings
        next_state, price, done = self._get_next_state(action)


        rewards = self.calculate_earnings(action, price)
        return next_state, rewards, done, {}

    def charge(self, rel_amount):
        """
        (Un-) Charge the Battery
        Args:
            rel_amount: Percentage of Battery to Charge, Uncharges if negative

        Returns:
            Current SOC (State of Charge)
        """
        self.SOC += rel_amount * self.TOTAL_STORAGE_CAPACITY
        self.SOC = np.clip(self.SOC, self.min_SOC, self.max_SOC)
        return self.SOC

    def calculate_earnings(self, rel_amount, price):
        """
        Calculates Earnings in the last episode
        Args:
            rel_amount: relative amount of power sold
            price: price to sell

        Returns:
            Earnings/Reward
        """
        earnings = rel_amount * self.TOTAL_STORAGE_CAPACITY * price
        self.TOTAL_EARNINGS += earnings
        return earnings

if __name__ == '__main__':
    env = Environment()
    env.reset()
    while True:
        action = (np.random.random(1) - 0.5) * 0.3
        next_state, reward, done, info = env.step(action)
        print(next_state["SOC"], reward, done, info)
        if done:
            break

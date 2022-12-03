from gym import core
from gym import spaces
import numpy as np
from batterytrading.data_loader import Data_Loader_np


class Environment(core.Env):
    def __init__(self, max_charge=0.15,
                 total_storage_capacity=1,
                 initial_charge=0.0,
                 max_SOC=1,
                 price_time_horizon=1.5,
                 data_root_path="",
                 time_interval="15min",
                 wandb_run=None,
                 n_past_timesteps=1,
                 time_features=False
                 ):
        """
        Initialize the Environment
        Args:
            max_charge: Maximum charge/discharge rate
            total_storage_capacity: Total Storage Capacity of the Battery
            initial_charge: Initial SOC of the Battery
            max_SOC: Maximum SOC of the Battery
        """
        # Initialize the wandb run (if wandb is used)
        if not (wandb_run is None):
            self.wandb_log = True
            self.wandb_run = wandb_run
        else:
            self.wandb_log = False

        # Initialize Environment
        self.SOC = initial_charge
        self.initial_charge = initial_charge
        self.TOTAL_EARNINGS = 0
        if time_interval == "H":
            self.max_charge = max_charge * 4
        else:
            self.max_charge = max_charge
        self.max_SOC = max_SOC
        self.min_SOC = 0
        self.TOTAL_STORAGE_CAPACITY = total_storage_capacity
        # Three actions: Hold, Charge, Discharge
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.state = NotImplementedError
        # Save which action is valid
        self.action_valid = []
        self.time_step = time_interval
        self.price_time_horizon = price_time_horizon
        self.data_loader = Data_Loader_np(price_time_horizon=price_time_horizon,
                                          root_path=data_root_path,
                                          time_interval=time_interval,
                                          n_past_timesteps=n_past_timesteps,
                                          time_features=time_features)

        features, _, _ = self._get_next_state(np.array(0))
        self.observation_space = spaces.Box(low= -100, high = 1000, shape=features.shape, dtype=np.float32)
        self.reset()


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
        # super().reset()
        return self.step(np.array(0))[0]

    def _get_next_state(self, action):
        """
        Get the next state of the Environment consisting of the SOC, day_ahead-action and the price
        Args:
            action: action to perform (hold, charge, discharge)
        Returns:
            next state, price, done
        """

        self.SOC = self.charge(action)
        features, price, done = self.data_loader.get_next_day_ahead_and_intraday_price()
        features = np.hstack([self.SOC, features])
        return (
            features,
            price,
            done,
        )

    def step(self, action):
        """
        Perform a (15min) step in the Environment
        Args:
            action: action to perform (hold, charge, discharge)

        Returns:
            next state, reward, done, info
        """
        # err_msg = f"{action!r} ({type(action)}) invalid"
        # assert self.action_space.contains(action), err_msg
        # assert self.state is not None, "Call reset before using step method."

        # Clip action into a valid space
        self.action_valid.append(int(np.abs(action) <= self.max_charge))
        # Reduce action to max_charge (maximum charge/discharge rate per period)
        action = np.clip(action, -self.max_charge, self.max_charge)
        # Clip action into the valid space
        # #(must not be smaller than current SOC, must not be larger than the difference
        # between maximum SOC and current SOC)
        action = np.clip(action, -self.SOC, self.max_SOC - self.SOC)
        # Calculate next state
        next_state, price, done = self._get_next_state(action)
        # Calculate reward/earnings
        rewards = self.calculate_earnings(action, price)
        # Log to wandb
        if self.wandb_log:
            self.wandb_run.log({"reward": rewards,
                                "action": action,
                                "price": price,
                                "SOC": self.SOC,
                                "action_valid": int(self.action_valid[-1]),
                                "total_earnings": self.TOTAL_EARNINGS})
        #next_state = np.hstack([next_state["SOC"], next_state["historic_price"][0], next_state["time_features"]])
        return next_state,rewards, done, {}

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

        sold_amount = (-rel_amount) * self.TOTAL_STORAGE_CAPACITY
        earnings = + sold_amount * price
        self.TOTAL_EARNINGS += earnings
        return earnings


if __name__ == "__main__":
    env = Environment(data_root_path="../../")
    env.reset()
    while True:
        action = (np.random.random(1) - 0.5) * 0.3
        next_state, reward, done, info = env.step(action)
        print(next_state, reward, done, info)
        if done:
            break

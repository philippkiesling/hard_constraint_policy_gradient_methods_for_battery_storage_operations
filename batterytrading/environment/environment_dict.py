import torch
from gym import core
from gym import spaces
import numpy as np
from batterytrading.data_loader import Data_Loader_np, RandomSampleDataLoader
from gym.wrappers import NormalizeObservation
from gym.wrappers.normalize import RunningMeanStd

class NormalizeObservationPartially(NormalizeObservation):
    def __init__(self, env, epsilon=1e-8, normalize_obs=True):
        super(NormalizeObservationPartially, self).__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            #self.observation_space.shape = (self.observation_space.shape[0],)
            self.obs_rms = RunningMeanStd(shape=(self.observation_space.shape[0]- 2,) ) # Minus 2 because we don't want to normalize the last two elements of the observation (bounds used for projecting the action to a valid action space)
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        obsN = obs[:-2]
        if self.is_vector_env:
            obsN = self.normalize(obsN)
        else:
            obsN = self.normalize(np.array([obsN]))[0]
        obs[:-2] = obsN
        return obs, rews, dones, infos

    def reset(self):
        obs = self.env.reset()
        obsN = obs[:-2]
        self.obs_rms = RunningMeanStd(shape=obsN.shape)

        if self.is_vector_env:
            obsN = self.normalize(obsN)
        else:
            obsN = self.normalize(np.array([obsN]))[0]
        obs[:-2] = obsN
        return obs

class Environment(core.Env):
    def __init__(self, max_charge=0.15,
                 total_storage_capacity=1,
                 initial_charge=0.0,
                 max_SOC=1,
                 price_time_horizon=1.5,
                 data_root_path="..",
                 time_interval="15min",
                 wandb_run=None,
                 n_past_timesteps=1,
                 time_features=True,
                 day_ahead_environment=False,
                 prediction_output="action",
                 max_steps=None,
                 reward_shaping=None,
                 cumulative_coef=0.01,
                 eval_env = False,
                 n_steps = None,
                 gaussian_noise = False,
                 noise_std = 0.01,
                 ):
        """
        Initialize the Environment

        Args:
            max_charge: Maximum charge/discharge rate
            total_storage_capacity: Total Storage Capacity of the Battery
            initial_charge: Initial SOC of the Battery
            max_SOC: Maximum SOC of the Battery
            price_time_horizon:
            data_root_path: Path to the data
            time_interval: time per step (15min, 30min, H)
            wandb_run: Wandb run to log the data
            n_past_timesteps: number of past day_ahead_steps to use as input
            time_features: Use time features
            day_ahead_environment: -
            prediction_output: We can either predict the action or the future SOC, or percentage of potential charge/discharge
            max_steps: Maximum number of steps in the environment
            reward_shaping: Reward shaping schedule, None, "linear", "constant", "cosine", "exponential"
            cumulative_coef: Coefficient for the cumulative reward shaping
            eval_env: If the environment is used for evaluation (Evaluation Environment is always one step ahead of the training environment)
            n_steps: Number of steps in the environment, needed for evaluation Environments (No influence on training environments)
            gaussian_noise: Add gaussian noise to the day ahead prices and intra day prices
            noise_std: Standard deviation of the gaussian noise
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
        self.ROLLING_EARNINGS = []
        self.action_list = []
        if time_interval == "H":
            self.max_charge = max_charge * 4
        else:
            self.max_charge = max_charge
        self.max_SOC = np.array(max_SOC)
        self.min_SOC = np.array(0)
        self.TOTAL_STORAGE_CAPACITY = total_storage_capacity

        self.cumulative_coef = cumulative_coef
        # Three actions: Hold, Charge, Discharg
        if self.action_space is None:
            self.action_space = spaces.Box(low=-0.15, high=0.15, shape=(1,), dtype=np.float32)
        self.state = NotImplementedError
        # Save which action is valid
        self.action_valid = []
        self.prediction_output = prediction_output

        self.time_step = time_interval
        self.day_ahead_environment = day_ahead_environment
        self.price_time_horizon = price_time_horizon
        self.data_loader = Data_Loader_np(price_time_horizon=price_time_horizon,
                                          root_path=data_root_path,
                                          time_interval=time_interval,
                                          n_past_timesteps=n_past_timesteps,
                                          time_features=time_features,
                                          gaussian_noise=gaussian_noise,
                                          noise_std=noise_std,
                                          )


        features, _, _= self._get_next_state(np.array([0]))
        # Set Dictionary for the Observation Space
        #self.observation_space = spaces.Box(low= -100, high = 1000, shape=features.shape, dtype=np.float32)
        self.observation_space = spaces.Dict({"features": spaces.Box(low=-100, high = 1000, shape=features.shape, dtype=np.float32),
                                              "action_bounds": spaces.Box(low= -0.15, high = 0.15, shape=(2,), dtype=np.float32)})
        #self.observation_space = spaces.Box(low=-100, high = 1000, shape=features.shape, dtype=np.float32)

        if max_steps is None:
            self.max_steps = 99999999999999
        else:
            self.max_steps = max_steps
        self.setup_reward_schedule(reward_shaping)

        self.n_steps = 0
        self.eval_env = eval_env
        if eval_env:
            self.max_steps = n_steps # We want to evaluate the agent for n_steps (same stepwidth as during training -> This ensures that the agent evaluated for the same amount of time as trained)
            self.eval_episode = 0
            dummy_action = np.array([0])
            # Skip the first n_steps + 3 Steps (To ensure that the agent is evaluated one timehorizon ahead of training)
            for i in range(0, n_steps+3):
                self.step(self.action_space.sample())
        #self.reset()

    def setup_reward_schedule(self, reward_schedule):

        schedule_type = reward_schedule["type"]
        if schedule_type is None or schedule_type == "constant":
            get_soc_reward_weight = lambda iteration: reward_schedule["base_soc_reward"]
            get_action_taking_reward_weight = lambda iteration: reward_schedule["base_action_taking_reward"]
            get_consecutive_action_reward_weight = lambda iteration: reward_schedule["base_consecutive_action_reward"]
            get_income_reward_weight = lambda iteration: reward_schedule["base_income_reward"]
        elif schedule_type == "linear":
            get_soc_reward_weight = lambda iteration: reward_schedule["base_soc_reward"] * (self.max_steps - iteration) / self.max_steps
            get_action_taking_reward_weight = lambda iteration: reward_schedule["base_action_taking_reward"] * (self.max_steps/2.5 - iteration) / self.max_steps
            get_consecutive_action_reward_weight = lambda iteration: reward_schedule["base_consecutive_action_reward"] * (self.max_steps - iteration) / self.max_steps
            get_income_reward_weight = lambda iteration: reward_schedule["base_income_reward"] * iteration / self.max_steps
        elif schedule_type == "exponential":
            get_soc_reward_weight = lambda iteration: reward_schedule["base_soc_reward"] * np.exp(-iteration / reward_schedule["decay"])
            get_action_taking_reward_weight = lambda iteration: reward_schedule["base_action_taking_reward"] * np.exp(-iteration / reward_schedule["decay"])
            get_consecutive_action_reward_weight = lambda iteration: reward_schedule["base_consecutive_action_reward"] * np.exp(-iteration / reward_schedule["decay"])
            get_income_reward_weight = lambda iteration: reward_schedule["base_income_reward"] * np.exp(iteration / reward_schedule["decay"])
        elif schedule_type == "cosine":
            get_soc_reward_weight = lambda iteration: reward_schedule["base_soc_reward"] + reward_schedule["base_soc_reward"]  * np.cos(iteration / self.max_steps * np.pi)
            get_action_taking_reward_weight = lambda iteration: reward_schedule["base_action_taking_reward"] * np.cos(iteration / self.max_steps * np.pi)
            get_consecutive_action_reward_weight = lambda iteration: reward_schedule["base_consecutive_action_reward"] * np.cos(iteration / self.max_steps * np.pi)
            # The income reward increases linearly
            get_income_reward_weight = lambda iteration: 1+  reward_schedule["base_income_reward"] * iteration / self.max_steps
            #get_income_reward_weight = lambda iteration: reward_schedule["base_income_reward"]
        else:
            raise ValueError("Unknown reward schedule type {}".format(schedule_type))

        self.get_soc_reward_weight = get_soc_reward_weight
        self.get_action_taking_reward_weight = get_action_taking_reward_weight
        self.get_consecutive_action_reward_weight = get_consecutive_action_reward_weight
        self.get_income_reward_weight = get_income_reward_weight
        return get_soc_reward_weight, get_action_taking_reward_weight, get_consecutive_action_reward_weight, get_income_reward_weight

    def get_current_timestamp(self):
        return self.data_loader.get_current_timestamp()

    def reset(self):
        """
        Reset the Environment
        Returns:
            Environment (reseted)
        """
        if not self.eval_env:
            self.SOC = self.initial_charge
            self.TOTAL_EARNINGS = 0
            self.ROLLING_EARNINGS = []
            self.action_valid = []
            self.data_loader.reset()
            # super().reset()
            self.n_steps = 0
        else:
            self.eval_episode += 1
        obs, price, done, info = self.step(self.action_space.sample())
        return obs#, price, done, info

    def _get_next_state(self, action):
        """
        Get the next state of the Environment consisting of the SOC, day_ahead-action and the price
        Args:
            action: action to perform (hold, charge, discharge)
        Returns:
            next state, price, done
        """
        action = action[0]
        self.SOC = self.charge(action)
        if self.day_ahead_environment:
            features, done = self.data_loader.get_next_day_ahead_price()
            intra_day_price, done= self.data_loader.get_next_intraday_price()
            price = intra_day_price[0]
        else:
            features, price, done = self.data_loader.get_next_features()
            #day_ahead, done = self.data_loader.get_next_day_ahead_price(set_current_index=False)

        up_max_charge = np.min([(self.max_SOC - self.SOC), self.max_charge])
        down_max_charge = np.max([-(self.SOC - self.min_SOC), -self.max_charge])

        #features = np.hstack([self.SOC, features])
        features = np.hstack([self.SOC, features,  down_max_charge, up_max_charge])

        self.set_bounds(down_max_charge, up_max_charge)
        #features = np.hstack([self.SOC, features])
        #features = np.array([features, np.array((down_max_charge, up_max_charge))])
        return (
            features,
            price,
            done
        )

    def set_bounds(self, down_max_charge, up_max_charge):
        #self.bounds = torch.Tensor(np.array([down_max_charge])), torch.Tensor(np.array([up_max_charge]))
        bounds = np.hstack([down_max_charge, up_max_charge])
        self.bounds = bounds


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
        self.n_steps += 1

        if self.prediction_output == "nextSOC":
            action = action - self.SOC # action is now the difference of nextSOC to the current SOC
        elif self.prediction_output == "percentage_action":
            action = (action-0.5) * self.max_charge
        epsilon = 1e-1
        # Clip action into a valid space
        valid = (np.abs(action) <= self.max_charge + epsilon) & (0 <= action + self.SOC + epsilon) & (action + self.SOC <= self.TOTAL_STORAGE_CAPACITY + epsilon)
        if not valid:
            print("Action not valid: {}".format(action))
        self.action_valid.append(float(valid))
        # Reduce action to max_charge (maximum charge/discharge rate per period)
        action = np.clip(action, -self.max_charge, self.max_charge)
        # Clip action into the valid space
        # #(must not be smaller than current SOC, must not be larger than the difference
        # between maximum SOC and current SOC)
        action = np.clip(action, -self.SOC, self.max_SOC - self.SOC)
        # Calculate next state
        next_state, price, done = self._get_next_state(action)

        # Calculate non income rewards
        soc_reward, action_taking_reward, consecutive_action_reward = self.calculate_shaping_rewards(action, price)
        # Calculate reward weights
        soc_reward_weight, action_taking_reward_weight, consecutive_action_reward_weight = self.calculate_reward_weights()
        income_reward = self.calculate_earnings(action, price)
        # Cumulative reward
        # Calculate reward/earnings
        # Different reward functions can be used
        cumulative_reward = self.calculate_reward_cumulative()

        rewards = soc_reward * soc_reward_weight + \
                  action_taking_reward * action_taking_reward_weight + \
                  income_reward  + consecutive_action_reward * consecutive_action_reward_weight + cumulative_reward * self.cumulative_coef

        # Option 2: Reward is the cumulative earnings
        #reward = self.calculate_reward_cumulative(price, action)
        # Option 3: Reward is earnings + SOC (penalize low SOC as it corresponds to no action) *lambda_SOC
        # General Idea:
        # Incentivice the model to charge the battery and compensate for costs of charging
        # Incentivice the model to hold the current SOC
        #rewards = self.calculate_earnings(action, price) + self.SOC * 0.25+ np.abs(action) *0.5
        # Option 4: Reward is earnings + SOC (penalize low SOC as it corresponds to no action) *lambda_SOC + action * lambda_action
        # rewards = self.calculate_earnings(action, price) + self.SOC * 0.01 + np.abs(action) * 0.01 + action * 0.01
        # Max reward possible: self.SOC = 96, action = 14,4, Energy Arbitrage: 0,15*14= 2,1 If we fully discharge the battery twice per day

        # Log to wandb
        if self.wandb_log:
            if self.eval_env:
                self.wandb_run.log({f"EVAL_reward": rewards,
                                f"EVAL_action": action,
                                f"EVAL_price": price,
                                f"EVAL_SOC": self.SOC,
                                f"EVAL_action_valid": float(self.action_valid[-1]),
                                f"EVAL_total_earnings": self.TOTAL_EARNINGS,
                                f"EVAL_monthly_earnings": np.mean(self.ROLLING_EARNINGS[-672:]),
                                f"EVAL_monthly_earnings (daily average)": np.mean(self.ROLLING_EARNINGS[-672:])*(24*4),
                                #f"EVAL_action_taking_reward": action_taking_reward,
                                #f"EVAL_income_reward": income_reward,
                                #f"EVAL_consecutive_action_reward": consecutive_action_reward,
                                #f"EVAL_cumulative_reward": cumulative_reward,
                                #f"EVAL_soc_reward": soc_reward,
                                #f"EVAL_action_taking_reward_weight": action_taking_reward_weight,
                                #f"EVAL_consecutive_action_reward_weight": consecutive_action_reward_weight,
                                #f"EVAL_soc_reward_weight": soc_reward_weight,
                                f"n_env_steps": self.n_steps,
                                # convert datetime object to unix timestamp
                                f"time": self.get_current_timestamp()
                                })
            else:
                self.wandb_run.log({"reward": rewards,
                                "action": action,
                                "price": price,
                                "SOC": self.SOC,
                                "action_valid": float(self.action_valid[-1]),
                                "total_earnings": self.TOTAL_EARNINGS,
                                "monthly_earnings": np.mean(self.ROLLING_EARNINGS[-672:]),
                                "monthly_earnings (daily average)": np.mean(self.ROLLING_EARNINGS[-672:])*(24*4),
                                "action_taking_reward": action_taking_reward,
                                "income_reward": income_reward,
                                "consecutive_action_reward": consecutive_action_reward,
                                "cumulative_reward": cumulative_reward,
                                "soc_reward": soc_reward,
                                "action_taking_reward_weight": action_taking_reward_weight,
                                "consecutive_action_reward_weight": consecutive_action_reward_weight,
                                "soc_reward_weight": soc_reward_weight,
                                f"n_env_steps": self.n_steps,
                                f"time": self.get_current_timestamp()
                                    })
        # Check if reward is a numpy array

        if isinstance(rewards, np.ndarray):
            rewards = rewards.ravel()
            rewards = rewards[0]

        if isinstance(rewards, np.ndarray):
            rewards = rewards[0]
        # Saves time of day (For pretraining with non sb3-models)
        #self.tod = next_state[-3]
        if self.eval_env:
            if self.n_steps % self.max_steps == 0:
                done = True
        elif self.n_steps == self.max_steps:
            done = True

        next_state_dict = {
            "features": next_state,
            "action_bounds": torch.Tensor(self.bounds)}
        #return next_state, rewards, done, {}
        return next_state_dict, rewards, done, {}
        #return next_state, rewards, done, {}

    def calculate_reward_cumulative(self):
        # Calculate earnings
        # Calculate reward
        cumulative_reward = 0.0
        if self.get_current_timestamp().hour *60 + self.get_current_timestamp().minute == 0:
            cumulative_reward = np.sum(self.ROLLING_EARNINGS[-96:])
        return cumulative_reward

    def calculate_reward_weights(self):
        soc_reward_weight = self.get_soc_reward_weight(self.n_steps)
        action_taking_reward_weight = self.get_action_taking_reward_weight(self.n_steps)
        consecutive_action_reward_weight = self.get_consecutive_action_reward_weight(self.n_steps)

        return soc_reward_weight, action_taking_reward_weight, consecutive_action_reward_weight

    def calculate_shaping_rewards(self, action, price):
        soc_reward = self.SOC
        action_taking_reward = np.abs(action)
        consecutive_action_reward = 0

        # If the action is the same as the last 6 actions the agent gets a reward

        if len(self.action_list) > 4:
            if np.all(np.round(self.action_list[-7:], 2) == np.round(action, 2)):
                consecutive_action_reward = 0
            elif np.all(np.round(self.action_list[-3:],2) == np.round(action, 2)):
                consecutive_action_reward = 1 * 3
            # Dont give the reward if the same action was taken more than 6 times
        self.action_list.append(action)

        return soc_reward, action_taking_reward, consecutive_action_reward

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
        self.ROLLING_EARNINGS.append(earnings)

        return earnings

class DiscreteEnvironment(Environment):
    def __init__(self, max_charge=0.15,
                 total_storage_capacity=1,
                 initial_charge=0.0,
                 max_SOC=1,
                 price_time_horizon=1.5,
                 data_root_path="..",
                 time_interval="15min",
                 wandb_run=None,
                 n_past_timesteps=1,
                 time_features=True,
                 day_ahead_environment=False,
                 prediction_output="action",
                 max_steps=None,
                 reward_shaping=None,
                 cumulative_coef=0.01,
                 eval_env = False,
                 n_steps = None,
                 gaussian_noise = False,
                 noise_std = 0.01,
                 ):
        """
        Initialize the Environment

        Args:
            max_charge: Maximum charge/discharge rate
            total_storage_capacity: Total Storage Capacity of the Battery
            initial_charge: Initial SOC of the Battery
            max_SOC: Maximum SOC of the Battery
            price_time_horizon:
            data_root_path: Path to the data
            time_interval: time per step (15min, 30min, H)
            wandb_run: Wandb run to log the data
            n_past_timesteps: number of past day_ahead_steps to use as input
            time_features: Use time features
            day_ahead_environment: -
            prediction_output: We can either predict the action or the future SOC, or percentage of potential charge/discharge
            max_steps: Maximum number of steps in the environment
            reward_shaping: Reward shaping schedule, None, "linear", "constant", "cosine", "exponential"
            cumulative_coef: Coefficient for the cumulative reward shaping
            eval_env: If the environment is used for evaluation (Evaluation Environment is always one step ahead of the training environment)
            n_steps: Number of steps in the environment, needed for evaluation Environments (No influence on training environments)
            gaussian_noise: Add gaussian noise to the day ahead prices and intra day prices
            noise_std: Standard deviation of the gaussian noise
        """
        # Set the action space to 5 discrete actions
        self.action_space = spaces.Discrete(5)
        self.discrete_to_continuous = {0: -0.15, 1: -0.075, 2: 0, 3: 0.075, 4: 0.15}
        self.discrete_to_continuous_array = np.fromiter(self.discrete_to_continuous.values(), dtype = float)
        # Call super constructor
        super().__init__(max_charge=max_charge,
                            total_storage_capacity=total_storage_capacity,
                            initial_charge=initial_charge,
                            max_SOC=max_SOC,
                            price_time_horizon=price_time_horizon,
                            data_root_path=data_root_path,
                            time_interval=time_interval,
                            wandb_run=wandb_run,
                            n_past_timesteps=n_past_timesteps,
                            time_features=time_features,
                            day_ahead_environment=day_ahead_environment,
                            prediction_output=prediction_output,
                            max_steps=max_steps,
                            reward_shaping=reward_shaping,
                            cumulative_coef=cumulative_coef,
                            eval_env=eval_env,
                            n_steps=n_steps,
                            gaussian_noise=gaussian_noise,
                            noise_std=noise_std)

        # Discretize the action space
    def valid_action_mask(self):
        # Check if the action is valid for the ActionMasker
        epsilon = 1e-6
        valid_mask = (np.abs(self.discrete_to_continuous_array) <= self.max_charge + epsilon) & \
                (0 <= self.discrete_to_continuous_array + self.SOC + epsilon) & \
                (self.discrete_to_continuous_array + self.SOC <= self.TOTAL_STORAGE_CAPACITY + epsilon)

        return valid_mask

    def step(self, action):
        """
        Step the Environment
        Args:
            action: Action to take

        Returns:
            next_state: Next State
            reward: Reward
            done: If the episode is done
            info: Additional Information
        """
        # Convert the action to a continuous action
        action = np.array([self.discrete_to_continuous[action]])
        next_state_dict, rewards, done, _ = super().step(action)
        return next_state_dict, rewards, done, _

if __name__ == "__main__":
    env = Environment(data_root_path="../../")
    #env = Environment(data_root_path="../../")
    env.reset()
    while True:
        action = (np.random.random(1) - 0.5) * 0.3
        next_state, reward, done, info = env.step(action)
        print(next_state, reward, done, info)
        if done:
            break
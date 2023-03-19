import torch
from gym import core
from gym import spaces
import numpy as np
from batterytrading.data_loader import Data_Loader_np, Data_Loader_np_solar, RandomSampleDataLoader
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

class ContinousEnergyArbitrageEnvironment(core.Env):
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
                 reward_shaping={"type": "linear", "coef": 0.01},
                 cumulative_coef=0.01,
                 eval_env = False,
                 skip_steps = None,
                 gaussian_noise = False,
                 noise_std = 0.01,
                 n_steps_till_eval = 672
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
            skip_steps: Number of steps in the environment, needed for evaluation Environments (No influence on training environments)
            gaussian_noise: Add gaussian noise to the day ahead prices and intra day prices
            noise_std: Standard deviation of the gaussian noise
        """
        self.n_steps_till_eval = n_steps_till_eval
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

        self.data_loader = Data_Loader_np_solar(price_time_horizon=price_time_horizon,
                                          root_path=data_root_path,
                                          time_interval=time_interval,
                                          n_past_timesteps=n_past_timesteps,
                                          time_features=time_features,
                                          gaussian_noise=gaussian_noise,
                                          noise_std=noise_std,
                                          start_index=skip_steps
                                          )
        self.n_steps = skip_steps

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

        self.eval_env = eval_env
        #if eval_env:
        #    self.max_steps = n_steps_till_eval # We want to evaluate the agent for n_steps (same stepwidth as during training -> This ensures that the agent evaluated for the same amount of time as trained)
        #    self.eval_episode = 0
        #    dummy_action = np.array([0])
            # Skip the first n_steps + 3 Steps (To ensure that the agent is evaluated one timehorizon ahead of training)
        self.startTime = self.get_current_timestamp()
        #self.skip_steps_vector(skip_steps)
        #self.reset()
        # Initialize the wandb run (if wandb is used)
        if not (wandb_run is None):
            self.wandb_log = True
            self.wandb_run = wandb_run
        else:
            self.wandb_log = False
        if self.eval_env:
            self.prefix = "eval_"
        else:
            self.prefix = "train_"

    def skip_steps_vector(self, skip_steps):
        self.n_steps = skip_steps
        self.data_loader.index = skip_steps
        self.data_loader.get_next_day_ahead_price()

        #if not skip_steps is None:
        #    for i in range(0, skip_steps ):
        #        self.step(np.array([0]))
        print(self.get_current_timestamp())
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
        #else:
        #    self.eval_episode += 1
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
        # Check if action is integer and convert to float if it is an array
        if not isinstance(action, float):
            action = action[0]
        #else:
        #    action = np.array([action])
        self.SOC = self.charge(action)
        if self.day_ahead_environment:
            features, done = self.data_loader.get_next_day_ahead_price()
            intra_day_price, done= self.data_loader.get_next_intraday_price()
            price = intra_day_price[0]
        else:
            features, price, done = self.data_loader.get_next_features()
            #day_ahead, done = self.data_loader.get_next_day_ahead_price(set_current_index=False)
        #features = np.hstack([features, day_ahead])
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
        self.down_max_charge = down_max_charge
        self.up_max_charge = up_max_charge
        self.bounds = bounds

    def step(self, action):
        """
        Perform a (15min) step in the Environment
        Args:
            action: action to perform (hold, charge, discharge)

        Returns:
            next state, reward, done, info
        """
        action = action[:1]
        self.n_steps += 1

        if self.prediction_output == "nextSOC":
            action = action - self.SOC # action is now the difference of nextSOC to the current SOC
        elif self.prediction_output == "percentage_action":
            action = (action-0.5) * self.max_charge
        epsilon = 1e-1

        # Clip action into a valid space
        valid = (np.abs(action) <= self.max_charge + epsilon) & (0 <= action + self.SOC + epsilon) & (action + self.SOC <= self.TOTAL_STORAGE_CAPACITY + epsilon)
        if not valid:
            #print("Action not valid: {}".format(action))
            pass
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
            self.wandb_run.log({f"{self.prefix}reward": rewards,
                                f"{self.prefix}action": action,
                                f"{self.prefix}price": price,
                                f"{self.prefix}SOC": self.SOC,
                                f"{self.prefix}action_valid": float(self.action_valid[-1]),
                                f"{self.prefix}total_earnings": self.TOTAL_EARNINGS,
                                f"{self.prefix}monthly_earnings": np.mean(self.ROLLING_EARNINGS[-672:]),
                                f"{self.prefix}monthly_earnings (daily average)": np.mean(
                                    self.ROLLING_EARNINGS[-672:]) * (24 * 4),
                                # f"EVAL_action_taking_reward": action_taking_reward,
                                # f"EVAL_income_reward": income_reward,
                                # f"EVAL_consecutive_action_reward": consecutive_action_reward,
                                # f"EVAL_cumulative_reward": cumulative_reward,
                                # f"EVAL_soc_reward": soc_reward,
                                # f"EVAL_action_taking_reward_weight": action_taking_reward_weight,
                                # f"EVAL_consecutive_action_reward_weight": consecutive_action_reward_weight,
                                # f"EVAL_soc_reward_weight": soc_reward_weight,
                                f"n_env_steps": self.n_steps,
                                #f"sold_PV{self.prefix}": sold_PV,

                                # convert datetime object to unix timestamp
                                f"time": self.get_current_timestamp().timestamp()
                                })        # Check if reward is a numpy array

        if isinstance(rewards, np.ndarray):
            rewards = rewards.ravel()
            rewards = rewards[0]

        if isinstance(rewards, np.ndarray):
            rewards = rewards[0]
        # Saves time of day (For pretraining with non sb3-models)
        #self.tod = next_state[-3]
        if self.eval_env:
            if self.n_steps % self.n_steps_till_eval == 0:
                done = True
        elif self.n_steps == self.max_steps:
            done = True
        # Feasibility gap: This allows us to increase the action space slightly to allow for more feasible actions
        self.bounds[0] = self.bounds[0] - 0.01
        self.bounds[1] = self.bounds[1] + 0.01
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
        #action_taking_reward = self.get_action_taking_reward_weight(self.n_steps)
        # If the action is the same as the last 6 actions the agent gets a reward

        #if len(self.action_list) > 4:
        #    if np.all(np.round(self.action_list[-7:], 2) == np.round(action, 2)):
        #        consecutive_action_reward = 0
        #    elif np.all(np.round(self.action_list[-3:],2) == np.round(action, 2)):
        #        consecutive_action_reward = 1 * 3
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

    def calculate_earnings(self, rel_amount, price, sold_PV, price_kicker = np.array([1.0])):
        """
        Calculates Earnings in the last episode
        Args:
            rel_amount: relative amount of power sold
            price: price to sell

        Returns:
            Earnings/Reward
        """

        sold_amount = (-rel_amount) * self.TOTAL_STORAGE_CAPACITY
        # If the agent buys energy, the price is increased by 5ct/kWh
        #if sold_amount < 0:
        #    price_kicker = 1.2
        #else:
        if sold_amount < 0:
            price_kicker = price_kicker
        else:
            price_kicker = np.array([1])
        earnings_battery = sold_amount * (price * price_kicker)
        earnings_PV = sold_PV * price
        earnings = earnings_battery + earnings_PV

        # check if earnings is not a numpy array
        if not isinstance(earnings, np.ndarray):
            pass
        self.TOTAL_EARNINGS += earnings
        self.ROLLING_EARNINGS.append(earnings)

        return earnings, earnings_PV, earnings_battery

    def valid_action_bounds(self):
        return self.bounds

class Continous_EA_PV_Environment(ContinousEnergyArbitrageEnvironment):
    def __init__(self, *args, **kwargs
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
            skip_steps: Number of steps in the environment, needed for evaluation Environments (No influence on training environments)
            gaussian_noise: Add gaussian noise to the day ahead prices and intra day prices
            noise_std: Standard deviation of the gaussian noise
        """
        # Call super constructor
        super(Continous_EA_PV_Environment, self).__init__(*args, **kwargs
                                                          )
        # change action-space to 2 contious actions
        self.action_space = spaces.Box(low=-np.stack([self.max_charge, 0]),
           high=np.stack([self.max_charge, 1.0]), shape=(2,), dtype=np.float32)

    def step(self, action):
        """
        Perform a (15min) step in the Environment
        Args:
            action: action to perform (hold, charge, discharge)

        Returns:
            next state, reward, done, info
        """
        self.n_steps += 1


        action_EA_original = action[:1]
        action_PV = action[1:]
        # Last episodes solar_production

        #action_PV = min(action_PV, self.data_loader.solar_production)

        saved_PV_original = self.data_loader.solar_production * action_PV

        saved_PV = min(saved_PV_original, self.up_max_charge)
        sold_PV = self.data_loader.solar_production - saved_PV
        self.SOC = self.charge(saved_PV)

        # Clip the totally saved energy into the valid space
        if action_EA_original >= 0:
            # TODO: THere is an error, since actionEA will be used to calculate earnings
            action_EA = min(action_EA_original, self.up_max_charge - saved_PV)
            #action_EA = min(action_EA_original, self.up_max_charge)
        else:
            action_EA = max(action_EA_original, self.down_max_charge)
        """
                else:
            action_EA_original



            # In case the agent sells, we store less PV energy, the additional energy is sold
            action_EA_original = max(action_EA_original, self.down_max_charge) # Calculate the actual sold amount
            if -action_EA_original > saved_PV:
                action_EA = action_EA_original + saved_PV
                saved_PV = np.array([0.0])
                sold_PV = self.data_loader.solar_production
            else:
                action_EA = np.array([0.0])
                saved_PV = saved_PV + action_EA

                sold_PV = self.data_loader.solar_production - saved_PV
        """


        epsilon = 1e-6
        # Clip action into a valid space
        valid = (np.abs(action_EA) <= self.max_charge + epsilon) & \
                (0 <= action_EA + self.SOC + epsilon) & \
                (action_EA + self.SOC <= self.TOTAL_STORAGE_CAPACITY + epsilon)

        self.action_valid.append(float(valid))

        # TODO: ADD SOLD PV HERE
        # Calculate next state
        next_state, price, done = self._get_next_state(action_EA + saved_PV)


        # Calculate non income rewards
        soc_reward, action_taking_reward, consecutive_action_reward = self.calculate_shaping_rewards(action_EA, price)
        # Calculate reward weights
        soc_reward_weight, action_taking_reward_weight, consecutive_action_reward_weight = self.calculate_reward_weights()
        income_reward, earnings_PV, earnings_battery = self.calculate_earnings(action_EA, sold_PV , price)
        # Cumulative reward
        # Calculate reward/earnings
        # Different reward functions can be used
        cumulative_reward = 0.0 #self.calculate_reward_cumulative()

        rewards = soc_reward * soc_reward_weight + \
                  action_taking_reward * action_taking_reward_weight + \
                  income_reward  + consecutive_action_reward * consecutive_action_reward_weight + cumulative_reward * self.cumulative_coef

        if isinstance(sold_PV, np.ndarray):
            sold_PV = sold_PV[0]
        if self.wandb_log:
            time_since_start = (self.get_current_timestamp() - self.startTime)
            wandb_time = self.startTime
            self.wandb_run.log({f"{self.prefix}reward": rewards,
                            f"{self.prefix}action_final": action_EA,
                            f"{self.prefix}price": price,
                            f"{self.prefix}SOC": self.SOC,
                            f"{self.prefix}action_valid": float(self.action_valid[-1]),
                            f"{self.prefix}total_earnings": self.TOTAL_EARNINGS,
                            f"{self.prefix}monthly_earnings": np.mean(self.ROLLING_EARNINGS[-672:]),
                            f"{self.prefix}monthly_earnings (daily average)": np.mean(self.ROLLING_EARNINGS[-672:])*(24*4),
                            f"{self.prefix}sold_PV": sold_PV,
                            f"{self.prefix}action_PV": action_PV,
                            f"{self.prefix}saved_PV": saved_PV,
                            f"{self.prefix}action_EA_original": action_EA_original,
                            f"EVAL_action_taking_reward": action_taking_reward,
                            f"EVAL_income_reward": income_reward,
                            f"EVAL_consecutive_action_reward": consecutive_action_reward,
                            f"EVAL_cumulative_reward": cumulative_reward,
                            f"EVAL_soc_reward": soc_reward,
                            #f"EVAL_action_taking_reward_weight": action_taking_reward_weight,
                            #f"EVAL_consecutive_action_reward_weight": consecutive_action_reward_weight,
                            #f"EVAL_soc_reward_weight": soc_reward_weight,
                            f"n_env_steps": self.n_steps,
                            f"{self.prefix}share_of_sold PV": sold_PV / (saved_PV + sold_PV),
                            f"{self.prefix}solar_production": self.data_loader.solar_production,
                            # convert datetime object to unix timestamp
                            f"{self.prefix}earnings_PV": earnings_PV,
                            f"{self.prefix}earnings_battery":  earnings_battery,
                            f"{self.prefix}up_max_charge": self.up_max_charge,
                            f"{self.prefix}down_max_charge": self.down_max_charge,
                            f"time": self.get_current_timestamp().timestamp()
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
            if self.n_steps % self.n_steps_till_eval == 0:
                done = True
        elif self.n_steps == self.max_steps:
            done = True
        # Feasibility gap: This allows us to increase the action space slightly to allow for more feasible actions
        self.bounds[0] = self.bounds[0] - 0.01
        self.bounds[1] = self.bounds[1] + 0.01
        next_state_dict = {
            "features": next_state,
            "action_bounds": torch.Tensor(self.bounds)}
        #return next_state, rewards, done, {}
        return next_state_dict, rewards, done, {}

    def _get_next_state(self, action_EA):
        """
        Get the next state of the Environment consisting of the SOC, day_ahead-action and the price
        Args:
            action_EA: action to perform (hold, charge, discharge)
        Returns:
            next state, price, done
        """
        # Check if action is integer and convert to float if it is an array
        if not isinstance(action_EA, float):
            action_EA = action_EA[0]
        #else:
        #    action = np.array([action])
        self.SOC = self.charge(action_EA )
        if self.day_ahead_environment:
            features, done = self.data_loader.get_next_day_ahead_price()
            intra_day_price, done= self.data_loader.get_next_intraday_price()
            price = intra_day_price[0]
        else:
            features, price, done = self.data_loader.get_next_features()
            #day_ahead, done = self.data_loader.get_next_day_ahead_price(set_current_index=False)
        #features = np.hstack([features, day_ahead])
        up_max_charge = min((self.max_SOC - self.SOC), self.max_charge)
        down_max_charge = max(-(self.SOC - self.min_SOC), -self.max_charge)

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
        self.down_max_charge = down_max_charge
        self.up_max_charge = up_max_charge
        self.bounds = bounds

class Continous_EA_PV_EnvironmentONEACTION(ContinousEnergyArbitrageEnvironment):
    def __init__(self, *args, **kwargs
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
            skip_steps: Number of steps in the environment, needed for evaluation Environments (No influence on training environments)
            gaussian_noise: Add gaussian noise to the day ahead prices and intra day prices
            noise_std: Standard deviation of the gaussian noise
        """
        # Call super constructor
        super(Continous_EA_PV_EnvironmentONEACTION, self).__init__(*args, **kwargs
                                                       )

        self.maxsolarproduction = 0.2
        # change action-space to 2 contious actions
        self.action_space = spaces.Box(low=-0.15,
           high = 0.15, shape=(1,), dtype=np.float32)
        # self.max_charge # original action space
    def step(self, action):
        """
        Perform a (15min) step in the Environment
        Args:
            action: action to perform (hold, charge, discharge)

        Returns:
            next state, reward, done, info
        """
        self.n_steps += 1
        solarproduction = self.data_loader.solar_production
        # Clip the totally saved energy into the valid space

        #self.action_valid.append(float(valid))
        #action = 0
        #Clip actions to maximum and minimum
        action = np.clip(action, self.down_max_charge, self.up_max_charge)
        # Calculate, how much of the action can be satisfied by the PV energy
        if action > 0:
            # First use PV to charge the battery, then buy energy
            # Is PV energy larger than the energy to be stored?
            saved_pv = min(solarproduction, action)
            # In case we have more energy than we can store at once, we sell the rest
            sold_PV = solarproduction - saved_pv
            # Rest of the action is satisfied by buying energy (grid_action)
            grid_action = action - saved_pv
            battery_action = grid_action + saved_pv
        else:
            # How much of energy should be sold from the battery, in contrast to the PV energy
            #energy_to_sell = min(-action, self.SOC)
            energy_to_buy = 0
            sold_PV = solarproduction
            battery_action = action
            saved_pv = 0.0

        # Check if sold PV + saved PV is equal to produced solar
        assert np.isclose(solarproduction, sold_PV + saved_pv), "PV energy not equal to produced solar energy"

        # Check that sold pv must be 0 if action is positive
        #assert sold_PV == 0 or action <= 0, "PV energy must be 0 if action is positive"
        #if sold_PV > 0 and action > 0:
        #    print(sold_PV, sold_PV, solarproduction, battery_action)
        # Next state is independent of PV energy (only depends on grid interaction
        next_state, price, done = self._get_next_state(battery_action)

        # Calculate non income rewards
        soc_reward, action_taking_reward, consecutive_action_reward = self.calculate_shaping_rewards(action, price)
        # Calculate reward weights
        soc_reward_weight, action_taking_reward_weight, consecutive_action_reward_weight = self.calculate_reward_weights()

        income_reward, earnings_pv, earnings_battery = self.calculate_earnings(rel_amount=battery_action, price=price, sold_PV=sold_PV)
        # Cumulative reward
        # Calculate reward/earnings
        # Different reward functions can be used
        cumulative_reward = 0.0 #self.calculate_reward_cumulative()
        # income_reward2 = earnings_pv * 0.0 + earnings_battery
        rewards = soc_reward * soc_reward_weight + \
                  action_taking_reward * action_taking_reward_weight + \
                  income_reward  + consecutive_action_reward * consecutive_action_reward_weight + cumulative_reward * self.cumulative_coef

        if isinstance(sold_PV, np.ndarray):
            sold_PV = sold_PV[0]
        if self.wandb_log:
            time_since_start = (self.get_current_timestamp() - self.startTime)
            wandb_time = self.startTime
            self.wandb_run.log({f"{self.prefix}reward": rewards,
                            f"{self.prefix}action_final": action,
                            f"{self.prefix}price": price,
                            f"{self.prefix}SOC": self.SOC,
                            #f"{self.prefix}action_valid": float(self.action_valid[-1]),
                            f"{self.prefix}total_earnings": self.TOTAL_EARNINGS,
                            f"{self.prefix}monthly_earnings": np.mean(self.ROLLING_EARNINGS[-672:]),
                            f"{self.prefix}monthly_earnings (daily average)": np.mean(self.ROLLING_EARNINGS[-672:])*(24*4),
                            f"{self.prefix}sold_PV": sold_PV,
                            f"{self.prefix}saved_PV": saved_pv,
                            f"EVAL_action_taking_reward": action_taking_reward,
                            f"EVAL_income_reward": income_reward,
                            f"EVAL_consecutive_action_reward": consecutive_action_reward,
                            f"EVAL_cumulative_reward": cumulative_reward,
                            f"EVAL_soc_reward": soc_reward,
                            #f"EVAL_action_taking_reward_weight": action_taking_reward_weight,
                            #f"EVAL_consecutive_action_reward_weight": consecutive_action_reward_weight,
                            #f"EVAL_soc_reward_weight": soc_reward_weight,
                            f"n_env_steps": self.n_steps,
                            f"{self.prefix}solar_production": self.data_loader.solar_production,
                            # convert datetime object to unix timestamp
                            f"{self.prefix}earnings_pv": earnings_pv,
                            f"{self.prefix}earnings_battery":  earnings_battery,
                            f"{self.prefix}up_max_charge": self.up_max_charge,
                            f"{self.prefix}down_max_charge": self.down_max_charge,
                                f"time": self.get_current_timestamp().timestamp()
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
            if self.n_steps % self.n_steps_till_eval == 0:
                done = True
        elif self.n_steps == self.max_steps:
            done = True
        # Feasibility gap: This allows us to increase the action space slightly to allow for more feasible actions
        self.bounds[0] = self.bounds[0] - 0.01
        self.bounds[1] = self.bounds[1] + 0.01

        next_state_dict = {
            "features": next_state,
            "action_bounds": torch.Tensor(self.bounds)}
        #return next_state, rewards, done, {}
        return next_state_dict, rewards, done, {}

    def _get_next_state(self, action):
        """
        Get the next state of the Environment consisting of the SOC, day_ahead-action and the price
        Args:
            action: action to perform (hold, charge, discharge)
        Returns:
            next state, price, done
        """
        # Check if action is integer and convert to float if it is an array
        if not isinstance(action, float):
            action = action[0]
        #else:
        #    action = np.array([action])
        self.SOC = self.charge(action)
        if self.day_ahead_environment:
            features, done = self.data_loader.get_next_day_ahead_price()
            intra_day_price, done= self.data_loader.get_next_intraday_price()
            price = intra_day_price[0]
        else:
            features, price, done = self.data_loader.get_next_features()
            #day_ahead, done = self.data_loader.get_next_day_ahead_price(set_current_index=False)
        #features = np.hstack([features, day_ahead])
        up_max_charge = min((self.max_SOC - self.SOC), self.max_charge)
        down_max_charge = max(-(self.SOC - self.min_SOC), -self.max_charge)

        #features = np.hstack([self.SOC, features])
        features = np.hstack([self.SOC, features,  down_max_charge, up_max_charge], dtype=np.float32)

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
        bounds = np.hstack([down_max_charge, up_max_charge], dtype=np.float32)
        self.down_max_charge = down_max_charge
        self.up_max_charge = up_max_charge
        self.bounds = bounds
    def valid_action_bounds(self):
        return self.bounds


class Discrete_EA_PV_Environment(Continous_EA_PV_Environment):
    def __init__(self, *args, **kwargs):

        """

        Args:
            *args:
            **kwargs:
        """
        """
                self.discrete_to_continuous = {0: np.array((-0.15, 0.0)),
                                       1: np.array((-0.075, 0.0)),
                                       2: np.array((0.0, 0.0)),
                                       3: np.array((0.075, 0)),
                                       4: np.array((0.15, 0)),
                                       5: np.array((0.0, 0.33)),
                                       6: np.array((0.0, 0.66)),
                                       7: np.array((0.0, 1)),
                                       8: np.array((0.075, 0.33)),
                                       9: np.array((0.075, 0.66)),
                                       10: np.array((0.075, 1)),
                                       11: np.array((0.15, 0.33)),
                                       12: np.array((0.15, 0.66)),
                                       13: np.array((0.15, 1.0)),

                                       }
                                               self.discrete_to_continuous = {0: np.array((-0.15, 0.0)),
                                       1: np.array((-0.075, 0.0)),
                                       2: np.array((0.0, 0.0)),
                                       3: np.array((0.0, 0.33)),
                                       4: np.array((0.0, 0.66)),
                                       5: np.array((0.0, 1)),
                                       6: np.array((0.075, 1.0)),
                                       7: np.array((0.145, 1.0))


                                       }
                                       """

        self.discrete_to_continuous = {0: np.array((-0.15, 0.0)),
                                       1: np.array((-0.075, 0.0)),
                                       2: np.array((0.0, 0.0)),
                                       3: np.array((0.0, 0.33)),
                                       4: np.array((0.0, 0.66)),
                                       5: np.array((0.0, 1.0)),



                                       }
        self.discrete_to_continuous_array = np.stack(self.discrete_to_continuous.values(), dtype = float)
        # Call super constructor
        super().__init__(*args, **kwargs)

        self.action_space = spaces.Discrete(len(self.discrete_to_continuous))

        pass
        # Discretize the action space
    def valid_action_mask(self):
        # Check if the action is valid for the ActionMasker
        epsilon = 1e-6
        possible_actions = self.discrete_to_continuous_array.copy()
        solar_production = self.data_loader.solar_production
        possible_actions[:, 1] = possible_actions[:, 1] * solar_production

        # Solar Action <= solar_production
        valid_mask_max_production = possible_actions[:, 1] <= solar_production
        # Discharge must not be smalelr than SOC
        valid_mask_maxdischarge = possible_actions.sum(axis=1) >= -self.SOC

        # Charge must not exceed left over capacity
        valid_mask_maxcharge  = possible_actions.sum(axis=1) <= self.TOTAL_STORAGE_CAPACITY - self.SOC

        valid_mask = valid_mask_max_production * valid_mask_maxdischarge * valid_mask_maxcharge
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
        action = self.discrete_to_continuous[action]
        next_state_dict, rewards, done, _ = super().step(action)
        return next_state_dict, rewards, done, _

class DiscreteContinousEnergyArbitrageEnvironment(ContinousEnergyArbitrageEnvironment):
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
                 skip_steps = None,
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
            skip_steps: Number of steps in the environment, needed for evaluation Environments (No influence on training environments)
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
                         skip_steps=skip_steps,
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

reward_shaping = {  "type": "linear",
    "base_soc_reward": 0.0,
    "base_action_taking_reward": -1 ,
    "base_consecutive_action_reward": 0.0,
    "base_income_reward": 1.0,
    "cumulative_coef": 0.0}
if __name__ == "__main__":
    env = Continous_EA_PV_Environment(data_root_path="../../", skip_steps=0, reward_shaping = reward_shaping)

    #env = Environment(data_root_path="../../")
    env.reset()
    while True:
        action = env.action_space.sample() #(np.random.random(1) - 0.5) * 0.3
        next_state, reward, done, info = env.step(action)
        print(next_state, reward, done, info)
        if done:
            break

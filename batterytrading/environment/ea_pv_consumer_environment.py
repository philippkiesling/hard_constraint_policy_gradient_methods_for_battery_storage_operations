from batterytrading.environment.environment_dict import ContinousEnergyArbitrageEnvironment
import torch
from gym import core
from gym import spaces
import numpy as np
from batterytrading.data_loader import Data_Loader_np, Data_Loader_np_solar, RandomSampleDataLoader
from gym.wrappers import NormalizeObservation
from gym.wrappers.normalize import RunningMeanStd

class Continous_EA_PV_Consumer(ContinousEnergyArbitrageEnvironment):
    def __init__(self, *args,consume_price = 20.0, max_demand = 9999, **kwargs,
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
        self.consume_price = consume_price
        self.max_demand = max_demand
        super(Continous_EA_PV_Consumer, self).__init__(*args, **kwargs
                                                       )

        self.maxsolarproduction = 0.2
        # change action-space to 2 contious actions
        self.action_space = spaces.Box(low=np.stack([-self.max_charge, -0.2]),
           high=np.stack([self.max_charge, 0.0]), shape=(2,))
        self.ROLLING_BASELINEINCOME = []
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
        solarproduction_original = self.data_loader.solar_production
        # Clip the totally saved energy into the valid space
        action_battery,  = action[:1]
        action_consumer = action[1:]
        #self.action_valid.append(float(valid))
        #action = 0
        #Clip actions to maximum and minimum
        action_consumer = np.clip(action_consumer, max(-self.max_demand, self.down_max_charge-solarproduction_original), 0.0)
        # Calculate, how much of the action can be satisfied by the PV energy
        # Calculate consumer action:
        # First consume as much energy as possible from the solar_production
        energy_consumed_from_solar = max(-solarproduction_original, action_consumer)
        solarproduction =solarproduction_original + energy_consumed_from_solar
        # consume The rest from the battery
        energy_consumed_from_battery = action_consumer - energy_consumed_from_solar
        # Calculate battery action
        #Reduce up_max_charge and down_max_charge by the amount consumed from battery
        # We use plus here, because the action is negative
        # Update SOC
        energy_consumed = energy_consumed_from_battery + energy_consumed_from_solar
        self.SOC += energy_consumed_from_battery
        # Update new valid space
        self.up_max_charge += energy_consumed_from_battery
        self.down_max_charge -= energy_consumed_from_battery
        # Clip the action to the new valid space
        action_battery = np.clip(action_battery, self.down_max_charge, self.up_max_charge)

        if action_battery > 0:
            # First use PV to charge the battery, then buy energy
            # Is PV energy larger than the energy to be stored?
            saved_pv = min(solarproduction, action_battery)
            # In case we have more energy than we can store at once, we sell the rest
            sold_PV = solarproduction - saved_pv
            # Rest of the action is satisfied by buying energy (grid_action)
            grid_action = action_battery - saved_pv
            battery_action = grid_action + saved_pv
        else:
            # How much of energy should be sold from the battery, in contrast to the PV energy
            #energy_to_sell = min(-action, self.SOC)
            energy_to_buy = 0
            sold_PV = solarproduction
            battery_action = action_battery
            saved_pv = 0.0

        # Check if sold PV + saved PV is equal to produced solar
        assert np.isclose(solarproduction, sold_PV + saved_pv), "PV energy not equal to produced solar energy"

        # Check that sold pv must be 0 if action is positive
        #assert sold_PV == 0 or action <= 0, "PV energy must be 0 if action is positive"
        #if sold_PV > 0 and action > 0:
        #    print(sold_PV, sold_PV, solarproduction, battery_action)
        # Next state is independent of PV energy (only depends on grid interaction
        next_state, price, done = self._get_next_state(battery_action+ energy_consumed_from_battery)

        # Calculate non income rewards
        soc_reward, action_taking_reward, consecutive_action_reward = self.calculate_shaping_rewards(battery_action, price)
        # Calculate reward weights
        soc_reward_weight, action_taking_reward_weight, consecutive_action_reward_weight = self.calculate_reward_weights()

        income_reward, earnings_pv, earnings_battery = self.calculate_earnings(rel_amount=battery_action, price=price, sold_PV=sold_PV)
        #if self.n_steps < 50000:
        #    consume_reward_weight = 1.0 #-5.0
        #else:
        consume_reward_weight = 1.0
        consume_reward = -energy_consumed * self.consume_price * consume_reward_weight
        # Cumulative reward
        # Calculate reward/earnings
        # Different reward functions can be used
        cumulative_reward = 0.0 #self.calculate_reward_cumulative()
        # income_reward2 = earnings_pv * 0.0 + earnings_battery
        rewards = soc_reward * soc_reward_weight + \
                  action_taking_reward * action_taking_reward_weight + \
                  income_reward  + \
                  consume_reward + \
                  consecutive_action_reward * consecutive_action_reward_weight + \
                  cumulative_reward * self.cumulative_coef

        baseline_reward_sellallsolar = solarproduction_original * price
        if self.consume_price > price:
            baseline_reward_sellandconsume = solarproduction_original * self.consume_price
        else:
            baseline_reward_sellandconsume = solarproduction_original * price
        self.ROLLING_BASELINEINCOME.append(baseline_reward_sellandconsume)
        rewards = rewards - baseline_reward_sellandconsume
        if isinstance(sold_PV, np.ndarray):
            sold_PV = sold_PV[0]
        if self.wandb_log:
            time_since_start = (self.get_current_timestamp() - self.startTime)
            wandb_time = self.startTime
            self.wandb_run.log({f"{self.prefix}reward": rewards,
                            f"{self.prefix}action_final": action_battery,
                            f"{self.prefix}price": price,
                            f"{self.prefix}SOC": self.SOC,
                            #f"{self.prefix}action_valid": float(self.action_valid[-1]),
                            f"{self.prefix}total_earnings": self.TOTAL_EARNINGS,
                            f"{self.prefix}monthly_earnings": np.mean(self.ROLLING_EARNINGS[-672:]),
                            f"{self.prefix}monthly_earnings (daily average)": np.mean(self.ROLLING_EARNINGS[-672:])*(24*4),
                            f"{self.prefix}monthly_baseline_earnings": np.mean(self.ROLLING_BASELINEINCOME[-672:])*(24*4),
                            f"{self.prefix}sold_PV": sold_PV,
                            f"{self.prefix}saved_PV": saved_pv,
                            f"{self.prefix}action_taking_reward": action_taking_reward,
                            f"{self.prefix}income_reward": income_reward,
                            f"{self.prefix}consecutive_action_reward": consecutive_action_reward,
                            f"{self.prefix}cumulative_reward": cumulative_reward,
                            f"{self.prefix}soc_reward": soc_reward,
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
                            f"{self.prefix}consume_reward": consume_reward,
                            f"{self.prefix}energy_consumed": energy_consumed,
                            f"{self.prefix}energy_consumed_from_battery": energy_consumed_from_battery,
                            f"{self.prefix}energy_consumed_from_grid": energy_consumed_from_solar,
                            f"{self.prefix}solarproductionleftforbattery": solarproduction,
                            f"time": self.get_current_timestamp().timestamp(),
                            f"{self.prefix}baseline_income_sellallsolar": baseline_reward_sellallsolar,
                            f"{self.prefix}baseline_income_sellandconsume": baseline_reward_sellandconsume,
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

class Discrete_EA_PV_Consumer(Continous_EA_PV_Consumer):
    def __init__(self, *args, **kwargs):

        self.discrete_to_continuous = {0: np.array((-0.15, 0.0)),
                                       1: np.array((-0.075, 0.0)),
                                       2: np.array((0.0, 0.0)),
                                       3: np.array((0.075, 0)),
                                       4: np.array((0.15, 0)),

                                       5: np.array((-0.15, -0.05)),
                                       6: np.array((-0.075, -0.05)),
                                       7: np.array((0.0, -0.05)),
                                       8: np.array((0.075, -0.05)),
                                       9: np.array((0.15, -0.05)),

                                       10: np.array((-0.15, -0.075)),
                                       11: np.array((-0.075, -0.075)),
                                       12: np.array((0.0, -0.075)),
                                       13: np.array((0.075, -0.075)),
                                       14: np.array((0.15, -0.075)),

                                       15: np.array((-0.15, -0.10)),
                                       16: np.array((-0.075, -0.10)),
                                       17: np.array((0.0, -0.10)),
                                       18: np.array((0.075, -0.10)),
                                       19: np.array((0.15, -0.10))}

        self.discrete_to_continuous_array = np.stack(self.discrete_to_continuous.values(), dtype=float)
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Discrete(len(self.discrete_to_continuous))

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

    def valid_action_mask(self):
        # Check if the action is valid for the ActionMasker

        epsilon = 1e-6
        possible_actions = self.discrete_to_continuous_array.copy()
        solar_production = self.data_loader.solar_production



        #possible_actions[:, 1] = possible_actions[:, 1] * solar_production
        # Actions embedded through action space size: a_consumer < 0
        possible_consumer, possible_battery = possible_actions[:, 1], possible_actions[:, 0]
        valid_mask_max_discharge = possible_consumer + possible_battery >= -solar_production + self.down_max_charge

        # Dont consume more than demand of consumer (currently not binding)
        valid_mask_demand = possible_consumer >= - self.max_demand

        # Dont do more charge/discharge operation than phyiscally possible
        valid_mask_max_charge = possible_consumer - possible_battery >= -self.up_max_charge -solar_production

        # Ensure the battery interaction is lower than the given thresholds
        valid_mask_chargeBattery = possible_actions[:, 0]  <= self.up_max_charge
        valid_mask_dischargeBattery = possible_actions[:, 0] >= self.down_max_charge


        valid_mask = valid_mask_max_discharge * \
                     valid_mask_demand * \
                     valid_mask_max_charge * \
                     valid_mask_chargeBattery * \
                     valid_mask_dischargeBattery
        return valid_mask
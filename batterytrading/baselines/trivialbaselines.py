
class BaselineModel():
    def __init__(self, env, policy_fn="schedule", n_lowest=35, n_highest=35):
        self.env = env
        self.state, reward, done, info = env.step(0)
        self.max_SOC = self.env.max_SOC
        self.min_SOC = self.env.min_SOC
        self.max_charge = self.env.max_charge
        start_timestamp = env.get_current_timestamp()
        offset_from_0 = {"H": start_timestamp.hour, "M": start_timestamp.minute, "S":start_timestamp.second}
        if policy_fn == "schedule":
            self.policy_fn = self.get_scheduled_strategy
        elif policy_fn == "min_max":
            self.policy_fn = self.get_min_max_policy
        elif policy_fn == "optimal":
            NotImplementedError("Optimal policy not implemented yet")
        else:
            NotImplementedError(f"Policy function {policy_fn} not implemented yet")

        if env.time_step == "H":
            self.max_charge = self.env.max_charge * 4
            self.planning_horizon = 12 - offset_from_0["H"]
            self.new_policy_entries = 24
        else:
            self.planning_horizon = int((36 + 12-offset_from_0["H"]-(offset_from_0["M"]//15)/4)*4)
            self.new_policy_entries = 24 * 4
        self.n_lowest = n_lowest
        self.n_highest = n_highest

    def train(self, total_timesteps = 10000):
        """
        if the first value of the environments state is not nan, find the new policy.
        This means, that we have a new (day_ahead) forecast available

        """
        self.state, reward, done, info = self.env.step(0)
        self.policy = [0 for i in range(self.planning_horizon)]
        self.total_policy = []
        self.total_policy += self.policy
        iteration = 0
        while not done and iteration < total_timesteps:
            current_action = self.policy.pop(0)
            self.state, reward, done, info = self.env.step(current_action)
            if self.env.get_current_timestamp().hour == 12 and self.env.get_current_timestamp().minute == 15:
                self.policy_fn()
            iteration += 1
        return self.total_policy

    def get_optimal_policy(self):
        """
        Get the optimal policy for the current state by iterating over all possible sequences of actions
        pass
        """
        pass

    def get_min_max_policy(self):
        """
        Get the min max strategy for the current state
        The min max strategy is buys and sells based on the min and max SOC of the battery
        """
        day_ahead_price = self.state[-24*4:]
        # get the n-lowest and n-highest prices from the day ahead price
        n = self.n_lowest
        lowest_prices = np.argpartition(day_ahead_price, self.n_lowest)[:self.n_lowest]
        highest_prices = np.argpartition(day_ahead_price, -self.n_highest)[-self.n_highest:]

        new_policy = [0 for i in range(self.new_policy_entries)]
        for i in lowest_prices:
            new_policy[i] = 1 * self.max_charge
        for i in highest_prices:
            new_policy[i] = -1 * self.max_charge
        self.policy += new_policy
        self.total_policy += self.policy

    def get_scheduled_strategy(self):
        """
        Get the scheduled strategy for the current state
        The scheduled strategy is buys and sells based on a preset schedule.
        This schedule is based on the historic day ahead price.
        3-4:45 am are usually the cheapest hours, 16:30-18:15 are usually the most expensive hours
        """
        new_policy = [0 for i in range(self.new_policy_entries)]

        if self.env.time_step == "H":
            buy_range = np.arange(3,5)
            sell_range = np.arange(16,19)
        else: # 15min
            buy_range = range(3*4, 3*4+7 )  # 3:00 - 4:45
            sell_range = range(16*4 , 16*4+7)  # 16:30 - 18:15

        for i in buy_range:
            new_policy[i] = 1 * self.max_charge
        for i in sell_range:
            new_policy[i] = -1 * self.max_charge
        self.policy += new_policy
        self.total_policy += new_policy

if __name__ == '__main__':
    from batterytrading.ppo import get_config
    from batterytrading.environment import Environment
    import numpy as np

    # Get Conifguration
    model_cfg, train_cfg = get_config("./batterytrading/baselines/cfg.yml")

    n = 35
    policy_iteration = BaselineModel(model_cfg["env"], policy_fn="min_max", n_lowest=n, n_highest=n)
    policy_iteration.train(total_timesteps= train_cfg["total_timesteps"])

    """ 
    for n in range(0,38, 1):
        env = Environment(day_ahead_environment = True)
        policy_iteration = TrivialModel(env,  policy_fn="min_max", n_lowest = n, n_highest= n)
        policy_iteration.train(**train_cfg)
        print(n, policy_iteration.env.TOTAL_EARNINGS)
    """
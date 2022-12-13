from batterytrading.ppo import get_config
from batterytrading.environment import Environment
import numpy as np
# Get Conifguration
model_cfg, train_cfg = get_config("./batterytrading/rl_models/cfg.yml")

class PolicyIteration():
    def __init__(self, env):
        self.env = env
        self.state, reward, done, info = env.step(0)
        self.max_SOC = self.env.max_SOC
        self.min_SOC = self.env.min_SOC
        self.max_charge = self.env.max_charge
        if env.time_interval == "H":
            self.max_charge = self.env.max_charge * 4

    def train(self):
        """
        if the first value of the environments state is not nan, find the new policy.
        This means, that we have a new (day_ahead) forecast available

        """
        self.state, reward, done, info = env.step(0)
        policy = self.state.shape()-1
        while not done:
            if not np.isnan(self.state[1]):
                policy = self.get_optimal_policy()
                current_action = policy.pop(0)
            else:
                current_action = policy.pop(0)
            self.state, reward, done, info = self.env.step(current_action)

    def get_optimal_policy(self):
        """
        Get the optimal policy for the current state by iterating over all possible sequences of actions
        pass
        """

    def calculate_SOC(self):
        """
        Calculate the state of charge of the battery
        """





if __name__ == '__main__':
    env = Environment(time_interval = "H", day_ahead_environment=True)
    policy_iteration = PolicyIteration(env)
    policy_iteration.train()


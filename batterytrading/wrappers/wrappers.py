import gym
import numpy as np
class NormalizeObservationDict(gym.Wrapper):
    def __init__(self, env):
        super(NormalizeObservationDict, self).__init__(env)
        self.ob_mean = None
        self.ob_std = None

    def observation(self, observation):
        features = observation["features"]
        if self.ob_mean is None:
            self.ob_mean = np.mean(features, axis=0)
            self.ob_std = np.std(features, axis=0)
        observation["features"] = (features - self.ob_mean) / self.ob_std
        return observation

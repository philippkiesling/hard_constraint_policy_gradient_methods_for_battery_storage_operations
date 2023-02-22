import numpy as np
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv

EXPECTED_METHOD_NAME = "valid_action_bounds"

def get_action_bounds(env: GymEnv) -> np.ndarray:
    """
    Checks whether gym env exposes a method returning invalid action masks

    :param env: the Gym environment to get masks from
    :return: A numpy array of the masks
    """

    if isinstance(env, VecEnv):
        return np.stack(env.env_method(EXPECTED_METHOD_NAME))
    else:
        return getattr(env, EXPECTED_METHOD_NAME)()
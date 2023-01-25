from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import torch.nn as nn
import torch as th


class CustomLSTMExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space, features):
        feature_dim1 = 0
        for key in observation_space:
            if key in features:
                feature_dim1 += observation_space[key].shape[0]
        feature_dim = (feature_dim1, )
        super().__init__(observation_space, feature_dim1)
        self.features = features
        self.flatten = nn.Flatten()
        #self.concat = nn.Concat()
    def forward(self, observations: th.Tensor) -> th.Tensor:
        #input_features =
        return th.cat([observations[value] for value in self.features], axis =1)

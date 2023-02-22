from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import torch.nn as nn
import torch as th
from torch import nn as nn


class FilterForFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extract that only returns the features specified in the features list.
    :param features: (List[str]) List of features to extract
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


class ActionNet(nn.Module):
    def __init__(self, action_net, projection_layer, mapping_layer):

        super(ActionNet, self).__init__()
        self.net = action_net
        self.mapping_layer = mapping_layer
        #self.linear = nn.Linear(1, 1)
        self.projection_layer = projection_layer

    def forward(self, latent_pi, action_bounds = None) -> th.Tensor:
        """
        Forward pass for the action network.

        :param latent_pi: Latent code for the actor
        :return: Mean actions
        """
        mean_actions_original = self.net(latent_pi)
        #mean_actions_original = self.linear(mean_actions_original)
        if isinstance(action_bounds, type(None)):
            return mean_actions_original
        else:
            from copy import deepcopy
            mean_actions = self.mapping_layer(self, mean_actions_original, action_bounds)
        #mean_actions, mean_actions_original = torch.clamp(mean_actions, action_bounds[0][0]-1e-3, action_bounds[0][1] + 1e-3)

            return mean_actions, mean_actions_original

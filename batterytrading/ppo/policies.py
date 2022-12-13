from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
import gym
from stable_baselines3.common.torch_layers import MlpExtractor


class ClampedActorCriticPolicy(ActorCriticPolicy):
    """
    An Actor Critic Policy that maps actions sampled from the normal distribution into the valid action space.

    """
    def __init__(self,  *args, **kwargs):
        bounds = kwargs.pop("bounds")
        super(ClampedActorCriticPolicy, self).__init__(*args, **kwargs)
        self.low = bounds[0]
        self.high = bounds[1]

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        actions = torch.clamp(actions, self.low, self.high)

        return actions, values, log_prob

class ClampedActorCriticPolicy2(ActorCriticPolicy):
    """
    An Actor Critic Policy that maps actions sampled from the normal distribution into the valid action space.

    """
    def __init__(self,  *args, **kwargs):
        bounds = kwargs.pop("bounds")
        super(ClampedActorCriticPolicy2, self).__init__(*args, **kwargs)
        self.low = bounds[0]
        self.high = bounds[1]
        self.observation_space = gym.spaces.Box(low=-100, high=1000, shape=(self.features_dim - 2,))

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim -2,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def map_action_to_valid_space(self, action, clamp_params):
        action = torch.clamp(action, clamp_params[0][0] +0e10, clamp_params[0][1])
        return action

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        features = features[:, :-2]
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        features = features[:, :-2]
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()


    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Get last two entires of obs to clamp actions to valid action space
        valid_check = obs[:, -2:]

        # Preprocess the observation if needed
        features = self.extract_features(obs)
        features = features[:, :-2]

        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        actions = self.map_action_to_valid_space(actions, valid_check)
        return actions, values, log_prob

class LinearProjectedActorCriticPolicy(ClampedActorCriticPolicy2):
    """
    An Actor Critic Policy that maps actions sampled from the normal distribution into the valid action space.

    """

    def __init__(self, *args, **kwargs):
        super(LinearProjectedActorCriticPolicy, self).__init__(*args, **kwargs)
        self.projection_layer = self.construct_optimization_layer()

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        last_SOC = obs[:, 0]
        next_SOC, values, log_prob = super(LinearProjectedActorCriticPolicy, self).forward(obs, deterministic)
        return next_SOC , values, log_prob

    def map_action_to_valid_space(self, action, clamp_params):
        #action = torch.sigmoid(action)
        action = self.projection_layer(action, clamp_params[0][:1], clamp_params[0][1:2])[0]
        return action

    def construct_optimization_layer(self):
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer
        n = 1
        _x = cp.Parameter(1)
        _lower_bound = cp.Parameter(1)
        _upper_bound = cp.Parameter(1)
        _action = cp.Variable(n)
        obj = cp.Minimize(cp.sum_squares(_action - _x))
        cons = [_action >= _lower_bound, _action <= _upper_bound]
        prob = cp.Problem(obj, cons)

        layer = CvxpyLayer(prob, parameters=[_x, _lower_bound, _upper_bound], variables=[_action])
        return layer

if __name__ == '__main__':
    # A simple test to see if the policy works
    import gym
    from stable_baselines3 import PPO
    from batterytrading.ppo import get_config
    from batterytrading.ppo.policies import ClampedActorCriticPolicy
    # Get Conifguration
    model_cfg, train_cfg = get_config("./batterytrading/ppo/cfg.yml")
    # Get the environment
    env = model_cfg["env"]
    model_cfg["policy"] = ClampedActorCriticPolicy
    policy_type = model_cfg.pop("policy_type")

    # Create the model
    model = PPO(**model_cfg)
    # Train the model
    model.learn(**train_cfg)
    # Get the action space
    action_space = env.action_space

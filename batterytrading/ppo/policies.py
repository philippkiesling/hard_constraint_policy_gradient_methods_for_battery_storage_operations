from stable_baselines3.common.policies import ActorCriticPolicy
import torch
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

class LinearProjectionActorCriticPolicy(ActorCriticPolicy):
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

        return actions, values, log_prob

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

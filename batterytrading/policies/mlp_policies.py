from stable_baselines3.common.policies import ActorCriticPolicy
# import ActionNet from batterytrading.policies.action_net import ActionNet
from batterytrading.policies.torch_layers import ActionNet
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)

from batterytrading.policies.torch_layers import FilterForFeaturesExtractor, ActionNet
import torch
from sb3_contrib.common.recurrent.type_aliases import RNNStates
import torch.nn as nn
from stable_baselines3.common.type_aliases import Schedule
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)


from abc import ABC, abstractmethod
import gym
from stable_baselines3.common.torch_layers import MlpExtractor
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from batterytrading.policies.mapping_functions import map_action_to_valid_space_cvxpy_layer, \
    construct_cvxpy_optimization_layer, \
    map_action_to_valid_space_clamp, \
    construct_optimization_layer_dummy


#acp = ActorCriticPolicy()

class ValidOutputBaseActorCriticPolicy(ActorCriticPolicy):
    """
    An Actor Critic Policy that maps actions sampled from the normal distribution into the valid action space.
    This is an abstract class that provides the most important functionalities.
    To create a subclass based on this clas,
    """
    def __init__(self,  *args, pretrain = None, **kwargs):
        bounds = kwargs.pop("bounds")
        super(ValidOutputBaseActorCriticPolicy, self).__init__(*args, **kwargs)
        features_extractor_kwargs = {'features': ["features"]}
        features_extractor_class = FilterForFeaturesExtractor


        #self.standard_forward = self.forward_pass_standard
        self.projection_layer = self.construct_optimization_layer(self)
        self.action_net = ActionNet(self.action_net, self.projection_layer, self.map_action_to_valid_space)


    def construct_optimization_layer(self):
        """
        How to construct the layer mapping model action to a valid space
        Returns:
            torch.layer
        """
        raise NotImplementedError

    #def _build_mlp_extractor(self) -> None:
    #    """
    #    Create the policy and value networks.
    #    Part of the layers can be shared.
    #    """
    #    # Note: If net_arch is None and some features extractor is used,
    #    #       net_arch here is an empty list and mlp_extractor does not
    ##    #       really contain any layers (acts like an identity module).
    #    self.mlp_extractor = MlpExtractor(
    #        self.features_dim ,
    #        net_arch=self.net_arch,
    #        activation_fn=self.activation_fn,
    #        device=self.device,
    #    )

    def map_action_to_valid_space(self, action, clamp_params):
        """
        Map the action sampled from the normal distribution to the valid action space.
        :param action: The action sampled from the normal distribution
        :param clamp_params: The parameters that are used to clamp the action to the valid action space.
        :return: The action mapped to the valid action space.
        """
        # Throw not implemented error since this is an abstract class.
        pass

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation.
        :return: the estimated values.
        """
        features = self.extract_features(obs)
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
        action_bounds = obs["action_bounds"]
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        if True:
            mean_actions = self.action_net(latent_pi)
            #print("No action_bounds specified")
        else:
            mean_actions  = self.action_net(latent_pi)
            mean_actions = torch.clamp(mean_actions, action_bounds[:, 0:1], action_bounds[:, 1:2])

        distribution = self._get_action_dist_from_mean(mean_actions, latent_pi)
        #distribution = self.get_distribution_torch(mean_actions)
        #log_prob = distribution.log_prob(actions)
        #mask_high = actions == action_bounds[:, 1:]
        #mask_low = actions == action_bounds[:, 0:1]
        #mask_middle =  torch.logical_not(mask_high + mask_low)
        #log_prob = mask_high * (torch.log(1 - distribution.cdf(actions))) + mask_low * torch.log(distribution.cdf(actions)) + mask_middle * distribution.log_prob(actions)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = obs["features"]
        action_bounds = obs["action_bounds"]
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)

        if action_bounds == None:
            mean_actions = self.action_net(latent_pi)
            print("No bounds Specified")
        else:
            mu, mu_original = self.action_net(latent_pi, action_bounds)
        #distribution  = self.get_distribution_torch(mu)
        #actions_original = distribution.sample()
        distribution = self._get_action_dist_from_mean(mu, latent_pi)
        actions_original = distribution.get_actions(deterministic=deterministic)

        if action_bounds is not None:
            # In the evaluation we clamp the actions, since we do not need the gradient (in comparison to forward pass, where we use the projection layer (cvxpy)
            actions = torch.clamp(actions_original, action_bounds[:, 0:1], action_bounds[:, 1:2])

        #mask_high = actions == action_bounds[:, 1:]
        #mask_low = actions == action_bounds[:, 0:1]
        #mask_middle =  torch.logical_not(mask_high + mask_low)
        #log_prob = mask_high * (torch.log(1 - distribution.cdf(actions))) + mask_low * torch.log(distribution.cdf(actions)) + mask_middle * distribution.log_prob(actions)
        #log_prob = log_prob.squeeze()
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def _get_action_dist_from_mean(self, mean_actions: torch.Tensor, latent_pi) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)#, mean_actions, original_mean
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")


class ClampedActorCriticPolicy(ValidOutputBaseActorCriticPolicy):
    """
    An Actor Critic Policy that maps actions sampled from the normal distribution into the valid action space.

    """
    def __init__(self, *args, **kwargs):
        self.map_action_to_valid_space = map_action_to_valid_space_clamp
        self.construct_optimization_layer = construct_optimization_layer_dummy
        super(ClampedActorCriticPolicy, self).__init__(*args, **kwargs)

class LinearProjectedActorCriticPolicy(ValidOutputBaseActorCriticPolicy):
    """
    An Actor Critic Policy that maps actions sampled from the normal distribution into the valid action space.
    """
    def __init__(self, *args, **kwargs):
        self.map_action_to_valid_space = map_action_to_valid_space_cvxpy_layer
        self.construct_optimization_layer = construct_cvxpy_optimization_layer

        super(LinearProjectedActorCriticPolicy, self).__init__(*args, **kwargs)



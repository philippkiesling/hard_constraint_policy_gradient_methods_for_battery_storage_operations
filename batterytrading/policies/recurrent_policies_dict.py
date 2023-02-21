from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from sb3_contrib.common.recurrent.type_aliases import RNNStates
import torch.nn as nn
from stable_baselines3.common.type_aliases import Schedule
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
#
from batterytrading.policies.mapping_functions import map_action_to_valid_space_cvxpy_layer, \
    construct_cvxpy_optimization_layer, \
    map_action_to_valid_space_clamp, \
    construct_optimization_layer_dummy, \
    map_action_to_valid_space_activationfn
import gym
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from batterytrading.policies.torch_layers import FilterForFeaturesExtractor, ActionNet
import torch as th
class ValidOutputBaseRecurrentActorCriticPolicy(MlpLstmPolicy):
    """
    Recurrent policy class for actor-critic algorithms (has both policy and value prediction).
    To be used with A2C, PPO and the likes.
    It assumes that both the actor and the critic LSTM
    have the same architecture.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic
        (in that case, only the actor gradient is used)
        By default, the actor and the critic have two separate LSTM.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        features_extractor_class = FilterForFeaturesExtractor
        features_extractor_kwargs = {'features':["features"]}
        super(ValidOutputBaseRecurrentActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            lstm_kwargs)

        self.projection_layer = self.construct_optimization_layer(self)
        # Overwrite the default feature dimensions of the LSTM

        self.lstm_kwargs = lstm_kwargs or {}
        self.shared_lstm = shared_lstm
        self.enable_critic_lstm = enable_critic_lstm
        self.lstm_actor = nn.LSTM(
            self.features_dim,
            lstm_hidden_size,
            num_layers=n_lstm_layers,
            **self.lstm_kwargs,
        )
        # For the predict() method, to initialize hidden states
        # (n_lstm_layers, batch_size, lstm_hidden_size)
        self.lstm_hidden_state_shape = (n_lstm_layers, 1, lstm_hidden_size)
        self.critic = None
        self.lstm_critic = None
        assert not (
            self.shared_lstm and self.enable_critic_lstm
        ), "You must choose between shared LSTM, seperate or no LSTM for the critic"

        # No LSTM for the critic, we still need to convert
        # output of features extractor to the correct size
        # (size of the output of the actor lstm)
        if not (self.shared_lstm or self.enable_critic_lstm):
            self.critic = nn.Linear(self.features_dim, lstm_hidden_size)

        # Use a separate LSTM for the critic
        if self.enable_critic_lstm:
            self.lstm_critic = nn.LSTM(
                self.features_dim,
                lstm_hidden_size,
                num_layers=n_lstm_layers,
                **self.lstm_kwargs,
            )
        self.action_net = ActionNet(self.action_net, self.projection_layer, self.map_action_to_valid_space)
        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def construct_optimization_layer(self):
        """
        How to construct the layer mapping model action to a valid space
        Returns:
            torch.layer
        """
        raise NotImplementedError

    def map_action_to_valid_space(self, action, clamp_params):
        pass

    def predict_values(
        self,
        obs: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(features, lstm_states, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Use LSTM from the actor
            latent_pi, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation.
        :param actions:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        action_bounds = obs["action_bounds"]
        latent_pi, _ = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        #if action_bounds == None:
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

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation. Observation
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = obs["features"]
        action_bounds = obs["action_bounds"]
        features = self.extract_features(obs)
        # latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi, lstm_states_pi = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            # Critic only has a feedforward network
            latent_vf = self.critic(features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Evaluate the values for the given observations
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
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf), actions_original, mu, mu_original



    def get_distribution_torch(self, mu):
        #mu = self.action_net(latent_pi)
        sigma_sq = torch.exp(self.log_std)

        from torch.distributions import Normal
        distribution = Normal(mu, sigma_sq.squeeze())
        #distribution = MultivariateNormal(mu, torch.diag(sigma_sq).unsqueeze(0))

        return distribution
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, action_bounds = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        if action_bounds == None:
            mean_actions = self.action_net(latent_pi)
        else:
            mean_actions, original_mean = self.action_net(latent_pi, action_bounds)
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
    def _get_action_dist_from_mean(self, mean_actions: th.Tensor, latent_pi) -> Distribution:
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


class ClampedMlpLstmPolicy(ValidOutputBaseRecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        self.map_action_to_valid_space = map_action_to_valid_space_clamp
        self.construct_optimization_layer = construct_optimization_layer_dummy
        super(ClampedMlpLstmPolicy, self).__init__(*args, **kwargs)

class LinearProjectedMlpLstmPolicy(ValidOutputBaseRecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        self.map_action_to_valid_space = map_action_to_valid_space_cvxpy_layer
        self.construct_optimization_layer = construct_cvxpy_optimization_layer
        super(LinearProjectedMlpLstmPolicy, self).__init__(*args, **kwargs)
        
#class ActivationFunctionProjectedMlpLstmPolicy(ValidOutputBaseRecurrentActorCriticPolicy):
#    def __init__(self, *args, **kwargs):
#        self.map_action_to_valid_space = map_action_to_valid_space_activationtanh
#        self.construct_optimization_layer = construct_optimization_layer_dummy
#
#        super(ActivationFunctionProjectedMlpLstmPolicy, self).__init__(*args, **kwargs)


class ActivationFunctionProjectedMlpLstmPolicy(ValidOutputBaseRecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        self.map_action_to_valid_space = map_action_to_valid_space_activationfn
        self.construct_optimization_layer = construct_optimization_layer_dummy

        super(ActivationFunctionProjectedMlpLstmPolicy, self).__init__(*args, **kwargs)
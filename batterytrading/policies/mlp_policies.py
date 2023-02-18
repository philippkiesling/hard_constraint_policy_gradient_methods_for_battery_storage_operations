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
        self.low = bounds[0]
        self.high = bounds[1]
        self.observation_space = gym.spaces.Box(low=-100, high=1000, shape=(self.features_dim - 2,))
        if pretrain is not None:
            #self.load_state_dict(torch.load(pretrain))
            #self.pretrain_mlp_extractor(**pretrain)
            pretrained_model = ActorCriticPolicy.load("./batterytrading/models/test_model")
            self.action_net = pretrained_model.action_net
            self.value_net = pretrained_model.value_net
            self.mlp_extractor = pretrained_model.mlp_extractor
            self.features_extractor = pretrained_model.features_extractor
        self.standard_forward = super(ValidOutputBaseActorCriticPolicy, self).forward
        self.projection_layer = self.construct_optimization_layer()

    #@abstractmethod
    def construct_optimization_layer(self):
        """
        How to construct the layer mapping model action to a valid space
        Returns:
            torch.layer
        """
        raise NotImplementedError

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
        """
        Map the action sampled from the normal distribution to the valid action space.
        :param action: The action sampled from the normal distribution
        :param clamp_params: The parameters that are used to clamp the action to the valid action space.
        :return: The action mapped to the valid action space.
        """
        # Throw not implemented error since this is an abstract class.
        raise NotImplementedError

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
        valid_check = obs[:, -2:]
        obs = obs[:, :-2]

        actions, values, log_prob = self.standard_forward(obs, deterministic)

        actions = self.map_action_to_valid_space(self, actions, valid_check)
        return actions, values, log_prob

    def pretrain_mlp_extractor(self, env, learnable_env, teacher_policy):
        """
        Pretrain the policy and value networks.
        Get an observation from the environment
        Simulateously sample observations from both env and learnable env. Feed the env's observation to self and the learnable env's observation to teacher policy
        Get the action from the teacher policy
        Get the action from the self
        Calculate the loss between the two actions
        Backpropagate the loss to the self
        Repeat
        Args:
            env:

        Returns:

        """
        # Get an observation from the environment
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        import numpy as np
        print("______________________")
        episode_len = 1
        n_episodes = 100
        teacher_action = teacher_policy.train(episode_len*n_episodes)
        # teacher_action = teacher_action[teacher_policy.planning_horizon:]
        para_list2 = []
        # Throw away the first episode_len actions of env
        #for i in range(episode_len):
        #    env.step(0)
        loss_lst = []
        action_list = []
        n_epochs = 100
        #self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        for x in range(n_epochs):
            env.reset()
            for n_episodes in range(n_episodes):
                print(n_episodes*episode_len, (n_episodes+1)*episode_len)
                episodic_teacher_action = teacher_action[n_episodes*episode_len:(n_episodes+1)*episode_len]
                action_list = []
                #obs = env.step(episodic_teacher_action[0])[0]
                teacher2_action_list = []
                sample_weights = []
                for i in range(episode_len):

                    time_stamp = env.get_current_timestamp()
                    time_stamp = time_stamp.hour * 60 + time_stamp.minute
                    # 3:00 - 4:45 Buy Range
                    if time_stamp >= 3*60 and time_stamp <= 4*60+45:
                        actionT = 0.15
                        w = 20/24
                    # 16:30 - 18:15 Sell Range
                    elif time_stamp>=16*60+30 and time_stamp <= 18*60+15:
                        actionT = -0.15
                        w = 20/24
                    else:
                        actionT = 0.0
                        w = 4/24
                    sample_weights.append(w)
                    obs = env.step(actionT)[0]
                    obs = obs[np.newaxis]
                    obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    actions, _, _ = self(obs, deterministic=True)
                    #if(episodic_teacher_action[i] != 0):
                    #    print("env_time: ", env.get_current_timestamp(), "Teacher action: ", episodic_teacher_action[i], "Student action: ", actions,"observation: ", obs)

                    #actions, values, log_prob = self(obs, deterministic=True)
                    #values.detach(), log_prob.detach(), actions.detach()
                    action_list.append(actions)
                    teacher2_action_list.append(actionT)
                # Calulate the loss between the two actions
                torched_actions = torch.cat(action_list)
                torched_teacher_actions = torch.tensor(teacher2_action_list, dtype = torch.float32)
                sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

                loss_mse = torch.nn.MSELoss()


                loss_mse = (loss_mse(torched_teacher_actions, torched_actions)*sample_weights).mean()
                # Backpropagate the loss to the self
                optimizer.zero_grad()
                loss_mse.backward()
                optimizer.step()
                para_list = []
                for para in self.action_net.parameters():
                    para_list.append(para)
                print(f"pretrainingloss {loss_mse}")
                if env.wandb_log:
                    env.wandb_run.log({"pretrainingloss": loss_mse})
                loss_lst.append(loss_mse.detach().numpy())

        print("______________________")

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



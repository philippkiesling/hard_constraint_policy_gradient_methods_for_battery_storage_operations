import gym
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
from batterytrading.policies.mapping_functions import map_action_to_valid_space_cvxpy_layer, \
    construct_cvxpy_optimization_layer, \
    map_action_to_valid_space_clamp, \
    construct_optimization_layer_dummy, \
    map_action_to_valid_space_activationfn, \
    map_action_to_valid_space_activationtanh
from abc import abstractmethod, ABC
import gym
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy, MultiInputLstmPolicy

class ValidOutputBaseRecurrentActorCriticPolicy(MlpLstmPolicy):
    """
    An Actor Critic Policy that maps actions sampled from the normal distribution into the valid action space.

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
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 10,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
        pretrain=None,
        bounds=(0,0),
        env_reference = None
    ):
        bounds = bounds
        super(ValidOutputBaseRecurrentActorCriticPolicy, self).__init__(
        observation_space = observation_space,
        action_space = action_space,
        lr_schedule = lr_schedule,
        net_arch = net_arch,
        activation_fn = activation_fn,
        ortho_init = ortho_init,
        use_sde = use_sde,
        log_std_init =log_std_init,
        full_std = full_std,
        sde_net_arch = sde_net_arch,
        use_expln = use_expln,
        squash_output = squash_output,
        features_extractor_class = features_extractor_class ,
        features_extractor_kwargs = features_extractor_kwargs,
        normalize_images = normalize_images,
        optimizer_class = optimizer_class,
        optimizer_kwargs = optimizer_kwargs,
        lstm_hidden_size = lstm_hidden_size,
        n_lstm_layers = n_lstm_layers,
        shared_lstm = shared_lstm,
        enable_critic_lstm = enable_critic_lstm,
        lstm_kwargs =lstm_kwargs)

        self.env_reference = env_reference
        self.low = bounds[0]
        self.high = bounds[1]

        self.observation_space = gym.spaces.Box(low=-100, high=1000, shape=(self.features_dim,))
        if pretrain is not None:
            #self.load_state_dict(torch.load(pretrain))
            #self.pretrain_mlp_extractor(**pretrain)
            pretrained_model = ActorCriticPolicy.load("./batterytrading/models/test_model")
            self.action_net = pretrained_model.action_net
            self.value_net = pretrained_model.value_net
            self.mlp_extractor = pretrained_model.mlp_extractor
            self.features_extractor = pretrained_model.features_extractor
        self.standard_forward = super(ValidOutputBaseRecurrentActorCriticPolicy, self).forward
        self.projection_layer = self.construct_optimization_layer(self)
        # Overwrite the default feature dimensions of the LSTM

        self.lstm_kwargs = lstm_kwargs if lstm_kwargs is not None else {}
        self.shared_lstm = shared_lstm
        self.enable_critic_lstm = enable_critic_lstm
        self.lstm_actor = nn.LSTM(
            self.features_dim,  #-2,
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

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.n_steps = 0

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
        obs: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        #features = features[:, :-2]
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
        obs: torch.Tensor,
        actions: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        #features = features[:, :-2]
        latent_pi, _ = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()


    def forward(self, obs: torch.Tensor, lstm_states: RNNStates, episode_starts:torch.tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        #obs = torch.rand(obs.shape)
        bounds = self.env_reference.bounds

        actions, values, log_prob, lstm_states = self.standard_forward(obs, lstm_states, episode_starts, deterministic=False)
        if not deterministic:
            self.n_steps += 1
        #if self.n_steps > 50000:
        actions = self.map_action_to_valid_space(self, actions, bounds)
        return actions, values, log_prob, lstm_states

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
        
class ActivationFunctionProjectedMlpLstmPolicy(ValidOutputBaseRecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        self.map_action_to_valid_space = map_action_to_valid_space_activationtanh
        self.construct_optimization_layer = construct_optimization_layer_dummy

        super(ActivationFunctionProjectedMlpLstmPolicy, self).__init__(*args, **kwargs)
from typing import Generator, Optional, Union

import numpy as np
from gym import spaces
from stable_baselines3.common.vec_env import VecNormalize
from typing import NamedTuple
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
import torch as th
from stable_baselines3.common.type_aliases import TensorDict

class ClampedRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    original_actions: th.Tensor
    original_mu: th.Tensor

class ClampedDictRolloutBufferSamples(ClampedRolloutBufferSamples):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    original_actions: th.Tensor
    original_mu: th.Tensor

class ClampedRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that also stores the LSTM cell and hidden states.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_shape: Shape of the buffer that will collect lstm states
        (n_steps, lstm.num_layers, n_envs, lstm.hidden_size)
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """


    def reset(self):
        super().reset()
        self.original_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.original_mu = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)


    def add(self, *args, orginal_actions, orginal_mu, **kwargs) -> None:
        """
        :param hidden_states: LSTM cell and hidden state
        """

        orginal_actions = orginal_actions.reshape((self.n_envs, self.action_dim))
        orginal_mu = orginal_mu.reshape((self.n_envs, self.action_dim))
        self.original_actions[self.pos] = np.array(orginal_actions).copy()
        self.original_mu[self.pos] = np.array(orginal_mu).copy()
        super().add(*args, **kwargs)
    def get(self, batch_size: Optional[int] = None) -> Generator[ClampedRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def get(self, batch_size: Optional[int] = None) -> Generator[ClampedRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "original_actions",
                "original_mu",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
                    self,
            batch_inds: np.ndarray,
            env: Optional[VecNormalize] = None
    ) -> ClampedRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.original_actions[batch_inds].flatten(),
            self.original_mu[batch_inds].flatten(),
        )
        return ClampedRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class ClampedDictRolloutBuffer(DictRolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space,  device, gae_lambda, gamma, n_envs=n_envs)

    def reset(self):
        """
        Reset the buffer
        Returns:

        """
        self.original_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.original_mu = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        super().reset()


    def add(self, *args,original_actions, original_mu , **kwargs) -> None:
        """
        :param original_actions: original actions sampled from the policy
        :param original_mu: original mu (mean of the gaussian) of the actions
        """
        self.original_actions[self.pos] = np.array(original_actions).copy()
        self.original_mu[self.pos] = np.array(original_mu).copy()
        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[ClampedDictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns", "original_actions", "original_mu"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ClampedDictRolloutBufferSamples:

        return ClampedDictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            original_actions=self.to_torch(self.original_actions[batch_inds].flatten()),
            original_mu=self.to_torch(self.original_mu[batch_inds].flatten()),
        )

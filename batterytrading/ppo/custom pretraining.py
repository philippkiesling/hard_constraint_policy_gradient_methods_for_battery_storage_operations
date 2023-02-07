import torch
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy
from batterytrading.baselines import BaselineModel
from batterytrading.ppo.model_setup_dict import get_config
#from batterytrading.ppo.policies import
import torch as th
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt

# Get Conifguration
model_cfg, train_cfg = get_config("./ppo/cfg.yml")
policy_cfg = model_cfg["policy_kwargs"]
policy_cfg["lr_schedule"] = model_cfg["learning_rate"]
policy_cfg["observation_space"] = model_cfg["env"].observation_space
policy_cfg["action_space"] = model_cfg["env"].action_space

student = MlpLstmPolicy(**policy_cfg)
env = model_cfg["env"]
action = env.action_space.sample()
obs, rewards, done, info = env.step(action)


def pretrain_predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
    """
    Get the policy action from an observation (and optional hidden state).
    Includes sugar-coating to handle different observations (e.g. normalizing images).

    :param observation: the input observation
    :param lstm_states: The last hidden and memory states for the LSTM.
    :param episode_starts: Whether the observations correspond to new episodes
        or not (we reset the lstm states in that case).
    :param deterministic: Whether or not to return deterministic actions.
    :return: the model's action and the next hidden state
        (used in recurrent policies)
    """
    # Switch to eval mode (this affects batch norm / dropout)
    self.set_training_mode(False)

    observation, vectorized_env = self.obs_to_tensor(observation)

    if isinstance(observation, dict):
        n_envs = observation[list(observation.keys())[0]].shape[0]
    else:
        n_envs = observation.shape[0]
    # state : (n_layers, n_envs, dim)
    if state is None:
        # Initialize hidden states to zeros
        state = np.concatenate([np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1)
        state = (state, state)

    if episode_start is None:
        episode_start = np.array([False for _ in range(n_envs)])

    #with th.no_grad():
    # Convert to PyTorch tensors
    states = th.tensor(state[0]).float().to(self.device), th.tensor(state[1]).float().to(self.device)
    episode_starts = th.tensor(episode_start).float().to(self.device)
    actions, states = self._predict(
        observation, lstm_states=states, episode_starts=episode_starts, deterministic=deterministic
    )
    return actions, states

# env of baseline model is deepcopy of env

base_line_model = BaselineModel(env =env,  policy_fn="schedule", n_lowest=35, n_highest=35)
# Optimizer Adam
optimizer = th.optim.Adam(student.parameters(), lr=0.00001)
# MSE loss
criterion = th.nn.MSELoss()
action = env.action_space.sample()

iter = 0
loss_list = []

obs_list = []
base_line_action_list = []
while not done:  # noqa: E712
    timestamp = env.get_current_timestamp()
    obs, rewards, done, info = env.step(env.action_space.sample())

    baseline_action = base_line_model.predict_with_external_env(timestamp)
    #baseline_action = th.tensor(np.array([[baseline_action]]), dtype=th.float32)

    obs_list.append(obs)
    base_line_action_list.append(baseline_action)
    iter += 1
    if iter == 10000:
        done = True

_states = None
# Start training process
bl_action_ls = []
non_bl_action_ls = []
for epoch in range(100):
    for iter in range(len(obs_list)):
        _states = None
        action, _states = pretrain_predict(student, obs,  state=_states, deterministic=True)
        # Calculate loss
        baseline_action = base_line_action_list[iter]
        if baseline_action == 0.15:
            bl_action_ls.append(action.item())
        #    print("baseline_action: {}".format(baseline_action))
        else:
            non_bl_action_ls.append(action.item())
        baseline_action = th.tensor(np.array([[baseline_action]]), dtype=th.float32)

        loss = criterion(baseline_action, action) * (1 + torch.abs(baseline_action)*10)
        # Backpropagation
        loss.backward()
        # Update weights
        optimizer.step()
        # Reset gradients
        optimizer.zero_grad()

        loss_list.append(loss.item())
        if iter % 672 == 0:
            print("Epoch {} Iteration: {} Loss: {}, bl: {}, non-bl {}".format(epoch, iter, np.mean(loss_list[-672:]), np.mean(bl_action_ls[-672:]), np.mean(non_bl_action_ls[-672:])))
            #print("Iteration: {} Loss: {}".format(iter, loss.item()))
            if iter % 6720*5 == 0:
                N = 96
                try:
                    plt.plot(np.convolve(np.array(loss_list[-6720*5:]), np.ones(N) / N, mode='valid'))
                    plt.title("loss_list")
                    #plt.plot(np.array(loss_list).rolling(672).mean())
                    plt.show()

                    plt.plot(np.convolve(np.array(bl_action_ls[-6720*5:]), np.ones(N) / N, mode='valid'))
                    #plt.plot(np.array(loss_list).rolling(672).mean())
                    plt.title("bl_action_ls")
                    plt.show()

                    plt.plot(np.convolve(np.array(non_bl_action_ls[-6720*5:]), np.ones(N) / N, mode='valid'))
                    plt.title("non_bl_action_ls")
                    # plt.plot(np.array(loss_list).rolling(672).mean())
                    plt.show()
                except:
                    pass

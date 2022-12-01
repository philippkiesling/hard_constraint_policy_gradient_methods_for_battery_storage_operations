import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from batterytrading.utils.neural_networks import (
    Control_Net,
    create_input_lstm,
)
print("============================================================================================")
# set device to cpu or cuda
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
elif torch.has_mps:
    device2 = torch.device("mps")

else:
    print("Device set to : cpu")
if torch.has_mps:
    device2 = torch.device("mps")
else:
    device2 = torch.device("cpu")

print("============================================================================================")


def create_input(data):
    """Extract required entries from a data dictionary, stack them and return them as a numpy array"""
    feature_list = ["SOC", "historic_price"]
    return np.hstack([data[feature] for feature in feature_list])


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor -> Calculate next action probabilities
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh(),
            )
            self.actor = Control_Net(input_size=1, hidden_size=2, output_size=4, fc_hidden=12, num_layers=2)
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1),
            )

        # critic -> Value Function
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.critic = Control_Net(input_size=1, hidden_size=2, output_size=4, fc_hidden=12, num_layers=2)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        if self.has_continuous_action_space:
            # self.actor = self.actor.to(device2)
            # state = state.to(device2)
            x_soc, x_price = self.actor.preprocess_input_unbatched(state)

            action_mean = self.actor(x_soc, x_price)
            # action_mean = action_mean.to(device)
            # self.actor = self.actor.to(device)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        if len(state.shape) == 1:
            state = state.unsqueeze(dim=0).T
        if self.has_continuous_action_space:
            x_soc, x_price = self.actor.preprocess_input_batched(state)
            action_mean = self.actor(x_soc, x_price)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        # Add here
        x_soc, x_price = self.actor.preprocess_input_batched(state)

        state_values = self.critic(x_soc, x_price)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std_init=0.6,
    ):
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print(
                    "setting actor output action_std to min_action_std : ",
                    self.action_std,
                )
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")

    def _create_input(self, next_state):
        """
        Create input for the network. This is a helper function for the select_action function.

        Args:
            next_state: next state of the environment (usually a dictionary)

        Returns:
            next_state: next state as numpy dict
        """
        import warnings

        warnings.warn(
            "Warning: Using the default _create_input function,\n "
            "this function horizontally stacks all values of a dict and should "
            "be overwritten depending on the desired input format and data"
        )

        next_state = np.hstack(next_state.values())
        next_state = np.nan_to_num(next_state)
        return next_state

    def select_action(self, state):
        """
        Select an action based on the current state of the environment.

        Args:
            state: current state of the environment (usually a dictionary)

        Returns:

        """
        state = self._create_input(state)
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


if __name__ == "__main__":
    from batterytrading.environment import Environment
    import numpy as np

    has_continuous_action_space = True

    max_ep_len = 400  # max timesteps in one episode
    max_training_timesteps = int(1e5)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 4  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(2e4)  # save model frequency (in num timesteps)
    action_std = 0.6

    update_timestep = 100 * 4  # update policy every n timesteps
    K_epochs = 40  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.01  # learning rate for actor network
    lr_critic = 0.01  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)

    env = Environment()

    env.reset()
    action = np.array([0])
    sum_of_rewards = 0
    state_dim = 1 + 144  # SOC + 144 past timesteps
    action_dim = 1
    action_std_decay_rate = 0.0001
    min_action_std = 0.0001

    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std,
    )

    # ppo_agent._create_input = create_input
    ppo_agent._create_input = create_input_lstm
    next_state, reward, done, info = env.step(action)
    reward_list = []
    action_list = []
    state_list = []
    #  Log the time in timels
    import time

    t0 = time.time()
    timels = []
    # Loop over all the data.
    for timestep in range(1, max_training_timesteps + 1):
        # next_state = [next_state["SOC"]]
        # next_state = np.hstack(next_state.values())
        # next_state = np.nan_to_num(next_state)
        action = ppo_agent.select_action(next_state)
        next_state, reward, done, info = env.step(action)
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)

        sum_of_rewards += reward
        reward_list.append(reward)
        action_list.append(action)
        state_list.append(next_state)

        if timestep % update_timestep == 0:
            print("sum_of_rewards : ", sum_of_rewards)
            # print("action", action)
            # print("next_state : ", next_state)
            timels.append(time.time() - t0)
            ppo_agent.update()
        ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

        if done:
            print(sum_of_rewards)
            break
        # state_dim =

        # action = trader.step(next_state, reward)
        # sum_of_rewards += reward
        # print(sum_of_rewards)
    import matplotlib.pyplot as plt

    plt.plot(reward_list)
    plt.show()

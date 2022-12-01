import time
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from batterytrading.environment import Environment
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback
from collections import OrderedDict

#wandb.init(project='batterytrading', dir = ".log/wandb")
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 250000,
    "env_name": "CartPole-v1",
}

run = wandb.init(project="batterytrading",
                 config=config,
                 sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                 monitor_gym=True,  # auto-upload the videos of agents playing the game
                 save_code=True,  # optional
                 )
env = Environment(time_interval="H", wandb_run=run)

model_cfg = OrderedDict([('batch_size', 128),
             ('clip_range', 0.01),
             ('clip_range_vf', 0.001),
             ('ent_coef', 0.0),
             #("env", "Pendulum-v1"),
             ("env", env),

             ('gae_lambda', 0.95),
             ('gamma', 0.95),
             ('learning_rate', 1e-6),
             ('max_grad_norm', 0.5),
             #('n_envs', 8),
             ('n_epochs', 10),
             ('n_steps', 24*7),
             #('n_timesteps', 400),
             ('policy', 'MlpLstmPolicy'),
             #('policy', 'MlpPolicy'),
             ('policy_kwargs', {
                    'log_std_init': -1.5,
                    'ortho_init': True,
                    #'enable_critic_lstm':False,
                    'activation_fn': nn.GELU}
                    #'lstm_hidden_size=128,
              ),
             ("normalize_advantage", True),
             ('sde_sample_freq', 4),
             ('use_sde', True),
             ('vf_coef', 0.5)
                        ]
             #('normalize_kwargs', {'norm_obs': False, 'norm_reward': False})]
             )

train_cfg = OrderedDict([
    ('callback', WandbCallback(gradient_save_freq=100, model_save_path=f".log/wandb/models/{run.id}")),
    #('normalize', {'norm_obs': True, 'norm_reward': True}),
    ('total_timesteps', 250000),
    ('log_interval', 10)

])


# custom policy: https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html
#env = DummyVecEnv([lambda: BatteryStorageEnv(train, config)])  # train env

TIME = time.strftime("%Y%m%dT%H%M", time.localtime())

model = PPO( verbose = 1, tensorboard_log=".log/log_tb/", **model_cfg)
#model = RecurrentPPO( verbose=1, tensorboard_log=".log/log_tb/", **model_cfg)

# Episodenlänge/ total number of samples to train on
model.learn(
            **train_cfg)

done = False
obs = env.reset()
reward_sum = 0
reward_ls = []
action_ls = []
obs_ls = []
import numpy as np
import matplotlib.pyplot as plt
while not done:  # noqa: E712
    action, _states = model.predict(obs)
    action = action[0]
    obs, rewards, done, info = env.step(action)
    #env.render()
    reward_sum += rewards
    reward_ls.append(rewards)
    action_ls.append(action)
    obs_ls.append(obs)
    if done:  # noqa: E712
        env.close()
print(reward_sum)


print("done")
#folder_path = "{}/{}".format(args.output_path, TIME)
#os.mkdir(folder_path)
#model.save("{}/ppo2_{}".format(folder_path, TIME))
#config.save_json("{}/config_{}.json".format(folder_path, TIME))


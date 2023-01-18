from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from batterytrading.ppo import get_config, ValidOutputBaseActorCriticPolicy, LinearProjectedActorCriticPolicy
from sb3_contrib import ARS
from batterytrading.environment import Environment

# Get Conifguration
model_cfg, train_cfg = get_config("./batterytrading/ars/cfg.yml")


policy_type = model_cfg.pop("policy_type")
model = ARS(**model_cfg)

model.learn(
            **train_cfg)



# Prediction Script
"""
env = model_cfg["env"]
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
"""

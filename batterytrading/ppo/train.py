from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from batterytrading.ppo import RecurrentPPOHardConstraints
#from batterytrading.policies import LinearProjectedActorCriticPolicy, \
#    ClampedMlpLstmPolicy, \
#    ClampedActorCriticPolicy, \
#    LinearProjectedMlpLstmPolicy, \
#    ActivationFunctionProjectedMlpLstmPolicy
from batterytrading.policies.recurrent_policies_dict import ClampedMlpLstmPolicy, LinearProjectedMlpLstmPolicy, ActivationFunctionProjectedMlpLstmPolicy
from batterytrading.ppo.model_setup_dict import get_config
#from batterytrading.ppo.policies import
from stable_baselines3.ppo import MultiInputPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableMultiInputActorCriticPolicy
from sb3_contrib import MaskablePPO
from maskable_recurrent.ppo_mask_recurrent import  MaskableRecurrentActorCriticPolicy
from maskable_recurrent.ppo_mask_recurrent import MaskableRecurrentPPO
# Get Conifguration
model_cfg, train_cfg = get_config("./ppo/cfg.yml")


policy_type = model_cfg.pop("policy_type")
if policy_type == "MlpPolicyMasked":
    #model_cfg["policy"] = MultiInputPolicy
    model_cfg["policy"] = MaskableMultiInputActorCriticPolicy
    #model = PPO( **model_cfg)
    model_cfg.pop("sde_sample_freq")
    model_cfg.pop("use_sde")
    model_cfg["policy_kwargs"].pop("log_std_init")
    model = MaskablePPO(**model_cfg)
elif policy_type == "MlpPolicy":
    model_cfg["policy"] = MultiInputPolicy
    #model_cfg["policy"] = MaskableMultiInputActorCriticPolicy
    model = PPO(**model_cfg)
    print(">>>>>>>> Using  MLP-PPO<<<<<<<<")
elif policy_type == "MlpLstmPolicyMasked":
    model_cfg["policy"] = MaskableRecurrentActorCriticPolicy #"MlpLstmPolicy"
    model = MaskableRecurrentPPO(**model_cfg)
elif policy_type == "MlpLstmPolicy":
    model_cfg["policy"] = "MlpLstmPolicy"
    model_cfg.pop("proj_coef")
    model_cfg.pop("clip_range_proj")
    model = RecurrentPPO(**model_cfg)
    print(">>>>>>>> Using Recurrent-PPO <<<<<<<<")
elif policy_type == "ClampedMlpPolicy":
    model_cfg["policy"] = ClampedActorCriticPolicy
    model = PPO(**model_cfg)
    print(">>>>>>>> Using ClampedMlp-PPO <<<<<<<<")
elif policy_type == "LinearProjectedMlpPolicy":
    model_cfg["policy"] = "MLPActorCriticPolicy"
    model = PPO(**model_cfg)
    print(">>>>>>>> Using LinearProjectedMlp-PPO <<<<<<<<")
elif policy_type == "ClampedMlpLstmPolicy":
    model_cfg["policy"] = ClampedMlpLstmPolicy
    model = RecurrentPPOHardConstraints(**model_cfg)
    print(">>>>>>>> Using ClampedMlpLstmPolicy-PPO <<<<<<<<")
elif policy_type == "ActivationFunctionProjectedMlpLstmPolicy":
    model_cfg["policy"] = ActivationFunctionProjectedMlpLstmPolicy
    model = RecurrentPPOHardConstraints(**model_cfg)
    print(">>>>>>>> Using ActivationFunctionProjectedMlpLstmPolicy-PPO <<<<<<<<")
elif policy_type == "LinearProjectedMlpLstmPolicy":
    model_cfg["policy"] = LinearProjectedMlpLstmPolicy
    model = RecurrentPPOHardConstraints(**model_cfg)
    print(">>>>>>>> Using LinearProjectedMlpLstm-PPO <<<<<<<<")
else:
    raise ValueError(f"Policy {policy_type} not implemented")

# Episodenlänge/ total number of samples to train on
model.learn(
            **train_cfg)


# Prediction Script
"""
env = model_cfg["env"]
done = False
model.eval_env = env

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

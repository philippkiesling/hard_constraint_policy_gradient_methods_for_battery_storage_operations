"""
Functions to create training and model config
"""
from sys import gettrace as sys_get_trace

import yaml
from pathlib import Path
from batterytrading.environment.environment_dict import ContinousEnergyArbitrageEnvironment, DiscreteContinousEnergyArbitrageEnvironment#, NormalizeObservationPartially, RandomSamplePretrainingEnv
import wandb
from wandb.integration.sb3 import WandbCallback
import torch.nn as nn
import gym
import numpy as np
from batterytrading.baselines import BaselineModel
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from batterytrading.wrappers import NormalizeObservationDict
from stable_baselines3.common.callbacks import EvalCallback
from batterytrading.policies.torch_layers import CustomLSTMExtractor
from sb3_contrib.common.maskable.evaluation import evaluate_policy as evaluate_maskable_policy
from maskable_recurrent.common.callbacks import EvalCallbackRecurrentActionMask, EvalCallbackRecurrentActionMask

def get_config(config_path):
    """
    loads config from yaml file
    A config file should contain the following keys:
    - env_config: config for environment
    - model_config: config for model
    - training_config: config for training
    - wandb_config: config for wandb

    Prepares config for training
    Args:
        config_path (str): path to config file

    Returns:
        (config, train_config) (tuple): config for ppo_model and training
    """
    DEBUG = sys_get_trace() is not None
    cfg = yaml.safe_load(Path(config_path).read_text())
    model_config = cfg["model"]
    env_config = cfg["env"]
    train_config = cfg["train"]
    wandb_config = cfg["wandb"]
    config_copy = cfg.copy()
    # create environment
    if wandb_config.pop("use_wandb") and not DEBUG:
        model_save_path = wandb_config.pop("model_save_path")
        run = wandb.init(**wandb_config)
        train_config["callback"] = WandbCallback(gradient_save_freq=10,
                                                 model_save_path=model_save_path + f"/{run.id}")
        wandb.save(config_path)
    else:
        run = None
    # create environment and add it to the model config
    env_config["n_steps"] = model_config["n_steps"]
    if "discrete" in model_config["env"].lower():
        env_config["type"] = DiscreteContinousEnergyArbitrageEnvironment
    else:
        env_config["type"] = ContinousEnergyArbitrageEnvironment

    env_config["wandb_run"] = run

    if env_config["n_envs"] <= -1:
        # raise ValueError("Pretraining with multiple environments is not supported")
        env_config.pop("n_envs")
        env = _get_configured_envOLD(env_config)
    else:
        print(f"Training with multiple environments {env_config['n_envs']}")
        #env = _get_pretrained_env(env_config)
        env, eval_env = _get_configured_env(env_config)

    model_config["env"] = env

    if model_config["pretrain"] == True:
        model_config["pretrain_env"] = _get_pretrained_env(env_config)
    else:
        model_config.pop("pretrain")

    # Convert learning rate to float
    baseLR = float(model_config.pop("lr_base"))
    if model_config["learning_rate"].lower() == "linear":
        model_config["learning_rate"] = lambda x: x * baseLR
    elif model_config["learning_rate"].lower() == "cosine":
        from math import pi
        model_config["learning_rate"] = lambda progress_remaining:  baseLR * (
                    1 + np.cos(pi * (1 - progress_remaining)) / 2)  # lambda f: f * 2.5e-4
    # elif model_config["learning_rate"].lower() == "cosine2":
    else:
        try:
            model_config["learning_rate"] = float(model_config["learning_rate"])
        except ValueError:
            raise ValueError("learning rate must be a float or a string for the schedule with 'linear' or 'cosine'")

    # resolve activation function
    model_config["policy_kwargs"]["activation_fn"] = _resolve_activation_function(
        model_config["policy_kwargs"]["activation_fn"])
    # resolve policy:
    model_config = _resolve_policy(model_config, env_config)
    # setup pretraining DEPRICATED PRETRAINING
    # model_config["policy_kwargs"]["pretrain"] = _setup_pretraining(model_config["policy_kwargs"]["pretrain"], env_config)

    model_config["policy_kwargs"]["features_extractor_class"] = CustomLSTMExtractor
    model_config["policy_kwargs"]["features_extractor_kwargs"] = {'features': ["features"]}
    # The Eval Callback is currently not working with invalid action masking
    #eval_callback = EvalCallbackActionMask(eval_env, eval_freq=672, n_eval_episodes=1, warn=True, deterministic=True, render=False)
    eval_callback = EvalCallbackRecurrentActionMask(eval_env, eval_freq=672, n_eval_episodes=1, warn=True, deterministic=True, render=False)
    #eval_callback = EvalCallback(eval_env, eval_freq=672, n_eval_episodes=1, warn=True, deterministic=True, render=False)

    #eval_callback = EvalCallback(eval_env, eval_freq=672, n_eval_episodes = 1,  warn=True, deterministic=True, render=False)
    eval_callback.evaluate_policy = evaluate_maskable_policy
    train_config["callback"] = eval_callback
    return model_config, train_config


def _resolve_policy(model_config, env_config):
    """
    resolves policy from string. Parses polcy_kwargs in some cases to get correct policy type.
    Args:
        model_config: config for model
        env_config: config for environment (needed for policy_kwargs oflinear projected)

    Returns:

    """

    policy = model_config["policy"].lower()
    if "clampedlstm" in policy:
        model_config["policy_type"] = "ClampedMlpLstmPolicy"
        lstm_kwargs = model_config["policy_kwargs"].pop("lstm_kwargs")
        #model_config["policy_kwargs"]["env_reference"] = model_config["env"]
        for key in lstm_kwargs:
            model_config["policy_kwargs"][key] = lstm_kwargs[key]
        model_config["policy_kwargs"]
        model_config["policy_kwargs"]["bounds"] = (- env_config["max_charge"], env_config["max_charge"])
    elif "activationfunction" in policy:
        model_config["policy_type"] = "ActivationFunctionProjectedMlpLstmPolicy"
        lstm_kwargs = model_config["policy_kwargs"].pop("lstm_kwargs")
        #model_config["policy_kwargs"]["env_reference"] = model_config["env"]
        for key in lstm_kwargs:
            model_config["policy_kwargs"][key] = lstm_kwargs[key]
        model_config["policy_kwargs"]["bounds"] = (- env_config["max_charge"], env_config["max_charge"])

    elif "linearprojectedlstm" in policy:
        model_config["policy_type"] = "LinearProjectedMlpLstmPolicy"
        #model_config["policy_kwargs"]["env_reference"] = model_config["env"]
        lstm_kwargs = model_config["policy_kwargs"].pop("lstm_kwargs")
        for key in lstm_kwargs:
            model_config["policy_kwargs"][key] = lstm_kwargs[key]
            model_config["policy_kwargs"]["bounds"] = (- env_config["max_charge"], env_config["max_charge"])

    elif "lstm" in policy:
        model_config["policy_type"] = "MlpLstmPolicy"
        lstm_kwargs = model_config["policy_kwargs"].pop("lstm_kwargs")
        #model_config.pop("proj_coef")
        for key in lstm_kwargs:
            model_config["policy_kwargs"][key] = lstm_kwargs[key]

    elif "clampedmlp" in policy:
        model_config["policy_type"] = "ClampedMlpPolicy"
        model_config["policy_kwargs"].pop("lstm_kwargs")
        model_config["policy_kwargs"]["bounds"] = (- env_config["max_charge"], env_config["max_charge"])

    elif "linearprojected" in policy:
        model_config["policy_type"] = "LinearProjectedMlpPolicy"
        model_config["policy_kwargs"].pop("lstm_kwargs")
        model_config["policy_kwargs"]["bounds"] = (- env_config["max_charge"], env_config["max_charge"])

    elif "linear" in policy:
        model_config["policy_type"] = "LinearPolicy"
        model_config["policy_kwargs"].pop("lstm_kwargs")
        model_config["policy_kwargs"].pop("activation_fn")
        model_config.pop("proj_coef")

    else:
        model_config["policy_type"] = "MlpPolicy"
        model_config["policy_kwargs"].pop("lstm_kwargs")
        #model_config.pop("proj_coef")

    return model_config


def _resolve_activation_function(activation_fn):
    """
    resolves activation function from string to function
    Args:
        activation_fn: String with name of activation function

    Returns:
        activation_fn: torch activation function
    """
    activation_fn = activation_fn.lower()

    if activation_fn == "relu":
        return nn.ReLU
    elif activation_fn == "tanh":
        return nn.Tanh
    elif activation_fn == "sigmoid":
        return nn.Sigmoid
    elif activation_fn == "leaky_relu":
        return nn.LeakyReLU
    elif activation_fn == "elu":
        return nn.ELU
    elif activation_fn == "selu":
        return nn.SELU
    elif activation_fn == "gelu":
        return nn.GELU
    else:
        raise ValueError("activation function not implemented")


global env_preprocessing
env_preprocessing = None
global first_env
first_env = True

def _get_configured_envOLD(env_config ):
    """
    creates environment with given config
    Args:
        env_config (dict): config for environment
    Returns:
        env (object): environment
    """
    global env_preprocessing
    if env_preprocessing is None:
        env_preprocessing = env_config.pop("preprocessing")
    EnergyArbitrageEnvironment = env_config.pop("type")
    #n_envs = env_config.pop("n_envs")
    env = EnergyArbitrageEnvironment(**env_config)
    env_config["wandb_run"] = None
    # gym.wrappers FilterObservation
    # returns dictionaries as observations
    if isinstance( env.observation_space, gym.spaces.Dict):
        dict_env = True
    else:
        dict_env = False

    if env_preprocessing["clipaction"]:
        env = gym.wrappers.ClipAction(env)
    if env_preprocessing["normalizeobservation"]:
        if dict_env:
            env = NormalizeObservationDict(env)
        else:
            env = gym.wrappers.NormalizeObservation(env)
        # env = NormalizeObservationPartially(env)
    if env_preprocessing["normalizereward"]:
        env = gym.wrappers.NormalizeReward(env)
    if env_preprocessing["transformobservation"]:
        env = gym.wrappers.TransformObservation(env, lambda x: x.flatten())
        # env = gym.wrappers.TransformObservation(env, lambda x: np.clip(x[0].flatten(), -10, 10))
    if env_preprocessing["transformreward"]:
        # env = gym.wrappers.TransformReward(env, lambda x: x.flatten())
        env = gym.wrappers.TransformReward(env, lambda x: np.clip(x, -10, 10))
    if True:
        # IMport action Masker
        from sb3_contrib.common.wrappers import ActionMasker
        env = ActionMasker(env, action_mask_fn= lambda x: x)

    return env


def _get_configured_env(env_config):
    """
    creates environment with given config
    Args:
        env_config (dict): config for environment
    Returns:
        env (object): environment
    """
    n_envs = env_config.pop("n_envs")
    n_steps = env_config.pop("n_steps")
    #envs = [make_env(env_config.copy(), i) for i in range(1, 1 + n_envs)]
    envs = [make_env(env_config.copy(), i) for i in range(0,  n_envs)]
    #envs = [gym.make('CartPole-v1') for i in range(n_envs)]
    #vec_env = DummyVecEnv(envs)


    # In case of macos, we need to set the no_proxy environment variable to avoid a bug in the multiprocessing library
    import os
    os.environ['no_proxy'] = '*'
    if n_envs >1:
        #vec_env = SubprocVecEnv(envs, start_method='fork')
        vec_env = DummyVecEnv(envs)
    else:
        vec_env = envs[0]()
    #vec_env = SubprocVecEnv(envs, start_method='fork')
    env_config["eval_env"] = True
    env_config["n_steps"] = n_steps
    env_config["gaussian_noise"] = False
    env_config["noise_std"] = -1
    eval_env = make_env(env_config.copy(), 0)
    eval_env = eval_env()
    return vec_env, eval_env

def make_env(env_config, env_id):

    return lambda: _get_configured_single_env(env_config, env_id)

def _get_configured_single_env(env_config, env_id):
    """
    creates environment with given config
    Args:
        env_config (dict): config for environment
    Returns:
        env (object): environment
    """
    #global env_preprocessing
    #if env_preprocessing is None:
    #    env_preprocessing = env_config.pop("preprocessing")
    #if env_id >0:
    #    env_config["wandb_run"] = None
    #global first_env
    if env_id > 0:
        env_config["wandb_run"] = None
    env_preprocessing = env_config.pop("preprocessing")

    EnergyArbitrageEnvironment = env_config.pop("type")
    env = EnergyArbitrageEnvironment(**env_config)

    if isinstance( env.observation_space, gym.spaces.Dict):
        dict_env = True
    else:
        dict_env = False
    #env_additional = Environment(**env_config)
    if env_preprocessing["clipaction"]:
        env = gym.wrappers.ClipAction(env)
    if env_preprocessing["normalizeobservation"]:
        if env_preprocessing["normalizeobservation"]:
            if dict_env:
                env = NormalizeObservationDict(env)
            else:
                env = gym.wrappers.NormalizeObservation(env)
        # env = NormalizeObservationPartially(env)
    if env_preprocessing["normalizereward"]:
        env = gym.wrappers.NormalizeReward(env)
    if env_preprocessing["transformobservation"]:
        env = gym.wrappers.TransformObservation(env,
                                lambda x: {"features": x["features"].flatten(),
                                           "action_bounds": x["action_bounds"]}
                                                )
        #env = gym.wrappers.TransformObservation(env,
        #                        lambda x: x["features"].flatten(),
        #                                   #"action_bounds": x["action_bounds"]

        #                                        )        # env = gym.wrappers.TransformObservation(env, lambda x: np.clip(x[0].flatten(), -10, 10))
    if env_preprocessing["transformreward"]:
        # env = gym.wrappers.TransformReward(env, lambda x: x.flatten())
        env = gym.wrappers.TransformReward(env, lambda x: np.clip(x, -10, 10))
    if True:
        # IMport action Masker
        from sb3_contrib.common.wrappers import ActionMasker
        env = ActionMasker(env, action_mask_fn=lambda env: env.valid_action_mask())

    return env

def _get_pretrained_env(env_config):
    """
    creates environment with given config
    Args:
        env_config (dict): config for environment
    Returns:
        env (object): environment
    """
    global env_preprocessing
    if env_preprocessing is None:
        env_preprocessing = env_config.pop("preprocessing")
    env_preprocessing = env_config.pop("preprocessing")
    env = RandomSamplePretrainingEnv(**env_config)
    if env_preprocessing["clipaction"]:
        env = gym.wrappers.ClipAction(env)
    if env_preprocessing["normalizeobservation"]:
        # env = gym.wrappers.NormalizeObservation(env)
        env = NormalizeObservationPartially(env)
    if env_preprocessing["normalizereward"]:
        env = gym.wrappers.NormalizeReward(env)
    if env_preprocessing["transformobservation"]:
        env = gym.wrappers.TransformObservation(env, lambda x: x.flatten())
        # env = gym.wrappers.TransformObservation(env, lambda x: np.clip(x[0].flatten(), -10, 10))
    if env_preprocessing["transformreward"]:
        # env = gym.wrappers.TransformReward(env, lambda x: x.flatten())
        env = gym.wrappers.TransformReward(env, lambda x: np.clip(x, -10, 10))
    return env


def _setup_pretraining(pretrain_config, env_config):
    """
    sets up pretraining configuration of model, returns dict or None

    Args:
        pretrain_config (dict): config for model
    Returns:
        pretrain_config (dict): config for model
    """
    if pretrain_config.pop("pretrain") == False:
        return None
    else:

        env = _get_configured_env(env_config)
        learnable_env_cfg = env_config.copy()
        # learnable_env_cfg["day_ahead_environment"] = True
        learnable_env = EnergyArbitrageEnvironment(**learnable_env_cfg)
        pretrain_config["env"] = env
        pretrain_config["learnable_env"] = learnable_env
        teacher_policy = BaselineModel(learnable_env)
        pretrain_config["teacher_policy"] = teacher_policy
        return pretrain_config

        return pretrain_config


if __name__ == "__main__":
    config = get_config("cfg.yml")
    print(config)

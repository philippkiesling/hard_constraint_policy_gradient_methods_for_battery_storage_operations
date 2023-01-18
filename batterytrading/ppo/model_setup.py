"""
Functions to create training and model config
"""
from sys import gettrace as sys_gettrace

import yaml
from pathlib import Path
from batterytrading.environment import Environment, NormalizeObservationPartially, RandomSamplePretrainingEnv
import wandb
from wandb.integration.sb3 import WandbCallback
import torch.nn as nn
import gym
import numpy as np
from batterytrading.baselines import BaselineModel

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
    DEBUG = sys_gettrace() is not None
    config = yaml.safe_load(Path(config_path).read_text())
    model_config = config["model"]
    env_config = config["env"]
    train_config = config["train"]
    wandb_config = config["wandb"]
    config_copy = config.copy()
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
    if model_config["env"] == "BatteryStorageEnv":
        env_config["wandb_run"] = run
        env = _get_configured_env(env_config)
        model_config["env"] = env
    if model_config["pretrain"] == True:
        model_config["pretrain_env"] = _get_pretrained_env(env_config)
    else:
        model_config.pop("pretrain")

    # Convert learning rate to float
    if model_config["learning_rate"] == "schedule":
        model_config["learning_rate"] = lambda f: f * 2.5e-4
    else:
        try:
            model_config["learning_rate"] = float(model_config["learning_rate"])
        except ValueError:
            raise ValueError("learning rate must be a float or 'schedule'")

    # resolve activation function
    model_config["policy_kwargs"]["activation_fn"] = _resolve_activation_function(model_config["policy_kwargs"]["activation_fn"])
    # resolve policy:
    model_config = _resolve_policy(model_config, env_config)
    # setup pretraining DEPRICATED PRETRAINING
    # model_config["policy_kwargs"]["pretrain"] = _setup_pretraining(model_config["policy_kwargs"]["pretrain"], env_config)

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
        model_config["policy_kwargs"]["env_reference"] = model_config["env"]

        for key in lstm_kwargs:
            model_config["policy_kwargs"][key] = lstm_kwargs[key]
        model_config["policy_kwargs"]
        model_config["policy_kwargs"]["bounds"] = (- env_config["max_charge"], env_config["max_charge"])
    elif "linearprojectedlstm" in policy:
        model_config["policy_type"] = "LinearProjectedMlpLstmPolicy"
        model_config["policy_kwargs"]["env_reference"] = model_config["env"]

        lstm_kwargs = model_config["policy_kwargs"].pop("lstm_kwargs")
        for key in lstm_kwargs:
            model_config["policy_kwargs"][key] = lstm_kwargs[key]
            model_config["policy_kwargs"]
            model_config["policy_kwargs"]["bounds"] = (- env_config["max_charge"], env_config["max_charge"])
    elif "lstm" in policy:
        model_config["policy_type"] = "MlpLstmPolicy"
        lstm_kwargs = model_config["policy_kwargs"].pop("lstm_kwargs")
        for key in lstm_kwargs:
            model_config["policy_kwargs"][key] = lstm_kwargs[key]
        model_config["policy_kwargs"]

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

    else:
        model_config["policy_type"] = "MlpPolicy"
        model_config["policy_kwargs"].pop("lstm_kwargs")

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

def _get_configured_env(env_config):
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

    env = Environment(**env_config)
    if env_preprocessing["clipaction"]:
        env = gym.wrappers.ClipAction(env)
    if env_preprocessing["normalizeobservation"]:
        env = gym.wrappers.NormalizeObservation(env)

        #env = NormalizeObservationPartially(env)
    if env_preprocessing["normalizereward"]:
        env = gym.wrappers.NormalizeReward(env)
    if env_preprocessing["transformobservation"]:
        env = gym.wrappers.TransformObservation(env, lambda x: x.flatten())
        # env = gym.wrappers.TransformObservation(env, lambda x: np.clip(x[0].flatten(), -10, 10))
    if env_preprocessing["transformreward"]:
        #env = gym.wrappers.TransformReward(env, lambda x: x.flatten())
        env = gym.wrappers.TransformReward (env, lambda x: np.clip(x, -10, 10))
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

    env = RandomSamplePretrainingEnv(**env_config)
    if env_preprocessing["clipaction"]:
        env = gym.wrappers.ClipAction(env)
    if env_preprocessing["normalizeobservation"]:
        #env = gym.wrappers.NormalizeObservation(env)
        env = NormalizeObservationPartially(env)
    if env_preprocessing["normalizereward"]:
        env = gym.wrappers.NormalizeReward(env)
    if env_preprocessing["transformobservation"]:
        env = gym.wrappers.TransformObservation(env, lambda x: x.flatten())
        # env = gym.wrappers.TransformObservation(env, lambda x: np.clip(x[0].flatten(), -10, 10))
    if env_preprocessing["transformreward"]:
        #env = gym.wrappers.TransformReward(env, lambda x: x.flatten())
        env = gym.wrappers.TransformReward (env, lambda x: np.clip(x, -10, 10))
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
        #learnable_env_cfg["day_ahead_environment"] = True
        learnable_env = Environment(**learnable_env_cfg)
        pretrain_config["env"] = env
        pretrain_config["learnable_env"] = learnable_env
        teacher_policy = BaselineModel(learnable_env)
        pretrain_config["teacher_policy"] = teacher_policy
        return pretrain_config



        return pretrain_config

if __name__ == "__main__":
    config = get_config("cfg.yml")
    print(config)
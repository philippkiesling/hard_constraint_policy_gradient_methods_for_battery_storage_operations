"""
Functions to create training and model config
"""
from sys import gettrace as sys_gettrace

import yaml
from pathlib import Path
from batterytrading.environment import Environment, NormalizeObservationPartially
import wandb
from wandb.integration.sb3 import WandbCallback
import torch.nn as nn
import gym
import numpy as np

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
        model_config["env"] = _get_configured_env(env_config)

    # Convert learning rate to float
    if model_config["learning_rate"] == "schedule":
        model_config["learning_rate"] = lambda f: f * 3e-4
    else:
        try:
            model_config["learning_rate"] = float(model_config["learning_rate"])
        except ValueError:
            raise ValueError("learning rate must be a float or 'schedule'")

    # resolve activation function
    model_config["policy_kwargs"]["activation_fn"] = _resolve_activation_function(model_config["policy_kwargs"]["activation_fn"])
    if "lstm" in model_config["policy"].lower():
        model_config["policy_type"] = "LSTM"
        lstm_kwargs = model_config["policy_kwargs"].pop("lstm_kwargs")
        for key in lstm_kwargs:
            model_config["policy_kwargs"][key] = lstm_kwargs[key]
        model_config["policy_kwargs"]
    elif "clampedmlp" in model_config["policy"].lower():
        model_config["policy_type"] = "ClampedMlpPolicy"
        model_config["policy_kwargs"].pop("lstm_kwargs")
        model_config["policy_kwargs"]["bounds"] = (- env_config["max_charge"], env_config["max_charge"])
    elif "linearprojected" in model_config["policy"].lower():
        model_config["policy_type"] = "LinearProjectedMlpPolicy"
        model_config["policy_kwargs"].pop("lstm_kwargs")
        model_config["policy_kwargs"]["bounds"] = (- env_config["max_charge"], env_config["max_charge"])
    elif "linear" in model_config["policy"].lower():
        model_config["policy_type"] = "LinearPolicy"
        model_config["policy_kwargs"].pop("lstm_kwargs")
        model_config["policy_kwargs"].pop("activation_fn")
    else:
        model_config["policy_type"] = "MLP"
        model_config["policy_kwargs"].pop("lstm_kwargs")
    # Set action bounds to valid values
    return model_config, train_config

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

def _get_configured_env(env_config):
    """
    creates environment with given config
    Args:
        env_config (dict): config for environment
    Returns:
        env (object): environment
    """
    env_preprocessing = env_config.pop("preprocessing")

    env = Environment(**env_config)
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

if __name__ == "__main__":
    config = get_config("cfg.yml")
    print(config)
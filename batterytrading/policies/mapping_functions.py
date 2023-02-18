import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

# Functions for cvxpylayer
def construct_cvxpy_optimization_layer(self):
    n = 1
    _x = cp.Parameter(1)
    _lower_bound = cp.Parameter(1)
    _upper_bound = cp.Parameter(1)
    _action = cp.Variable(n)
    obj = cp.Minimize(cp.sum_squares(_action - _x))
    cons = [_action >= _lower_bound, _action <= _upper_bound]
    prob = cp.Problem(obj, cons)

    layer = CvxpyLayer(prob, parameters=[_x, _lower_bound, _upper_bound], variables=[_action])

    self.projection_loss = nn.MSELoss(reduction='none')

    return layer


def construct_cvxpy_optimization_layer_improved_version(self):
    """
    This is an improved version of the cvxpy layer. It is more efficient due to the ECOS solver for box_constraints and.
    It is also more stable due to the use of the huber loss function.
    The L1 regularization is also removed.
    Returns:
        torch.nn.Module: CvxpyLayer that maps the action to the valid space
    """
    n = 1
    _x = cp.Parameter(1)
    _lower_bound = cp.Parameter(1)
    _upper_bound = cp.Parameter(1)
    _action = cp.Variable(n)
    obj = cp.Minimize(cp.huber(_action - _x) + 0.1 * cp.norm1(_x))
    cons = [_action >= _lower_bound, _action <= _upper_bound]
    prob = cp.Problem(obj, cons)

    prob.solve(solver=cp.ECOS)

    layer = CvxpyLayer(prob, parameters=[_x, _lower_bound, _upper_bound], variables=[_action])
    self.projection_loss = nn.MSELoss(reduction='none')
    return layer


def map_action_to_valid_space_cvxpy_layer(self, action_original, clamp_params):
    # action = torch.sigmoid(action)
    # Add a small value to the lower bound to avoid numerical issues
    # print("action", action)
    # print("clamp_params", clamp_params)
    # print(clamp_params[0][:1]-1e-2, clamp_params[0][1:2] + 1e-2)
    # print("action", action, "clamp_params", clamp_params)
    try:
        action = action_original

        #action = self.projection_layer(action_original, clamp_params[:, :1] - 1e-3, clamp_params[:, 1:] + 1e-3)[0]
    except:
        print("Mapping Failed", "action", action_original, "clamp_params", clamp_params)
        action = action_original
    #projection_loss = self.projection_loss(action, action_original)

    return action#, action_original


# Functions for Clamping the values
def map_action_to_valid_space_clamp(self, action, clamp_params):
    action_original = action
    #action = torch.clamp(action, action[0][0]-1e-3, clamp_params[0][1] + 1e-3)
    action = torch.clamp(action, clamp_params[:, 0:1], clamp_params[:, 1:2])

    return action

#
def map_action_to_valid_space_activationfn(self, action, clamp_params):
    min_value, max_value = clamp_params[0][0], clamp_params[0][1]
    action = torch.relu(action) + min_value
    action = action / (max_value - min_value)
    action = torch.sigmoid(action) * (max_value - min_value)
    return action

def map_action_to_valid_space_dummy(self, action, clamp_params):
    return action

def map_action_to_valid_space_activationtanh(self, action, clamp_params):
    action_space_low, action_space_high = clamp_params[0][0], clamp_params[0][1]
    # Add a linear layer to the action to allow it to have negative values
    action = self.projection_layer(action)
    action =torch.tanh(action) * (action_space_high - action_space_low) / 2 + (action_space_high + action_space_low) / 2
    return action

def construct_optimization_layer_dummy(self):
    """
    For the clamped version, we don't need to construct an optimization layers since we use torch.clamp
    Returns:

    """
    return torch.nn.Linear(1,1)

import torch as th
# Write a class ClampedDiagGaussianDistribution that inherits from DiagGaussianDistribution
# and overwrite the _get_action_dist_from_latent method
class ClampedDiagGaussianDistribution(DiagGaussianDistribution):
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        mean_actions = self.action_net(latent_pi)
        # Add a linear layer to the action to allow it to have negative values
        mean_actions = self.projection_layer(mean_actions)
        return self.proba_distribution(mean_actions, self.log_std)
    def __init__(self):
        super().__init__()
        #self.projection_layer = torch.nn.Linear(1,1)

    def get_actions(deterministic=False):
        # Add a linear layer to the action to allow it to have negative values
        #mean_actions = self.projection_layer(mean_actions)
        print("WARNING: THIS IS NOT THE CORRECT IMPLMENTATION, SINCE IT DOES NOT CLIP THE ACTION")
        print("FUTURE: IMPLEMENT THIS WITH torch.clamp or cvxpy projection layer")
        return super().get_actions(deterministic=deterministic)

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor) -> "DiagGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        # TODO: QUESTION - Can I create a distirbution here, that is automatically clamped?
        # BENEFIT: Only Change in one spot required(Here) rather than  creating new ppo
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self
def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
    """
    Retrieve action distribution given the latent codes.

    :param latent_pi: Latent code for the actor
    :return: Action distribution
    """
    mean_actions = self.action_net(latent_pi)

    if isinstance(self.action_dist, DiagGaussianDistribution):
        return self.action_dist.proba_distribution(mean_actions, self.log_std)
    elif isinstance(self.action_dist, CategoricalDistribution):
        # Here mean_actions are the logits before the softmax
        return self.action_dist.proba_distribution(action_logits=mean_actions)
    elif isinstance(self.action_dist, MultiCategoricalDistribution):
        # Here mean_actions are the flattened logits
        return self.action_dist.proba_distribution(action_logits=mean_actions)
    elif isinstance(self.action_dist, BernoulliDistribution):
        # Here mean_actions are the logits (before rounding to get the binary actions)
        return self.action_dist.proba_distribution(action_logits=mean_actions)
    elif isinstance(self.action_dist, StateDependentNoiseDistribution):
        return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
    else:
        raise ValueError("Invalid action distribution")

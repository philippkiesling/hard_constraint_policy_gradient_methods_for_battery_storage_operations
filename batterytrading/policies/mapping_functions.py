import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch


# Functions for cvxpylayer
def construct_cvxpy_optimization_layer():
    n = 1
    _x = cp.Parameter(1)
    _lower_bound = cp.Parameter(1)
    _upper_bound = cp.Parameter(1)
    _action = cp.Variable(n)
    obj = cp.Minimize(cp.sum_squares(_action - _x))
    cons = [_action >= _lower_bound, _action <= _upper_bound]
    prob = cp.Problem(obj, cons)

    layer = CvxpyLayer(prob, parameters=[_x, _lower_bound, _upper_bound], variables=[_action])
    return layer


def construct_cvxpy_optimization_layer_improved_version():
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
    return layer


def map_action_to_valid_space_cvxpy_layer(self, action, clamp_params):
    # action = torch.sigmoid(action)
    # Add a small value to the lower bound to avoid numerical issues
    # print("action", action)
    # print("clamp_params", clamp_params)
    # print(clamp_params[0][:1]-1e-2, clamp_params[0][1:2] + 1e-2)
    # print("action", action, "clamp_params", clamp_params)
    action = self.projection_layer(action, clamp_params[0] - 1e-3, clamp_params[1] + 1e-3)[0]
    return action


# Functions for Clamping the values
def map_action_to_valid_space_clamp(self, action, clamp_params):
    action = torch.clamp(action, clamp_params[0][0], clamp_params[1][0])
    return action

#
def map_action_to_valid_space_activationfn(self, action, clamp_params):
    min_value, max_value = clamp_params
    action = torch.relu(action) + min_value
    action = action / (max_value - min_value)
    action = torch.sigmoid(action) * (max_value - min_value)
    return action

def map_action_to_valid_space_activationtanh(self, action, clamp_params):
    action_space_low, action_space_high = clamp_params
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

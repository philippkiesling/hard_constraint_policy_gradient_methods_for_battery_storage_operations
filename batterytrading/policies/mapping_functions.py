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

def map_action_to_valid_space_cvxpy_layer(self, action, clamp_params):
    # action = torch.sigmoid(action)
    # Add a small value to the lower bound to avoid numerical issues
    #print("action", action)
    #print("clamp_params", clamp_params)
    #print(clamp_params[0][:1]-1e-2, clamp_params[0][1:2] + 1e-2)
    action = self.projection_layer(action, clamp_params[0] - 1e-4, clamp_params[1] + 1e-4)[0]
    return action

# Functions for Clamping the values
def map_action_to_valid_space_clamp(self, action, clamp_params):
    action = torch.clamp(action, clamp_params[0][0] +0e10, clamp_params[0][1])
    return action

def construct_optimization_layer_dummy():
    """
    For the clamped version, we don't need to construct an optimization layers since we use torch.clamp
    Returns:

    """
    return None
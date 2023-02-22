from batterytrading.policies.mapping_functions import map_action_to_valid_space_cvxpy_layer,\
    construct_cvxpy_optimization_layer,  \
construct_cvxpy_optimization_layer_improved_version

import torch
import torch.nn as nn

# Create a pytorch model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #self.linear = nn.Linear(1, 1)
        self.projection_layer = construct_cvxpy_optimization_layer(self)

    def forward(self, x):
        #x = self.linear(x)
        return map_action_to_valid_space_cvxpy_layer(self, x, clamp_params = torch.tensor([[-0.15, 0.15]]))

model = Model()
# Construct the cvxpy layer
import numpy as np
# run the code bellow for 30 values for i in -0.3 to 0.3
clamp_params = torch.tensor([[-0.15, 0.15]])
#stacked_params = torch.stack([clamp_params.repeat(1, 1) for _ in range(10)], dim=0)

gradient_list = []
output_list = []
cvxpylayer = model.projection_layer
# define torch clamp
# Define a lambda function to use torch.clamp with 3 input arguments
torch_clamp_layer = lambda x, lower_bound, upper_bound: torch.clamp(x, lower_bound, upper_bound)

# Define a lambda function that uses activation functions to map the action to the valid space with 3 input arguments
# The function should use Tanh
#torch_activation_layer = lambda x, lower_bound, upper_bound: torch.tanh(x) * (upper_bound - lower_bound) / 2 + (upper_bound + lower_bound) / 2

#torch_activation_layer = lambda x, lower_bound, upper_bound: torch.relu(x) + lower_bound

torch_activation_layer  = lambda action, min_value, max_value: torch.sigmoid((torch.relu(action))
                                                                             / (max_value - min_value)) * (max_value - min_value)

layer = cvxpylayer

for i in np.linspace(-0.3, 0.3, 100):
    x = torch.tensor([i], dtype=torch.float32, requires_grad=True)

    out = model.forward(x)
    #out.requires_grad = True  # Set requires_grad=True to enable gradient computation
    y, = layer(x, clamp_params[:, :1] , clamp_params[:, 1:])
    output_list.append(y.detach().numpy())
    y.sum().backward()
    gradient_list.append(x.grad.detach().numpy())
# Create matplotlib figure for output_list


import matplotlib.pyplot as plt
output_list = np.array(output_list).ravel()
plt.xticks([-0.15, 0.15], ["$c_{lower}$", "$c_{upper}$"])
plt.yticks([-0.15, 0.15], ["$c_{lower}$", "$c_{upper}$"])
plt.plot(np.linspace(-0.3, 0.3, 100), output_list, c = '#004b41')
plt.savefig("output.png", dpi = 600)
plt.show()
# Create matplotlib figure for gradient_list
gradient_list = np.array(gradient_list).ravel()
plt.plot(np.linspace(-0.3, 0.3, 100), gradient_list, c = '#004b41')
plt.xticks([-0.15, 0.15], ["$c_{lower}$", "$c_{upper}$"])
plt.yticks([0, 1])
#ax = plt.gca()
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
plt.savefig("gradient.png", dpi = 600)
plt.show()
pass


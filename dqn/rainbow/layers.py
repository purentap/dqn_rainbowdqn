from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class HeadLayer(torch.nn.Module):
    """ Multi-function head layer. Structure of the layer changes depending on
    the "extensions" dictionary. If "noisy" is active, Linear layers become
    Noisy Linear. If "dueling" is active, the dueling architecture must be employed,
    and lastly, if "distributional" is active, output shape should change
    accordingly.

    Args:
        in_size (int): Input size of the head layer
        act_size (int): Action size
        extensions (Dict[str, Any]): A dictionary that keeps extension information
        hidden_size (Optional[int], optional): Size of the hidden layer in Dueling 
        architecture. Defaults to None.

    Raises:
        ValueError: if hidden_size is not given while dueling is active
    """

    def __init__(self, in_size: int, act_size: int, extensions: Dict[str, Any],
                 hidden_size: Optional[int] = None):
        super().__init__()
        self.is_distributional = extensions["distributional"]
        self.is_noisy = extensions["noisy"]
        self.is_dueling = extensions["dueling"]
        if self.is_distributional != False:
            self.n_acts = act_size
            self.n_atoms = extensions["distributional"]["natoms"]
            self.v_min = extensions["distributional"]["vmin"]
            self.v_max = extensions["distributional"]["vmax"]
            self.output_layer = nn.Linear(64, act_size*self.n_atoms)

        if self.is_noisy:
            self.output_layer= NoisyLinear(in_size, act_size, extensions["noisy"]["init_std"])

        if self.is_dueling:
            self.value_layer = nn.Sequential(
                nn.Linear(64,64),
                nn.ReLU(),
                nn.Linear(64,1)
            )
            self.advantage_layer = nn.Sequential(
                nn.Linear(64,64),
                nn.ReLU(),
                nn.Linear(64, act_size)
            )
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """ Run last layer with the given features 

        Args:
            features (torch.Tensor): Input to the layer

        Returns:
            torch.Tensor: Q values or distributions
        """
        if self.is_distributional:
            out = self.output_layer(features)
            out = F.softmax(out.view(-1, self.n_acts, self.n_atoms), dim=-1)
        if self.is_noisy:
            out = self.output_layer(features)
            out = F.softmax(out)
        if self.is_dueling:
            value = self.value_layer(features)
            adv = self.advantage_layer(features)
            q = value + adv - adv.mean(dim=-1, keepdim= True)
            out =q
        return out 

    def reset_noise(self) -> None:
        """ Call reset_noise function of all child layers. Only used when 
        "noisy" is active. """
        for module in self.children():
            module.reset_noise()


class NoisyLinear(torch.nn.Module):
    """ Linear Layer with Noisy parameters. Noise level is set initially and
    kept fixed until "reset_noise" function is called. In training mode,
    noisy layer works stochastically while in evaluation mode it works as a
    standard Linear layer (using mean parameter values).

    Args:
        in_size (int): Input size
        out_size (int): Outout size
        init_std (float): Initial Standard Deviation
    """

    def __init__(self, in_size: int, out_size: int, init_std: float):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.init_std = init_std

        #initialize parameters
        #2 distinct gaussian noise parameters for weight and bias
        self.weight_mu = nn.Parameter(torch.rand(out_size, in_size))
        self.weight_sigma = nn.Parameter(torch.rand(out_size, in_size))
        #self.weight_epsilon = torch.rand(out_size, in_size)

        self.bias_mu = nn.Parameter(torch.rand(out_size)) 
        self.bias_sigma = nn.Parameter(torch.rand(out_size))

        #self.weight_epsilon, self.bias_epsilon = self.reset_noise()
        
        self.reset_noise()
    def reset_noise(self) -> None:
        """ Reset Noise of the parameters"""
        e_i = torch.randn(self.in_size)
        e_j = torch.randn(self.out_size)

        def f_(x):
            return x.sign().mul(x.abs().sqrt())
        f_i = f_(e_i)
        f_j = f_(e_j)
        
        weight_eps = torch.outer(f_i, f_j)
        bias_eps = f_j

        self.weight_epsilon = weight_eps.T #take transpose for broadcasting in forward() shape: (out, in)
        self.bias_epsilon = bias_eps
        #return weight_eps, bias_eps

        #self.weight_epsilon = torch.randn()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward function that works stochastically in training mode
        and deterministically in eval mode.

        Args:
            input (torch.Tensor): Layer input

        Returns:
            torch.Tensor: Layer output
        """
        if self.training:
            out = F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else: 
            out = F.linear(input, self.weight_mu, self.bias_mu)
        return out
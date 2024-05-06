from typing import Dict, Any
from copy import deepcopy
from collections import namedtuple
import numpy as np
import torch
import torch.nn.functional as F
from dqn.replaybuffer.uniform import UniformBuffer, BaseBuffer
from dqn.replaybuffer.prioritized import PriorityBuffer
from dqn.dqn.model import DQN


class RainBow(DQN):
    """ Rainbow DQN agent with selectable extensions.

    Args:
        valuenet (torch.nn.Module): Q network
        nact (int): Number of actions
        extensions (Dict[str, Any]): Extension information
    """

    def __init__(self, valuenet: torch.nn.Module, nact: int, extensions: Dict[str, Any], *buffer_args):

        self.extensions = extensions
        if extensions["prioritized"]:
            buffer = PriorityBuffer(
                *buffer_args,
                alpha=extensions["prioritized"]["alpha"]
            )
        else:
            buffer = UniformBuffer(*buffer_args)
        if extensions["distributional"]:
            self.vmin = extensions["distributional"]["vmin"]
            self.vmax = extensions["distributional"]["vmax"]
            self.natoms = extensions["distributional"]["natoms"]
            self.support = torch.linspace(self.vmin, self.vmax, self.natoms)

        super().__init__(valuenet, nact, buffer)

    def greedy_policy(self, state: torch.Tensor, *args) -> int:
        """ The greedy policy that changes its behavior depending on the
        value of the "distributional" option in the extensions dictionary. If
        distributional values are activated, use expected_value method.

        Args:
            state (torch.Tensor): Torch state

        Returns:
            int: action
        """
        if self.extensions["distributional"]:
            value_dist = self.valuenet(state)
            return self.expected_value(value_dist).argmax().item()
        return super().greedy_policy(state)

    def loss(self, batch: BaseBuffer.Transition, gamma: float) -> torch.Tensor:
        """ Loss method that switches loss function depending on the value
        of the "distributional" option in extensions dictionary. 

        Args:
            batch (BaseBuffer.Transition): Batch of Torch Transitions
            gamma (float): Discount Factor

        Returns:
            torch.Tensor: Value loss
        """
        if self.extensions["distributional"]:
            return self.distributional_loss(batch, gamma)
        return self.vanilla_loss(batch, gamma)

    def vanilla_loss(self, batch: BaseBuffer.Transition, gamma: float) -> torch.Tensor:
        """ MSE (L2, L1, or smooth L1) TD loss with double DQN extension in
        mind. Different than DQN loss, we keep the batch axis to make this
        compatible with the prioritized buffer. Note that: For target value calculation 
        "_next_action_network" should be used. Set target network and action network to
        eval mode while calculating target value if the networks are noisy.

        Args:
            batch (BaseBuffer.Transition): Batch of Torch Transitions
            gamma (float): Discount Factor

        Returns:
            torch.Tensor: Value loss
        """

        states = getattr(batch, "state")
        actions = getattr(batch, "action")
        terminal = getattr(batch, "terminal")

        states = torch.tensor(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(getattr(batch, "reward")).reshape(-1, )
        terminal = torch.tensor(terminal)
        next_states = getattr(batch, "next_state")
        next_states = torch.tensor(next_states)

        valuenet_scores = self.valuenet(states)
        valnet_values = torch.gather(valuenet_scores, 1, actions)

        targetnet_scores = self._next_action_network(next_states)
        targetnet_max_q, _ = torch.max(targetnet_scores, dim=1)
        Q_targets = rewards + (gamma * targetnet_max_q * ((1-terminal).reshape(-1,)))
        Q_expected = valnet_values.reshape(-1,)
        
        loss = (Q_targets - Q_expected).pow(2)
        return loss
        #return super().loss(batch, gamma)



    def expected_value(self, values: torch.Tensor) -> torch.Tensor:
        """ Return the expected state-action values. Used when distributional
            value is activated.

        Args:
            values (torch.Tensor): Value tensor of distributional output (B, A, Z). B,
                A, Z denote batch, action, and atom respectively.

        Returns:
            torch.Tensor: the expected value of shape (B, A)
        """
        expected_state_action_vals = values * self.support
        expected_state_action_vals = torch.sum(values, dim=-1) 
        return expected_state_action_vals        


    def distributional_loss(self, batch: BaseBuffer.Transition, gamma: float) -> torch.Tensor:
        """ Distributional RL TD loss with KL divergence (with Double
        Q-learning via "_next_action_network" at target value calculation).
        Keep the batch axis. Set noisy network to evaluation mode while calculating
        target values.

        Args:
            batch (BaseBuffer.Transition): Batch of Torch Transitions
            gamma (float): Discount Factor

        Returns:
            torch.Tensor: Value loss
        """
        #vmin = self.extensions["distributional"]["vmin"]
        #vmax = self.extensions["distributional"]["vmax"]
        #natoms = self.extensions["distributional"]["natoms"]

        support = self.support

        states = getattr(batch, "state")
        actions = getattr(batch, "action")
        terminal = getattr(batch, "terminal")
        next_states = getattr(batch, "next_state")

        states = torch.tensor(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(getattr(batch, "reward")).reshape(-1, )
        terminal = torch.tensor(terminal)
        next_states = torch.tensor(next_states)
        batch_size = states.shape[0]
        #compute current distribution
        current_dist = self.valuenet(states)
        actions_expanded = actions.unsqueeze(1).expand(current_dist.shape[0], 1 , self.natoms) #expand it to B,1,N_ATOMS
        current_dist = current_dist.gather(1, actions_expanded).squeeze(1)

        dz = (self.vmax - self.vmin) / (self.natoms-1)
        #compute target distribution
        next_dist = self._next_action_network(next_states) * support
        targetnet_max_q_values, next_optimal_actions = torch.max(torch.sum(next_dist, -1), dim= 1)[0],torch.max(torch.sum(next_dist, -1), dim= 1)[1]
        
        next_action_expanded= next_optimal_actions.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2)) #next action dist in the shae of B,1,51 if action is 3 for example its like [[[3,3,3,3,3,3....]]]
        next_dist   = next_dist.gather(1, next_action_expanded).squeeze(1) #we gather the atoms that correnponds to the distribution of next action from the next_dist 
        
        #get projection 
        rewards_expanded = rewards.unsqueeze(1).expand_as(next_dist)
        terminal_expanded = terminal.expand_as(next_dist)
        support_expanded = support.unsqueeze(0).expand_as(next_dist)

        Tz = rewards_expanded + (gamma * support_expanded * (1-terminal_expanded))
        Tz = Tz.clamp(min = self.vmin, max = self.vmax) 

        proj_dist = torch.zeros((batch_size, self.natoms))
        b  = (Tz - self.vmin) / dz
        ml, mu  = b.floor(), b.ceil()

        #there is probably better way to do this using vectorization but i couldnt think of any
        for i in range(batch_size):
            ml_ = ml[i].to(dtype=torch.long)
            mu_ = mu[i].to(dtype=torch.long)
            b_ = b[i].to(dtype=torch.long)
            if terminal[i].item() == 0: #if transition is not terminal
                proj_dist[i][ml_] += next_dist[i] * (mu_ - b_)
                proj_dist[i][mu_] += next_dist[i] * (b_ - ml_)
            else: #if transition is terminal
                proj_dist[i][ml_] += (mu_ - b_)
                proj_dist[i][mu_] += (b_ - ml_)
        
        loss = F.kl_div( current_dist.log(), proj_dist) #input, target
        return loss

    @property
    def _next_action_network(self) -> torch.nn.Module:
        """ Return the network used for the next action calculation (Used for
        Double Q-learning)

        Returns:
            torch.nn.Module: Q network to find target/next action
        """

        if self.extensions["prioritized"] != False or self.extensions["distributional"] != False:
            return self.targetnet
        


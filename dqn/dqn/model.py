import torch
import torch.nn as F
import numpy as np

from dqn.replaybuffer.uniform import UniformBuffer
from dqn.base_dqn import BaseDQN


class DQN(BaseDQN):
    """ Deep Q Network agent.

    Args:
        valuenet (torch.nn.Module): Neural network to estimate Q values
        nact (int):  Number of actions (or outputs)
        buffer (UniformBuffer): Uniform Replay Buffer
    """

    def __init__(self, valuenet: torch.nn.Module, nact: int, buffer: UniformBuffer):
        super().__init__(valuenet, nact)
        self.buffer = buffer

    def push_transition(self, transition: UniformBuffer.Transition) -> None:
        """ Push transition to replay buffer.

        Args:
            transition (UniformBuffer.Transition): One step transition
        """
        self.buffer.push(transition)

    def loss(self, batch: UniformBuffer.Transition, gamma: float) -> torch.Tensor:
        """ TD loss that uses the target network to estimate target values

        Args:
            batch (UniformBuffer.Transition): Batch of transitions
            gamma (float): Discount factor

        Returns:
            torch.Tensor: TD loss
        """
        states = getattr(batch, "state")
        actions = getattr(batch, "action")
        states = torch.tensor(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(getattr(batch, "reward")).reshape(-1, )


        
        next_states = getattr(batch, "next_state")
        next_states = torch.tensor(next_states)

        valuenet_scores = self.valuenet(states)
        valnet_values = torch.gather(valuenet_scores, 1, actions)

        targetnet_scores = self.targetnet(next_states)
        targetnet_max_q, _ = torch.max(targetnet_scores, dim=1)
        Q_targets = rewards +(gamma * targetnet_max_q)
        Q_expected = valnet_values.reshape(-1,)

        td_loss = F.functional.mse_loss(Q_targets, Q_expected)
        return td_loss
        
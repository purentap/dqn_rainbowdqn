""" Vanilla Replay Buffer
"""
from typing import Tuple
from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np

class BaseBuffer(ABC):
    """ Base class for 1-step NumPy based FIFO buffers. 

    Args:
        capacity (int): Maximum size of the buffer
        state_shape (Tuple): Shape of a single the state
        state_dtype (np.dtype): Data type of the states
    """

    Transition = namedtuple("Transition",
                            "state action reward next_state terminal")

    def __init__(self, capacity: int, state_shape: Tuple[int], state_dtype: np.dtype):

        self.capacity = capacity

        if not isinstance(state_shape, (tuple, list)):
            raise ValueError("State shape must be a list or a tuple")

        self.transition_info = self.Transition(
            {"shape": state_shape, "dtype": state_dtype},
            {"shape": (1,), "dtype": np.int64},
            {"shape": (1,), "dtype": np.float32},
            {"shape": state_shape, "dtype": state_dtype},
            {"shape": (1,), "dtype": np.float32},
        )

        self.buffer = self.Transition(
            *(np.zeros((capacity, *x["shape"]), dtype=x["dtype"])
              for x in self.transition_info)
        )

    def __len__(self) -> int:
        """ Capacity of the buffer

        Returns:
            int: Buffer capacity
        """
        return self.capacity

    @abstractmethod
    def push(self, transition: "Transition", *args, **kwargs) -> None:
        """ Push a transition object (with single elements) to the buffer

        Args:
            transition (Transition): transition to push to buffer
        """
        pass

    @abstractmethod
    def sample(self, batchsize: int, *args, **kwargs) -> "Transition":
        """ Sample a batch of transitions

        Args:
            batchsize (int): Batch size

        Returns:
            Transition: Transition object of batch of samples
        """
        pass


class UniformBuffer(BaseBuffer):
    """ Base class for 1-step NumPy based FIFO buffers. 

    Args:
        capacity (int): Maximum size of the buffer
        state_shape (Tuple): Shape of a single the state
        state_dtype (np.dtype): Data type of the states
    """

    def __init__(self, capacity: int, state_shape: Tuple[int], state_dtype: np.dtype):
        super().__init__(capacity, state_shape, state_dtype)
        self.write_index = 0  # pointing the next writing index
        self.size = 0
#"state action reward next_state terminal")
    def push(self, transition: BaseBuffer.Transition) -> None:
        """ Push a transition object (with single element) to the buffer.
        FIFO implementation using <write_index>. <write_index> keeps track of the next
        available index to write. Remember to update <size> attribute as we
        push transitions.

        Args:
            transition (Transition): transition to push to buffer
        """
        
        if self.size == self.capacity:
            self.write_index = 0
            self.size = 0 #not sure abt this
        idx = self.write_index
        
        _, transition  = transition
        self.buffer.state[idx] = transition.state
        self.buffer.action[idx] = transition.action
        self.buffer.reward[idx] = transition.reward
        self.buffer.next_state[idx] = transition.next_state
        self.buffer.terminal[idx] = transition.terminal
        self.write_index += 1
        self.size += 1
        
    def sample(self, batchsize: int, *args, **kwargs) -> BaseBuffer.Transition:
        """ Uniformly sample a batch of transitions from the buffer.

        Args:
            batchsize (int): Batch size

        Returns:
            Transition: Transition object of batch of samples. T(states,
             actions, rewards, terminals, next_states) where "T" is the
             transition namedtuple.
        """
        if batchsize > self.size:
            return None
        
        sample_ids = np.random.choice(self.size, batchsize, replace=False)
        return self.Transition(*[x[sample_ids] for x in self.buffer])


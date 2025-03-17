import random
import numpy as np
from collections import deque, namedtuple

# Define a namedtuple for a single transition.
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "mask"])

class UniformReplay:
    def __init__(self, memory_size, batch_size):
        """
        Initialize a uniform replay buffer.
        
        Args:
            memory_size (int): Maximum number of transitions to store.
            batch_size (int): Number of transitions to sample per batch.
        """
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=memory_size)
    
    def feed(self, data):
        """
        Add transitions to the replay buffer.
        
        Expects `data` to be a dictionary with keys:
            'state', 'action', 'reward', 'next_state', 'mask'
        where each value is a NumPy array with shape (num_agents, ...).
        For each agent, a separate transition is stored.
        """
        # Determine number of transitions in this call (i.e. number of agents)
        # We assume that the first dimension is the number of agents.
        num_transitions = data["state"].shape[0]
        
        for i in range(num_transitions):
            transition = Transition(
                state = data["state"][i],
                action = data["action"][i],
                reward = data["reward"][i],
                next_state = data["next_state"][i],
                mask = data["mask"][i]
            )
            self.buffer.append(transition)
    
    def sample(self):
        """
        Uniformly sample a batch of transitions.
        
        Returns:
            Transition: A namedtuple containing batches (as NumPy arrays) for:
                state, action, reward, next_state, mask.
                Each field is a NumPy array of shape (batch_size, ...).
        """
        batch = random.sample(self.buffer, self.batch_size)
        
        # Gather each field from the sampled transitions.
        states = np.array([transition.state for transition in batch])
        actions = np.array([transition.action for transition in batch])
        rewards = np.array([transition.reward for transition in batch])
        next_states = np.array([transition.next_state for transition in batch])
        masks = np.array([transition.mask for transition in batch])
        
        return Transition(state=states, action=actions, reward=rewards, next_state=next_states, mask=masks)
    
    def size(self):
        """Return the current number of transitions stored in the replay buffer."""
        return len(self.buffer)

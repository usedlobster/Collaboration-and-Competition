"""
Models.py

Implements some utility classes

    ReplayBuffer    -   A replay buffer helper
    OUNoise         -

"""

import torch
import numpy as np
from collections import deque
import random
import copy


class ReplayBuffer:
    """
    A Simple replay buffer , holds config.BUFFER_SIZE samples
    """
    def __init__(self, config):
        self.config = config
        self.memory = deque(maxlen=config.BUFFER_SIZE)  # internal memory (deque)
        self.batch_size = config.BATCH_SIZE

    def add(self, obs):
        # simple just add observation,to a deque
        self.memory.append(obs)

    # idx is the agent number requesting the sample
    def sample(self, idx):
        """ get config.BATCH_SIZE samples """
        # get a sample
        batch = random.sample(self.memory, k=self.config.BATCH_SIZE)

        # safe me writing self.config.device all the time
        dev = self.config.device

        # This only works, as we know there are only 2 agents.
        #
        # we could do something like [idx] + [ x in range( number_agents) if x !=idx ] for other cases !?
        #

        if idx == 0:
            a_order = [0, 1]
        else:
            a_order = [1, 0]

        # this way the agent gets its own states first followed by the opponents
        # without this one agent will perform well, and the other will try to help

        # get the state / action / next_state samples as tensors

        b_s = [torch.from_numpy(np.vstack([b['states'][i] for b in batch])).float().to(dev) for i in a_order]
        b_a = [torch.from_numpy(np.vstack([b['actions'][i] for b in batch])).float().to(dev) for i in a_order]
        b_ns = [torch.from_numpy(np.vstack([b['next_states'][i] for b in batch])).float().to(dev) for i in a_order]

        #
        # get the rewards / dones flag's
        #
        # we didn't need to stack these first - but not changing now - as model loaded.
        #

        b_r = [torch.from_numpy(np.vstack([b['rewards'][i] for b in batch])).float().to(dev) for i in a_order]
        b_d = [torch.from_numpy(np.vstack([b['dones'][i] for b in batch]).astype(np.uint8)).float().to(dev) for i in
               a_order]

        return b_s, b_a, b_r[idx], b_ns, b_d[idx]

    def __len__(self):
        return len(self.memory)


class OUNoise:
    """
        Ornstein-Uhlenbeck process.

        Provide a noise sample to add to the action exploration space.
    """
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.state = copy.copy(self.mu)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state

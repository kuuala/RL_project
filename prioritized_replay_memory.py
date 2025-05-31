from collections import deque

import numpy as np

from transition import Transition


class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.max_priority = 1.0
        self.min_priority = 0.01

    def push(self, *args):
        self.memory.append(Transition(*args)._replace(td_error=self.max_priority))

    def sample(self, batch_size):
        self.beta = min(1.0, self.beta + self.beta_increment)
        priorities = np.array([(abs(experience.td_error) + 1e-5) ** self.alpha for experience in self.memory])
        probabilities = priorities / priorities.sum()
        actual_batch_size = min(batch_size, len(self.memory))
        sample_indices = np.random.choice(
            range(len(self.memory)), size=actual_batch_size, p=probabilities)
        weights = (len(self.memory) * probabilities[sample_indices]) ** (-self.beta)
        weights /= weights.max()
        return [self.memory[i] for i in sample_indices], sample_indices, weights

    def update_td_errors(self, indices, td_errors):
        current_max = 0
        for idx, td_error in zip(indices, td_errors):
            td_error_abs = max(min(abs(td_error), 5.0), self.min_priority)
            self.memory[idx] = self.memory[idx]._replace(td_error=td_error_abs)
            if td_error_abs > current_max:
                current_max = td_error_abs
        if current_max > self.max_priority:
            self.max_priority = current_max

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()

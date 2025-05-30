from collections import deque

import numpy as np

from transition import Transition


class PrioritizedReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        priorities = np.array([abs(experience.td_error) for experience in self.memory])
        sum_priorities = np.sum(priorities)
        if sum_priorities == 0:
            probabilities = np.ones_like(priorities) / len(priorities)
        else:
            probabilities = priorities / sum_priorities
        actual_batch_size = min(batch_size, len(self.memory))
        sample_indices = np.random.choice(
            range(len(self.memory)), size=actual_batch_size, p=probabilities)
        return [self.memory[i] for i in sample_indices], sample_indices

    def update_td_errors(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.memory[idx] = self.memory[idx]._replace(td_error=abs(td_error))

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()

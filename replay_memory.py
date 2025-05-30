import random
from collections import deque

from transition import Transition


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size), None

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()

    def update_td_errors(self, indices, td_errors):
        pass

import math
import random
from enum import Enum
from itertools import count

import gymnasium as gym
import torch
from torch import nn
from torch import optim

from nets import DQN_Net, DuelingDQN_Net, NoisyDQN_Net
from prioritized_replay_memory import PrioritizedReplayMemory
from replay_memory import ReplayMemory
from transition import Transition

BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.5
EPS_END = 0.01
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
REPLAY_MEMORY = 20000


class DQN_ENV(Enum):
    CART_POLE = "CartPole-v1"
    LUNAR_LANDER = "LunarLander-v3"
    MOUNTAIN_CAR = "MountainCar-v0"


class DQN_MODE(Enum):
    BASE = 1
    PRIORITIZED = 2
    DUELING = 3
    NOISY = 4


class DQN:
    def __init__(self, env_name: DQN_ENV, mode_name: DQN_MODE, seed=42):
        self.env = gym.make(env_name.value)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode_name
        self.env_name = env_name
        self.num_episodes = 2000
        torch.manual_seed(seed)
        random.seed(seed)
        self.env.reset(seed=seed)

        if self.mode == DQN_MODE.PRIORITIZED:
            self.memory = PrioritizedReplayMemory(REPLAY_MEMORY)
        else:
            self.memory = ReplayMemory(REPLAY_MEMORY)

        n_actions = self.env.action_space.n
        state, _ = self.env.reset()
        n_observations = len(state)

        if self.mode == DQN_MODE.DUELING:
            net_class = DuelingDQN_Net
        elif self.mode == DQN_MODE.NOISY:
            net_class = NoisyDQN_Net
        elif self.mode == DQN_MODE.BASE or self.mode == DQN_MODE.PRIORITIZED:
            net_class = DQN_Net
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Supported modes are: {list(DQN_MODE)}")

        if self.env_name == DQN_ENV.LUNAR_LANDER:
            hidden_size = 256
        else:
            hidden_size = 128

        self.policy_net = net_class(n_observations, n_actions, hidden_size).to(self.device)
        self.target_net = net_class(n_observations, n_actions, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.steps_done = 0

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions, indices = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        if self.mode == DQN_MODE.PRIORITIZED:
            with torch.no_grad():
                td_errors = (expected_state_action_values.unsqueeze(1) - state_action_values).squeeze().tolist()
                self.memory.update_td_errors(indices, td_errors)

    def select_action(self, state):
        if self.mode == DQN_MODE.NOISY:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def normalize_state(self, state):
        if self.env_name == DQN_ENV.MOUNTAIN_CAR:
            state = (state - torch.tensor(self.env.observation_space.low, device=self.device)) / \
                    (torch.tensor(self.env.observation_space.high, device=self.device) - torch.tensor(
                        self.env.observation_space.low, device=self.device))
        return state

    def train_dqn(self):
        episode_rewards = []
        for _ in range(self.num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            state = self.normalize_state(state).unsqueeze(0)
            total_reward = 0
            for _ in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                total_reward += reward
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device)
                    next_state = self.normalize_state(next_state).unsqueeze(0)

                if self.mode == DQN_MODE.PRIORITIZED:
                    with torch.no_grad():
                        q_value = self.policy_net(state)[0, action.item()]
                        if next_state is not None:
                            next_q_value = self.target_net(next_state).max(1)[0]
                            target = reward + GAMMA * next_q_value
                        else:
                            target = reward
                        td_error = abs((target - q_value).item())
                else:
                    td_error = 1.0

                self.memory.push(state, action, next_state, reward, td_error)
                state = next_state
                self.optimize_model()

                target_net_state_dict = self.target_net.state_dict()
                for target, policy in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    target.data.copy_(TAU * policy.data + (1 - TAU) * target.data)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_rewards.append(total_reward)
                    break

        return episode_rewards

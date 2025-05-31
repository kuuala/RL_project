import math
import random
from enum import Enum
from itertools import count

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch import optim

from nets import DQN_Net, DuelingDQN_Net, NoisyDQN_Net
from prioritized_replay_memory import PrioritizedReplayMemory
from replay_memory import ReplayMemory
from transition import Transition


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
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.005
        self.lr = 1e-4
        self.hidden_size = 256
        self.state_min = torch.tensor(self.env.observation_space.low, device=self.device)
        self.state_max = torch.tensor(self.env.observation_space.high, device=self.device)
        self.training_criteria_length = 50

        if self.env_name == DQN_ENV.MOUNTAIN_CAR:
            self.std_init = 0.1
            self.eps_start = 0.95
            self.eps_end = 0.01
            self.eps_decay = 1500
            self.training_criteria = -110
            self.num_episodes = 1500
            self.replay_memory_size = 20000
            self.beta = 0.4
            self.beta_increment = (1 - self.beta) / (self.num_episodes * 200)
            self.goal_position = 0.5
            self.mid_point = self.state_min[0] + (self.state_max[0] - self.state_min[0]) / 2
        elif self.env_name == DQN_ENV.LUNAR_LANDER:
            self.std_init = 0.1
            self.eps_start = 0.9
            self.eps_end = 0.01
            self.eps_decay = 1500
            self.training_criteria = 200
            self.num_episodes = 500
            self.replay_memory_size = 30000
            self.beta = 0.4
            self.beta_increment = (1 - self.beta) / (self.num_episodes * 200)
        elif self.env_name == DQN_ENV.CART_POLE:
            self.std_init = 0.2
            self.eps_start = 0.5
            self.eps_end = 0.05
            self.eps_decay = 500
            self.training_criteria = 475
            self.num_episodes = 1000
            self.replay_memory_size = 10000
            self.beta = 0.4
            self.beta_increment = (1 - self.beta) / (self.num_episodes * 200)

        torch.manual_seed(seed)
        random.seed(seed)

        if self.mode == DQN_MODE.PRIORITIZED:
            self.memory = PrioritizedReplayMemory(self.replay_memory_size, beta=self.beta,
                                                  beta_increment=self.beta_increment)
        else:
            self.memory = ReplayMemory(self.replay_memory_size)

        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        n_actions = self.env.action_space.n
        state, _ = self.env.reset(seed=seed)
        n_observations = len(state)

        if self.mode == DQN_MODE.DUELING:
            net_class = DuelingDQN_Net
        elif self.mode == DQN_MODE.NOISY:
            net_class = NoisyDQN_Net
            self.lr = 1e-3
        elif self.mode == DQN_MODE.BASE or self.mode == DQN_MODE.PRIORITIZED:
            net_class = DQN_Net
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Supported modes are: {list(DQN_MODE)}")

        if self.mode == DQN_MODE.NOISY:
            self.policy_net = net_class(n_observations, n_actions, self.hidden_size, self.std_init).to(self.device)
            self.target_net = net_class(n_observations, n_actions, self.hidden_size, self.std_init).to(self.device)
        else:
            self.policy_net = net_class(n_observations, n_actions, self.hidden_size).to(self.device)
            self.target_net = net_class(n_observations, n_actions, self.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.steps_done = 0

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions, indices, weights = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if self.mode == DQN_MODE.NOISY:
            self.target_net.sample_noise()
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss(reduction="none")
        individual_losses = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        if self.mode == DQN_MODE.PRIORITIZED:
            weights = torch.tensor(weights, device=self.device, dtype=torch.float32).unsqueeze(1)
            loss = (weights * individual_losses).mean()
        else:
            loss = individual_losses.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if self.mode == DQN_MODE.PRIORITIZED:
            with torch.no_grad():
                td_errors = (expected_state_action_values.unsqueeze(1) - state_action_values).squeeze().abs().tolist()
                self.memory.update_td_errors(indices, td_errors)

    def select_action(self, state):
        if self.mode == DQN_MODE.NOISY:
            self.policy_net.sample_noise()
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def normalize_state(self, state):
        if self.env_name == DQN_ENV.MOUNTAIN_CAR:
            state = (state - self.state_min) / (self.state_max - self.state_min)
        return state

    def get_reward_mountain_car(self, observation):
        if observation[0] >= self.goal_position:
            return 0
        if observation[0] > self.mid_point:
            return (-1 / (self.mid_point - self.goal_position)) * (observation[0] - self.goal_position)
        return -1

    def early_stopping(self, episode_rewards):
        if len(episode_rewards) < self.training_criteria_length:
            return False
        recent_rewards = episode_rewards[-self.training_criteria_length:]
        return np.mean(recent_rewards) >= self.training_criteria

    def train_dqn(self):
        self.policy_net.train()
        self.target_net.train()

        episode_rewards = []
        log_interval = max(1, self.num_episodes // 20)
        for episode in range(1, self.num_episodes + 1):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            state = self.normalize_state(state).unsqueeze(0)
            total_reward = 0
            for _ in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                total_reward += reward
                if self.env_name == DQN_ENV.MOUNTAIN_CAR:
                    reward = self.get_reward_mountain_car(observation)
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device)
                    next_state = self.normalize_state(next_state).unsqueeze(0)

                self.memory.push(state, action, next_state, reward, 1.0)
                state = next_state
                self.optimize_model()

                for target, policy in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    target.data.copy_(self.tau * policy.data + (1 - self.tau) * target.data)

                if done:
                    episode_rewards.append(total_reward)
                    break

            if self.early_stopping(episode_rewards):
                print(
                    f"Early stopping at episode {episode} with average reward {np.mean(episode_rewards[-self.training_criteria_length:]):.2f}")
                break

            if episode % log_interval == 0:
                percent_done = (episode / self.num_episodes) * 100
                avg_reward = np.mean(episode_rewards[-log_interval:])
                print(
                    f"Training progress: {percent_done:.0f}% ({episode}/{self.num_episodes} episodes), average reward {avg_reward:.2f}")

        self.env.close()
        self.memory.clear()
        return episode_rewards

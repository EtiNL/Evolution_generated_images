import torch
import torch.optim as optim
import numpy as np
from model import DQN_CNN
import random

class Agent:
    def __init__(self, input_shape, action_dim, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.model = DQN_CNN(input_shape, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()
        self.action_dim = action_dim
        self.gamma = gamma  # Discount factor

        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        if random.random() < self.epsilon:
            return np.random.rand(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            with torch.no_grad():
                action = self.model(state).numpy().flatten()
            return action

    def store_experience(self, replay_buffer, state, action, reward, next_state, done):
        replay_buffer.add((state, action, reward, next_state, done))

    def train(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return

        experiences = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert lists to tensors efficiently
        states = torch.FloatTensor(np.array(states)).unsqueeze(1)  # Shape: [batch_size, 1, 200, 200]
        actions = torch.FloatTensor(np.array(actions))  # Convert to FloatTensor
        rewards = torch.FloatTensor(np.array(rewards))  # Shape: [batch_size]
        next_states = torch.FloatTensor(np.array(next_states)).unsqueeze(1)  # Shape: [batch_size, 1, 200, 200]
        dones = torch.FloatTensor(np.array(dones))  # Shape: [batch_size]

        q_values = self.model(states)  # Shape: [batch_size, action_dim]
        next_q_values = self.model(next_states)  # Shape: [batch_size, action_dim]

        # Ensure the actions are within the valid range
        actions = actions.clamp(0, self.action_dim - 1)  # Clamp actions to be within valid range
        actions = actions[:, 0]  # Use only the first dimension of actions for indexing

        q_expected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        q_target = rewards + (1 - dones) * self.gamma * next_q_values.max(1)[0]

        loss = self.loss_fn(q_expected, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if random.random() < 0.01:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

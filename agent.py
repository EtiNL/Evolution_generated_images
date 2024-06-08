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
        states = torch.FloatTensor(np.array(states)).unsqueeze(1)
        actions = torch.FloatTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatFloatTensor(np.array(next_states)).unsqueeze(1)
        dones = torch.FloatTensor(np.array(dones))

        print(f"states shape: {states.shape}")
        print(f"actions shape: {actions.shape}")
        print(f"rewards shape: {rewards.shape}")
        print(f"next_states shape: {next_states.shape}")
        print(f"dones shape: {dones.shape}")

        q_values = self.model(states)
        next_q_values = self.model(next_states)

        print(f"q_values shape: {q_values.shape}")
        print(f"next_q_values shape: {next_q_values.shape}")

        q_target = rewards + (1 - dones) * self.gamma * next_q_values.max(1)[0]
        q_expected = q_values.gather(1, actions.argmax(dim=2).unsqueeze(1))

        print(f"q_target shape: {q_target.shape}")
        print(f"q_expected shape: {q_expected.shape}")

        loss = self.loss_fn(q_expected, q_target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if random.random() < 0.01:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
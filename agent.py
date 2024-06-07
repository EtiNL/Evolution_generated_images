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
            # Random action (exploration)
            return np.random.rand(self.action_dim)
        else:
            # Best known action (exploitation)
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            with torch.no_grad():
                action = self.model(state).numpy().flatten()
            return action

    def train(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)
        action = torch.FloatTensor(action).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        done = torch.tensor([done], dtype=torch.float32)

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        # Adjust q_target based on whether the next state is terminal
        q_target = reward + (1 - done) * self.gamma * next_q_values.max(1)[0]
        q_expected = q_values.gather(1, action.argmax().unsqueeze(0).unsqueeze(1))

        loss = self.loss_fn(q_expected, q_target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

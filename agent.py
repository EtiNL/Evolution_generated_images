import torch
import torch.optim as optim
import numpy as np
from model import DQN_CNN

class Agent:
    def __init__(self, input_shape, action_dim, lr=1e-3):
        self.model = DQN_CNN(input_shape, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()
        self.action_dim = action_dim

    def select_action(self, state):
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

        q_target = reward + (1 - done) * next_q_values.max(1)[0]
        q_expected = q_values.gather(1, action.argmax().unsqueeze(0).unsqueeze(1))

        loss = self.loss_fn(q_expected, q_target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

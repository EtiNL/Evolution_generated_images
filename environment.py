import numpy as np
import gym
from gym import spaces
from PIL import Image
from draw_particles import Draw_particules
from score import loss

class CustomEnv(gym.Env):
    def __init__(self, targetImg_path):
        super(CustomEnv, self).__init__()
        
        self.target = np.array(Image.open(targetImg_path)).astype(np.uint8)
        self.toile = np.zeros_like(self.target).astype(np.uint8)
        self.previous_loss = loss(self.target, self.toile)
        
        # Define action and observation space
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.target.shape[0], self.target.shape[1]) , dtype=np.uint8)
        
        self.current_step = 0
    
    def reset(self):
        self.current_step = 0
        self.toile = np.zeros_like(self.target).astype(np.uint8)
        return np.sum(np.abs(self.target - self.toile), axis=2) / np.max(np.abs(self.target - self.toile))
    
    def step(self, action):
        self.current_step += 1
        x_pos, y_pos, radius = action
        x_pos, y_pos, radius = x_pos * self.target.shape[0], y_pos * self.target.shape[1], max(1, radius * min(self.target.shape[:2]) / 2)
        self.toile = Draw_particules(self.target, self.toile, np.array(x_pos), np.array(y_pos), np.array(radius))
        next_state = np.sum(np.abs(self.target - self.toile), axis=2) / np.max(np.abs(self.target - self.toile))
        current_loss = loss(self.target, self.toile)
        reward = self.previous_loss - current_loss
        self.previous_loss = current_loss
        done = current_loss < 10
        return next_state, reward, done, {}
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass

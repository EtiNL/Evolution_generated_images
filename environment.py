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
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.target.shape[0], self.target.shape[1], 3), dtype=np.uint8)
        
        self.current_step = 0
    
    def reset(self):
        self.current_step = 0
        self.toile = np.zeros_like(self.target).astype(np.uint8)
        return np.sum(np.abs(self.target - self.toile), axis=2) / np.max(np.abs(self.target - self.toile))
    
    def step(self, action):
        self.current_step += 1
        x_pos, y_pos, radius = action
        x_pos = np.clip(x_pos * self.target.shape[1], 0, self.target.shape[1] - 1)
        y_pos = np.clip(y_pos * self.target.shape[0], 0, self.target.shape[0] - 1)
        radius = np.clip(max(1, radius * min(self.target.shape[:2]) / 2), 1, min(self.target.shape[:2]) / 2)
        
        x_pos = np.array([x_pos])
        y_pos = np.array([y_pos])
        radius = np.array([radius])

        # Debugging output
        print(f"Step: {self.current_step}, x_pos: {x_pos}, y_pos: {y_pos}, radius: {radius}")
        
        self.toile = Draw_particules(self.target, self.toile, x_pos, y_pos, radius)
        next_state = np.sum(np.abs(self.target - self.toile), axis=2) / np.max(np.abs(self.target - self.toile))
        current_loss = loss(self.target, self.toile)
        reward = self.previous_loss - current_loss
        print("loss: ", current_loss, "loss variation: ", reward,)
        self.previous_loss = current_loss
        done = current_loss < 10
        return next_state, reward, done, {}
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass

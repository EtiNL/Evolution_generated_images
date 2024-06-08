import numpy as np
import gym
from gym import spaces
from PIL import Image
from draw_particles import Draw_particules
from score import loss
import cv2

def load_and_resize_images(img_path, target_size=(200, 200)):
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    return img_array

class CustomEnv(gym.Env):
    def __init__(self, targetImg_path, semaphore):
        super(CustomEnv, self).__init__()
        
        self.semaphore = semaphore
        self.target = load_and_resize_images(targetImg_path)
        
        self.toile = np.zeros_like(self.target).astype(np.uint8)
        self.init_loss = loss(self.target, self.toile, self.semaphore)
        self.previous_loss = self.init_loss
        print(f"{targetImg_path.split('.')} goal loss = ", self.init_loss*0.2)
        
        # Define action and observation space
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.target.shape[0], self.target.shape[1], 3), dtype=np.uint8)
        
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.toile = np.zeros_like(self.target).astype(np.uint8)
        self.previous_loss = loss(self.target, self.toile, self.semaphore)
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

        self.toile = Draw_particules(self.target, self.toile, x_pos, y_pos, radius, self.semaphore)
        next_state = np.sum(np.abs(self.target - self.toile), axis=2) / np.max(np.abs(self.target - self.toile))
        current_loss = loss(self.target, self.toile, self.semaphore)
        reward = self.previous_loss - current_loss

        # Provide intermediate rewards for partial progress
        if current_loss < self.previous_loss:
            reward += 5.0  # Reward for making progress

        # Penalize stagnation
        if current_loss == self.previous_loss:
            reward -= 0.1  # Small penalty for no progress

        self.previous_loss = current_loss
        if self.current_step < 5000:
            done = current_loss/self.init_loss <= 0.2
        else:
            done = False
            reward -= 100
        if done: reward += 100
        return next_state, reward, done, {}
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass

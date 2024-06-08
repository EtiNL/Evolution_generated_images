import numpy as np
import gym
from gym import spaces
from PIL import Image
from draw_particles import Draw_particules
from score import loss
import cv2
import asyncio

def load_and_resize_images(img_path, target_size=(200, 200)):
    print(f"Loading and resizing image: {img_path}")
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    print("Image loaded and resized.")
    return img_array

class CustomEnv(gym.Env):
    def __init__(self, targetImg_path, semaphore):
        print(f"Initializing CustomEnv with image: {targetImg_path}")
        super(CustomEnv, self).__init__()
        
        self.semaphore = semaphore
        self.target = load_and_resize_images(targetImg_path)
        
        self.toile = np.zeros_like(self.target).astype(np.uint8)
        
        # Since we cannot await in __init__, we setup init_loss later
        self.init_loss = None
        self.previous_loss = None
        
        print(f"CustomEnv initialized with image: {targetImg_path}")

    async def setup(self):
        self.init_loss = await loss(self.target, self.toile, self.semaphore)
        self.previous_loss = self.init_loss
        print(f"Goal loss = {self.init_loss*0.2}")

    async def reset(self):
        print("Resetting environment...")
        self.current_step = 0
        self.toile = np.zeros_like(self.target).astype(np.uint8)
        self.previous_loss = await loss(self.target, self.toile, self.semaphore)
        print("Environment reset.")
        return np.sum(np.abs(self.target - self.toile), axis=2) / np.max(np.abs(self.target - self.toile))

    async def step(self, action):
        print(f"Step {self.current_step}, action: {action}")
        self.current_step += 1
        x_pos, y_pos, radius = action
        x_pos = np.clip(x_pos * self.target.shape[1], 0, self.target.shape[1] - 1)
        y_pos = np.clip(y_pos * self.target.shape[0], 0, self.target.shape[0] - 1)
        radius = np.clip(max(1, radius * min(self.target.shape[:2]) / 2), 1, min(self.target.shape[:2]) / 2)

        x_pos = np.array([x_pos])
        y_pos = np.array([y_pos])
        radius = np.array([radius])

        self.toile = await Draw_particules(self.target, self.toile, x_pos, y_pos, radius, self.semaphore)
        next_state = np.sum(np.abs(self.target - self.toile), axis=2) / np.max(np.abs(self.target - self.toile))
        current_loss = await loss(self.target, self.toile, self.semaphore)
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
        print(f"Step {self.current_step} completed, reward: {reward}, done: {done}")
        return next_state, reward, done, {}
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass

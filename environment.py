import numpy as np
import gym
from gym import spaces
from PIL import Image
import cv2
import torch.multiprocessing as mp
import os
import wandb
import torch
import random

def Draw_particules(targetIm, testIm, x_coordinates, y_coordinates, radius, semaphore):
    with semaphore:
        target_tensor = torch.FloatTensor(targetIm)
        test_tensor = torch.FloatTensor(testIm)

        for x, y, r in zip(x_coordinates, y_coordinates, radius):
            y, x, r = int(y), int(x), r
            rr, cc = np.ogrid[:target_tensor.shape[0], :target_tensor.shape[1]]
            circle = (rr - y) ** 2 + (cc - x) ** 2 <= r ** 2
            color = torch.mean(target_tensor[circle], dim=0)
            test_tensor[circle] = color
        return test_tensor.numpy()
            
def loss(targetIm, testIm, semaphore):
    with semaphore:
        target_tensor = torch.FloatTensor(targetIm)
        test_tensor = torch.FloatTensor(testIm)
        loss_val = torch.mean(torch.abs(target_tensor - test_tensor))
    return loss_val.item()

def load_and_resize_images(img_path, target_size=(200, 200)):

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"The image path {img_path} does not exist.")
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    return img_array

class CustomEnv(gym.Env):
    def __init__(self, rank, targetImg_paths, semaphore):
        super(CustomEnv, self).__init__()

        self.semaphore = semaphore
        self.target_paths = targetImg_paths
        self.rank = rank

        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(200, 200), dtype=np.float32)

    def setup(self):
        
        try:
            self.target = load_and_resize_images(random.choice(self.target_paths))
            self.target_size = self.target.shape[0]*self.target.shape[1]
            self.toile = np.zeros_like(self.target).astype(np.uint8)
            self.init_loss = loss(self.target, self.toile, self.semaphore)
            self.previous_loss = self.init_loss
            self.current_step = 0
            
            print(f"Agent {self.rank}:")
            print(f"    target: {self.target_path}")
            print(f"Goal loss = {self.init_loss * 0.1}")
            
            return np.sum(np.abs(self.target - self.toile), axis=2) / np.max(np.abs(self.target - self.toile))
        
        except Exception as e:
            print(f"Exception during setup: {e}")
            wandb.log({"setup_exception": str(e)})
            return None

    def step(self, action):
        self.current_step += 1
        x_pos, y_pos = action
        x_pos = x_pos * self.target.shape[1]
        y_pos = y_pos * self.target.shape[0]
        
        radius = self.find_radius(x_pos, y_pos)

        try:
            self.toile = Draw_particules(self.target, self.toile, np.array([x_pos]), np.array([y_pos]), np.array([radius]), self.semaphore)
            if self.toile is None:
                raise ValueError("Draw_particules returned None")
            next_state = np.sum(np.abs(self.target - self.toile), axis=2) / np.max(np.abs(self.target - self.toile))
            current_loss = loss(self.target, self.toile, self.semaphore)
            absolute_reward = self.previous_loss - current_loss
            proportional_reward = self.target_size*(self.previous_loss - current_loss)/(np.pi*int(radius)**2)
            reward = absolute_reward + proportional_reward
            wandb.log({"proportional reward" : proportional_reward,
                       "rayon": int(radius)})
            self.previous_loss = current_loss
            done = self.current_step >= 10_000 or current_loss <= 0.1 * self.init_loss
            if done:
                reward += 100 if current_loss <= 0.1 * self.init_loss else -100
            return next_state, reward, done, {}
        except Exception as e:
            print(f"Exception during step: {e}")
            wandb.log({"step_exception": str(e)})
            return np.zeros(self.observation_space.shape), 0, True, {}
    
    def find_radius(self, x_pos, y_pos):
        epsilon = 2
        rad_plus = min(self.target.shape[:2]) / 2
        rad_minus = 1
        rad_mid = (rad_plus-rad_minus)/2
        loss_plus = loss(self.target, Draw_particules(self.target, np.copy(self.toile), np.array([x_pos]), np.array([y_pos]), np.array([rad_plus]), self.semaphore), self.semaphore)
        loss_minus = loss(self.target, Draw_particules(self.target, np.copy(self.toile), np.array([x_pos]), np.array([y_pos]), np.array([rad_minus]), self.semaphore), self.semaphore)
        loss_mid = loss(self.target, Draw_particules(self.target, np.copy(self.toile), np.array([x_pos]), np.array([y_pos]), np.array([rad_mid]), self.semaphore), self.semaphore)

        while np.abs(rad_plus-rad_minus)>epsilon:

            if np.min([loss_plus, loss_mid, loss_minus]) == loss_plus:
                swap = rad_mid
                rad_mid += (rad_plus-rad_mid)/2
                rad_minus = swap
                
                toile_minus = Draw_particules(self.target, np.copy(self.toile), np.array([x_pos]), np.array([y_pos]), np.array([rad_minus]), self.semaphore)
                loss_minus = loss(self.target, toile_minus, self.semaphore)
                
                toile_mid = Draw_particules(self.target, np.copy(self.toile), np.array([x_pos]), np.array([y_pos]), np.array([rad_mid]), self.semaphore)
                loss_mid = loss(self.target, toile_mid, self.semaphore)

            elif np.min([loss_plus, loss_mid, loss_minus]) == loss_minus:
                swap = rad_mid
                rad_mid -= (rad_mid-rad_minus)/2
                rad_plus = swap
                
                toile_plus = Draw_particules(self.target, np.copy(self.toile), np.array([x_pos]), np.array([y_pos]), np.array([rad_plus]), self.semaphore)
                loss_plus = loss(self.target, toile_plus, self.semaphore)
                
                toile_mid = Draw_particules(self.target, np.copy(self.toile), np.array([x_pos]), np.array([y_pos]), np.array([rad_mid]), self.semaphore)
                loss_mid = loss(self.target, toile_mid, self.semaphore)
            else:
                rad_minus += (rad_mid-rad_minus)/2
                rad_plus -= (rad_plus-rad_mid)/2
                
                toile_plus = Draw_particules(self.target, np.copy(self.toile), np.array([x_pos]), np.array([y_pos]), np.array([rad_plus]), self.semaphore)
                loss_plus = loss(self.target, toile_plus, self.semaphore)
                
                toile_minus = Draw_particules(self.target, np.copy(self.toile), np.array([x_pos]), np.array([y_pos]), np.array([rad_minus]), self.semaphore)
                loss_minus = loss(self.target, toile_minus, self.semaphore)


        if np.min([loss_plus, loss_mid, loss_minus]) == loss_plus:
            return rad_plus
        
        elif np.min([loss_plus, loss_mid, loss_minus]) == loss_mid:
            return rad_mid
        
        else:
            return rad_minus

    def render(self, mode='human'):
        pass

    def close(self):
        pass
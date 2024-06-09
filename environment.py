import numpy as np
import gym
from gym import spaces
from PIL import Image
from draw_particles import Draw_particules
from score import loss
import cv2
import asyncio
import os
import wandb

def load_and_resize_images(img_path, target_size=(200, 200)):
    print(f"Loading and resizing image: {img_path}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"The image path {img_path} does not exist.")
    img = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB format
    img = img.resize(target_size)
    img_array = np.array(img)
    print("Image loaded and resized.")
    return img_array

class CustomEnv(gym.Env):
    def __init__(self, targetImg_path, semaphore):
        print(f"Initializing CustomEnv with image: {targetImg_path}")
        super(CustomEnv, self).__init__()

        self.semaphore = semaphore
        self.target_path = targetImg_path

        # Gym action and observation spaces
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(200, 200, 3), dtype=np.uint8)

        print("CustomEnv initialization completed.")

    async def setup(self):
        try:
            self.target = load_and_resize_images(self.target_path)
            self.toile = np.zeros_like(self.target).astype(np.uint8)
            self.init_loss = await loss(self.target, self.toile, self.semaphore)
            self.previous_loss = self.init_loss
            print(f"{self.target_path} Goal loss = {self.init_loss * 0.2}")
            self.current_step = 0
            return np.sum(np.abs(self.target - self.toile), axis=2) / np.max(np.abs(self.target - self.toile))
        except Exception as e:
            print(f"Exception during setup: {e}")
            wandb.log({"setup_exception": str(e)})

    async def step(self, action):
        self.current_step += 1
        x_pos, y_pos, radius = action
        x_pos = np.clip(x_pos * self.target.shape[1], 0, self.target.shape[1] - 1)
        y_pos = np.clip(y_pos * self.target.shape[0], 0, self.target.shape[0] - 1)
        radius = np.clip(max(1, radius * min(self.target.shape[:2]) / 2), 1, min(self.target.shape[:2]) / 2)

        x_pos = np.array([x_pos])
        y_pos = np.array([y_pos])
        radius = np.array([radius])

        try:
            self.toile = await Draw_particules(self.target, self.toile, x_pos, y_pos, radius, self.semaphore)
            next_state = np.sum(np.abs(self.target - self.toile), axis=2) / np.max(np.abs(self.target - self.toile))
            current_loss = await loss(self.target, self.toile, self.semaphore)
            reward = self.previous_loss - current_loss

            if current_loss < self.previous_loss:
                reward += 5.0

            if current_loss == self.previous_loss:
                reward -= 0.1

            self.previous_loss = current_loss
            done = self.current_step >= 5000 or current_loss / self.init_loss <= 0.2
            if done:
                reward += 100
            return next_state, reward, done, {}
        except Exception as e:
            print(f"Exception during step: {e}")
            wandb.log({"step_exception": str(e)})
            return None, None, None, None

    def render(self, mode='human'):
        pass

    def close(self):
        pass
    
async def test():
    semaphore = asyncio.Semaphore(1)
    image_path = 'trainning_images/c_sascha_fonseca_wildlife_photographer_of_the_year-taille1200_63877da71854f.jpg'
    env = CustomEnv(image_path, semaphore)
    await env.setup()
    if not hasattr(env, 'target'):
        print(f"Image at {image_path} not found. Exiting.")
        return
    state = await env.reset()
    for _ in range(10):
        action = np.random.rand(3)
        state, reward, done, _ = await env.step(action)
        if done:
            break

if __name__ == "__main__":
    asyncio.run(test())

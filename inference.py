import torch
import numpy as np
from environment import CustomEnv
from model import DQN_CNN

def load_model(model_path, input_shape, output_dim):
    model = DQN_CNN(input_shape, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def make_inference(env, model, num_steps=1000):
    state = env.reset()
    total_reward = 0
    for step in range(num_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        with torch.no_grad():
            action = model(state_tensor).numpy().flatten()

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break
    return total_reward

if __name__ == "__main__":
    model_path = 'final_model.pth'
    target_img_path = ''
    input_shape = (1, 64, 64)  # Update according to your observation space shape
    action_dim = 3  # Update according to your action space dimension

    env = CustomEnv(target_img_path)
    model = load_model(model_path, input_shape, action_dim)

    total_reward = make_inference(env, model, num_steps=1000)
    print(f"Total Reward: {total_reward}")

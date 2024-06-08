# train.py
import numpy as np
import os
import random
from PIL import Image
import torch
import torch.multiprocessing as mp
import argparse
from environment import CustomEnv
from agent import Agent
from replay_buffer import ReplayBuffer
import wandb
from get_dataset import get_images

def train(env, agent, replay_buffer, num_episodes=10, batch_size=32):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_experience(replay_buffer, state, action, reward, next_state, done)
            agent.train(replay_buffer, batch_size)
            state = next_state
            total_reward += reward

        wandb.log({"Episode": episode + 1, "Total Reward": total_reward, "Epsilon": agent.epsilon})

def parallel_train(image_paths, agent, replay_buffer, num_episodes=10, batch_size=32, target_size=(64, 64)):
    if not image_paths:
        print("No images found in the specified directory.")
        return
    
    random_image_path = random.choice(image_paths)
    env = CustomEnv(random_image_path)
    env.target = np.array(Image.open(random_image_path).resize(target_size)).astype(np.uint8)
    train(env, agent, replay_buffer, num_episodes, batch_size)

if __name__ == "__main__":
    training_folder_path = get_images()
    parser = argparse.ArgumentParser(
        prog='Train DQN Agent on Multiple Images',
        description='Train multiple DQN agents on random images from a folder using multiprocessing.',
        epilog='')

    parser.add_argument('--num_agents', type=int, default=4, help='Number of agents to train in parallel')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to train each agent')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--buffer_capacity', type=int, default=10000, help='Capacity of the replay buffer')
    args = parser.parse_args()

    wandb.init(project="DQN-training", config={
        "num_agents": args.num_agents,
        "num_episodes": args.num_episodes,
        "batch_size": args.batch_size,
        "buffer_capacity": args.buffer_capacity
    })

    input_shape = (1, 200, 200)
    action_dim = 3

    replay_buffer = ReplayBuffer(args.buffer_capacity)
    agents = [Agent(input_shape, action_dim) for _ in range(args.num_agents)]

    if not os.path.exists(training_folder_path):
        raise FileNotFoundError(f"Training folder path {training_folder_path} does not exist.")

    image_paths = [os.path.join(training_folder_path, f) for f in os.listdir(training_folder_path) if f.endswith('.jpg') or f.endswith('.png')]

    if not image_paths:
        raise FileNotFoundError(f"No images found in the training folder path {training_folder_path}.")

    processes = []
    for agent in agents:
        p = mp.Process(target=parallel_train, args=(image_paths, agent, replay_buffer, args.num_episodes, args.batch_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    torch.save(agents[0].model.state_dict(), 'final_model.pth')

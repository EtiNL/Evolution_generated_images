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
import pycuda.driver as cuda
import asyncio
import time

async def train(env, agent, replay_buffer, num_episodes=10, batch_size=32):
    await env.setup()  # Ensure the environment is properly set up

    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}")
        state = await env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = await env.step(action)
            agent.store_experience(replay_buffer, state, action, reward, next_state, done)
            agent.train(replay_buffer, batch_size)
            state = next_state
            total_reward += reward

        wandb.log({"Episode": episode + 1, "Total Reward": total_reward, "Epsilon": agent.epsilon})
        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

async def parallel_train(image_paths, agent, replay_buffer, num_episodes=10, batch_size=32, target_size=(64, 64), semaphore=None):
    if not image_paths:
        print("No images found in the specified directory.")
        return
    
    # Initialize CUDA context in each process
    cuda.init()
    cuda_device = cuda.Device(0)
    cuda_context = cuda_device.make_context()

    try:
        await semaphore.acquire()
        try:
            random_image_path = random.choice(image_paths)
            print(f"Selected image: {random_image_path}")
            env = CustomEnv(random_image_path, semaphore)
            env.target = np.array(Image.open(random_image_path).resize(target_size)).astype(np.uint8)
            print(f"Starting training on image: {random_image_path}")
            await train(env, agent, replay_buffer, num_episodes, batch_size)
            print(f"Finished training on image: {random_image_path}")
        finally:
            semaphore.release()
    finally:
        cuda_context.pop()

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Ensure 'spawn' method is used for multiprocessing
    training_folder_path = '/content/Evolution_generated_images/trainning_images'
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

    semaphore = asyncio.Semaphore(1)

    loop = asyncio.get_event_loop()
    tasks = []
    for agent in agents:
        tasks.append(parallel_train(image_paths, agent, replay_buffer, args.num_episodes, args.batch_size, (64, 64), semaphore))
    loop.run_until_complete(asyncio.gather(*tasks))

    torch.save(agents[0].model.state_dict(), 'final_model.pth')

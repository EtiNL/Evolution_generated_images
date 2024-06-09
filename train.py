import numpy as np
import os
import random
from PIL import Image
import torch
import torch.multiprocessing as mp
from environment import CustomEnv
from agent import Agent
from replay_buffer import ReplayBuffer
import wandb
import argparse

def train(rank, env, agent, replay_buffer, num_episodes=10, batch_size=32, semaphore=None):
    wandb.init(project="DQN-training", name=f"agent_{rank}", reinit=True)
    torch.set_num_threads(1)
    for episode in range(num_episodes):
        state = env.setup()
        if state is None:
            wandb.log({"train_status": f"setup_failed_{rank}"})
            continue
        total_reward = 0
        done = False
        step_count = 0
        wandb.log({"train_status": f"train_{rank}"})
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            if next_state is None:
                wandb.log({"train_status": f"step_failed_{rank}"})
                break
            agent.store_experience(replay_buffer, state, action, reward, next_state, done)
            agent.train(replay_buffer, batch_size)
            state = next_state
            total_reward += reward
            step_count += 1
            if step_count % 100 == 0:
                wandb.log({
                    "Agent": rank,
                    "Episode": episode + 1,
                    "Step": step_count,
                    "Step Reward": reward,
                    "Total Reward": total_reward,
                    "Loss": env.previous_loss,
                    "Goal Loss": (env.init_loss * 0.2) * 100,
                    "Epsilon": agent.epsilon
                })

        wandb.log({
            "Agent": rank,
            "Episode": episode + 1,
            "Total Reward": total_reward,
            "Loss": env.previous_loss,
            "Goal Loss": (env.init_loss * 0.2) * 100,
            "Epsilon": agent.epsilon
        })

if __name__ == "__main__":
    mp.set_start_method('spawn')
    training_folder_path = '/content/Evolution_generated_images/trainning_images'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, default=4)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--buffer_capacity', type=int, default=10000)
    args = parser.parse_args()

    wandb.init(project="DQN-training", config={
        "num_agents": args.num_agents,
        "num_episodes": args.num_episodes,
        "batch_size": args.batch_size,
        "buffer_capacity": args.buffer_capacity
    })

    input_shape = (1, 200, 200, 3)
    action_dim = 3

    replay_buffer = ReplayBuffer(args.buffer_capacity)
    agents = [Agent(input_shape, action_dim) for _ in range(args.num_agents)]

    if not os.path.exists(training_folder_path):
        raise FileNotFoundError(f"Training folder path {training_folder_path} does not exist.")

    image_paths = [os.path.join(training_folder_path, f) for f in os.listdir(training_folder_path) if f.endswith('.jpg') or f.endswith('.png')]

    if not image_paths:
        raise FileNotFoundError(f"No images found in the training folder path {training_folder_path}.")

    semaphore = mp.Semaphore(1)

    processes = []
    for rank in range(args.num_agents):
        env = CustomEnv(random.choice(image_paths), semaphore)
        p = mp.Process(target=train, args=(rank, env, agents[rank], replay_buffer, args.num_episodes, args.batch_size, semaphore))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    torch.save(agents[0].model.state_dict(), 'final_model.pth')
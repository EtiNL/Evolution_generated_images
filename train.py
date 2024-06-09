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
import traceback

async def train(env, agent, replay_buffer, num_episodes=10, batch_size=32):
    for episode in range(num_episodes):
        state = await env.setup()
        total_reward = 0
        done = False
        step_count = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = await env.step(action)
            agent.store_experience(replay_buffer, state, action, reward, next_state, done)
            agent.train(replay_buffer, batch_size)
            state = next_state
            total_reward += reward
            step_count += 1
            if step_count % 100 == 0:
                wandb.log({
                    "Episode": episode + 1,
                    "Step": step_count,
                    "Total Reward": total_reward,
                    "Loss": env.previous_loss,
                    "Goal Loss": (env.init_loss * 0.2) * 100,
                    "Epsilon": agent.epsilon
                })

        wandb.log({
            "Episode": episode + 1,
            "Total Reward": total_reward,
            "Loss": env.previous_loss,
            "Goal Loss": (env.init_loss * 0.2) * 100,
            "Epsilon": agent.epsilon
        })

async def parallel_train(image_paths, agent, replay_buffer, num_episodes=10, batch_size=32, target_size=(64, 64), semaphore=None):
    if not image_paths:
        wandb.alert(title="Error", text="No images found in the specified directory.")
        return
    
    try:
        random_image_path = random.choice(image_paths)
        wandb.log({"Selected Image": random_image_path})
        env = CustomEnv(random_image_path, semaphore)
        env.target = np.array(Image.open(random_image_path).resize(target_size)).astype(np.uint8)
        
        wandb.log({"Training Status": "Started"})
        await train(env, agent, replay_buffer, num_episodes, batch_size)
        wandb.log({"Training Status": "Completed"})
    
    except FileNotFoundError as fnf_error:
        wandb.alert(title="FileNotFoundError", text=str(fnf_error))
    except OSError as os_error:
        wandb.alert(title="OSError", text=str(os_error))
    except RuntimeError as runtime_error:
        wandb.alert(title="RuntimeError", text=str(runtime_error))
    except Exception as e:
        wandb.alert(title="Exception", text=str(e))
        traceback.print_exc()

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
    tasks = [parallel_train(image_paths, agent, replay_buffer, args.num_episodes, args.batch_size, (64, 64), semaphore) for agent in agents]
    loop.run_until_complete(asyncio.gather(*tasks))

    torch.save(agents[0].model.state_dict(), 'final_model.pth')

import os
import torch
import torch.multiprocessing as mp
from environment import CustomEnv
from agent import Agent
from replay_buffer import ReplayBuffer
import wandb
import argparse

def train(rank, env, agent, shared_model, target_model, replay_buffer, num_episodes=10, batch_size=32, semaphore=None, target_update_interval=10):
    wandb.init(project="DQN-training", name=f"agent_{rank}")
    torch.set_num_threads(1)
    agent.model = shared_model  # Use the shared model
    agent.target_model = target_model  # Use the shared target model

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
            wandb.log({
                "Episode": episode + 1,
                "Step": step_count,
                "Step Reward": reward,
                "Total Reward": total_reward,
                "Loss": env.previous_loss,
                "Goal Loss": (env.init_loss * 0.1),
                "Epsilon": agent.epsilon
            })
        
        # Update the target model every `target_update_interval` episodes
        if (episode + 1) % target_update_interval == 0:
            target_model.load_state_dict(shared_model.state_dict())
            wandb.log({"target_model_update": episode + 1})

if __name__ == "__main__":
    mp.set_start_method('spawn')
    training_folder_path = '/content/Evolution_generated_images/trainning_images'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, default=4)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--buffer_capacity', type=int, default=10000)
    parser.add_argument('--target_update_interval', type=int, default=10)
    args = parser.parse_args()

    input_shape = (1, 200, 200)
    action_dim = 2

    replay_buffer = ReplayBuffer(args.buffer_capacity)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shared_model = Agent(input_shape, action_dim).model  # Create a shared model
    shared_model.to(torch.device(device))
    target_model = Agent(input_shape, action_dim).model  # Create a target model
    target_model.to(torch.device(device))
    target_model.load_state_dict(shared_model.state_dict())
    target_model.eval()  # Target model is not updated directly

    if not os.path.exists(training_folder_path):
        raise FileNotFoundError(f"Training folder path {training_folder_path} does not exist.")

    image_paths = [os.path.join(training_folder_path, f) for f in os.listdir(training_folder_path) if f.endswith('.jpg') or f.endswith('.png')]

    if not image_paths:
        raise FileNotFoundError(f"No images found in the training folder path {training_folder_path}.")

    semaphore = mp.Semaphore(1)

    processes = []
    for rank in range(args.num_agents):
        env = CustomEnv(rank, image_paths, semaphore)
        agent = Agent(input_shape, action_dim, model=shared_model)
        p = mp.Process(target=train, args=(rank, env, agent, shared_model, target_model, replay_buffer, args.num_episodes, args.batch_size, semaphore, args.target_update_interval))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    torch.save(shared_model.state_dict(), 'final_model.pth')

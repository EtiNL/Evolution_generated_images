import numpy as np
from environment import CustomEnv
from agent import Agent
import argparse

def train(env, agent, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)  # CNN handles 2D state
            next_state, reward, done, _ = env.step(action)
            agent.train(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

if __name__ == "__main__":
    targetImg_path = 'Evolution_generated_images/La_force_des_vagues.JPG'
    parser = argparse.ArgumentParser(
                    prog='',
                    description='',
                    epilog='')
    
    parser.add_argument('filename') 
    args = parser.parse_args()
    env = CustomEnv(args.filename)
    input_shape = (1, env.observation_space.shape[0], env.observation_space.shape[1])
    agent = Agent(input_shape, env.action_space.shape[0])  # Adjust input dimension for CNN
    
    train(env, agent)

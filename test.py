import numpy as np
from env import Ventilator  
from agent import Agent
import gym
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
def main():
    # Initialize the environment
    env = Ventilator()
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    
    n_games = 100
    num_steps = 1000
    evaluate = False
    best_score = float('-inf')
    V_target = 500
    load_checkpoint = False
    if load_checkpoint:
        evaluate = False
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action,V_target)
            observation_ = observation_.squeeze()
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_models()

    V = 0 
    frequency = 0.005  # Breathing rate in Hz (12 breaths per minute)
    angular_frequency = 2 * np.pi * frequency  # Angular frequency
    time_step = 0.001

    for i in range(n_games):
        if i > 80: evaluate = True

        score = 0
        Q_list = []
        V_list = []
        V_target_list = []
        Q_target_list = []
        observation = env.reset()
        for step in range(num_steps):
            if step < 500:
                V_target = 500
                if V_target == 0:
                    observation =  env.reset()
                action = agent.choose_action(observation, evaluate)
                # action = tf.where(V>-1, tf.constant(5.0, shape=(1,)), action)
                # print(action)
                next_state, Q, reward, done, info = env.step(action,V_target)
                # print(Q)
                score += reward
                V = next_state[0]
                # Q_target = (V - V_list[-1])/time_step if step >0 else 0
                Q_target = 300
                # print(Q_target)
                agent.remember(observation, action, reward, next_state, done)
                agent.learn()
                observation = next_state
            else:
                Q_target = 0
                Q = 0

            V_list.append(V) 
            Q_list.append(Q)
            V_target_list.append(V_target) 
            Q_target_list.append(Q_target) 


        print(f'score {i}: ', score)
        if score > best_score:
            best_score = score
            agent.save_models()

        #Plot the Volume
        time = np.arange(0, len(V_list) * time_step, time_step) 

        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(time, Q_list, label='Volume ', color='blue')
        plt.plot(time, Q_target_list, label='Target Volume ', color='green', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (rad/s)')
        plt.title('Speed vs Time')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plot_path = f"./plots/V_{i}.png"
        plt.savefig(plot_path)
        plt.close()
        
if __name__ == "__main__":
    main()

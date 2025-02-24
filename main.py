import numpy as np
from agent import Agent
from env import DCMotor
from utils import plot_learning_curve
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # env = gym.make('Pendulum-v1')
    env = DCMotor()    

    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    n_games = 150

    figure_file = 'plots/pendulum.png'

    best_score = float('-inf')
    score_history = []
    load_checkpoint = True
    evaluate = False
    omega_target = 1
    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset(omega_target)
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action, omega_target)
            observation_ = observation_.squeeze()
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        # evaluate = True
    # else:
        # evaluate = False
    for i in range(n_games):
        if i > 15 :
            evaluate = True
        # if i %20 == 0:
        #     omega_target = np.random.randint(-11,0)

        observation = env.reset(omega_target) 
        done = False
        score = 0
        step = 0
        list_speed = []
        list_u = []
        list_omega_target = []
        while step < 200:
            negative = False
            omega = (step/200) * 2 * np.pi
            omega_target = np.sin(omega)
            if omega_target < 0:
                negative = True
                omega_target = abs(omega_target)
            action = agent.choose_action(observation, evaluate)
            # if omega_target < 0: action = -action
            observation_, reward, done , info = env.step(action, omega_target)
            if negative:
                u = -observation_[0] 
                speed = -observation_[1]
                list_omega_target.append(-omega_target)  # Append target omega

            else: 
                u = observation_[0]
                speed = observation_[1]
                list_omega_target.append(omega_target)  # Append target omega
            # print("action: ",action)
            # print(speed, "and", omega_target)
            score += reward
            list_speed.append(speed)
            list_u.append(u)
            observation_ = observation_.squeeze()
            step += 1    

            agent.remember(observation, action, reward, observation_, done)
            # if not load_checkpoint:
            #     agent.learn()
            # observation = observation_
            agent.learn(step)
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        time_step = 0.01              # Time step
        time = np.arange(0, len(list_speed) * time_step, time_step)  # Time array

        # Plot speed and control input
        plt.figure(figsize=(10, 6))

        # Plot speed
        plt.subplot(2, 1, 1)
        plt.plot(time, list_speed, label='Speed (rad/s)', color='blue')
        plt.plot(time, list_omega_target, label='Target Speed (rad/s)', color='green', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (rad/s)')
        plt.title('Speed vs Time')
        plt.grid(True)
        plt.legend()

        # Plot control input
        plt.subplot(2, 1, 2)
        plt.plot(time, list_u, label='Control Input (Voltage)', color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('Control Input vs Time')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plot_path = f"./plots/speed_and_u_{i}.png"
        plt.savefig(plot_path)
        plt.close()
        if score > best_score:
            best_score = score
            agent.save_models()
        # if avg_score > best_score:
        #     best_score = avg_score
        #     # if not load_checkpoint:
        #     agent.save_models()
        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

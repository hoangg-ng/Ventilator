import numpy as np
import gymnasium as gym 
import matplotlib.pyplot as plt
import os

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

class Ventilator:
    def __init__(self, Q_target, max_steps=500):
        super(Ventilator, self).__init__()
        self.env = gym.make
        self.Q = 0  # Initial state Q
        self.V = 0
        self.Q_target = Q_target  # Target value to reach
        self.max_steps = max_steps  # Maximum steps per episode
        self.current_step = 0
        self.time_step = 0.001
        self.A = np.array([[0, 1], [-2500 / 3, -175 / 3]])
        self.B = np.array([[0], [442500]])
        self.error_prev = 0
        self.prev_error = 0
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)
        self.integral_error = 0
        self.error = 6

    def reset(self):
        """Resets the environment to the initial state."""
        self.Q = np.array([[0.0], [0.0]])
        self.current_step = 0
        return np.array([self.Q[0,0], self.Q_target])  # Return initial state (Q, Q_target)

    def step(self, action):
        """
        Simulates one step in the environment.
        Args:
            action (float): Action taken by the agent (u).
            
        Returns:
            next_state (tuple): The next state (Q, Q_target).
            reward (float): Reward for the action.
            done (bool): Whether the episode is finished.
        """
        self.current_step += 1
        self.V += self.Q[0,0] * self.time_step
        u = action
        self.Q = self.A.dot(self.Q) * self.time_step+ self.B * u * self.time_step+ self.Q
        self.error = self.Q_target - self.Q[0,0]

        # Calculate reward (negative distance from the target Q_target)
        reward = -abs(self.Q_target - self.Q[0,0])

        # Check if the episode is done
        # done = self.current_step >= self.max_steps or np.isclose(self.Q, self.Q_target, atol=1e-2)
        done = self.current_step >= self.max_steps or np.isclose(self.Q[0, 0], self.Q_target, atol=1e-2)


        # Return the next state, reward, and done
        next_state = np.array([self.Q[0,0], self.Q_target])
        return next_state, reward, done

# Q-learning agent
class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_space = state_space  # State space size
        self.action_space = action_space  # Action space size
        self.lr = learning_rate  # Learning rate
        self.gamma = discount_factor  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Exploration decay rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate

        # Initialize Q-table (discretized states and actions)
        self.q_table = np.zeros((*state_space, action_space))

    def get_discrete_state(self, state):
        """Discretizes the continuous state, clamping values to valid bounds."""
        Q, Q_target = state
        # breakpoint()
        Q_bin = np.clip(int((Q + 10) * 10), 0, self.state_space[0] - 1)  # Clamp between 0 and max index
        Q_target_bin = np.clip(int((Q_target + 10) * 10), 0, self.state_space[1] - 1)  # Clamp between 0 and max index
        return (Q_bin, Q_target_bin)

    def choose_action(self, state):
        """Chooses an action using epsilon-greedy strategy."""
        discrete_state = self.get_discrete_state(state)
        if np.random.rand() < self.epsilon:
            # Explore: Random action
            return np.random.randint(self.action_space)
        else:
            # Exploit: Best action from Q-table
            return np.argmax(self.q_table[discrete_state])

    def update_q_table(self, state, action, reward, next_state, done):
        """Updates the Q-table using the Q-learning update rule."""
        discrete_state = self.get_discrete_state(state)
        discrete_next_state = self.get_discrete_state(next_state)

        # Q-value update rule
        current_q = self.q_table[discrete_state][action]
        max_future_q = np.max(self.q_table[discrete_next_state]) if not done else 0
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)

        # Update Q-table
        self.q_table[discrete_state][action] = new_q

        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

def evaluate_agent(agent, Q_target, num_episodes=10):
    """
    Evaluates the trained agent on the environment.
    
    Args:
        agent (QLearningAgent): Trained Q-learning agent.
        Q_target (float): Target value to reach.
        num_episodes (int): Number of episodes to run for evaluation.

    Returns:
        dict: Evaluation results including average cumulative reward, 
            average steps to convergence, and average final error.
    """
    env = Ventilator(Q_target)
    total_rewards = []
    steps_to_converge = []
    final_errors = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            # Choose the best action (exploitation only, no exploration during evaluation)
            discrete_state = agent.get_discrete_state(state)
            action_idx = np.argmax(agent.q_table[discrete_state])
            action = (action_idx - 10) / 10.0  # Map action index to continuous action (-1.0 to 1.0)

            # Step in the environment
            next_state, reward, done = env.step(action)

            # Accumulate reward and steps
            episode_reward += reward
            steps += 1

            # Move to the next state
            state = next_state

        # Store evaluation metrics
        total_rewards.append(episode_reward)
        steps_to_converge.append(steps)  # Steps taken in this episode
        final_errors.append(abs(env.Q - Q_target))  # Final error

    # Calculate average metrics
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(steps_to_converge)
    avg_final_error = np.mean(final_errors)

    return {
        "Average Cumulative Reward": avg_reward,
        "Average Steps to Converge": avg_steps,
        "Average Final Error": avg_final_error,
    }


# Main training loop

def train_agent(Q_target, episodes=1000):
    # Define environment and agent
    env = Ventilator(Q_target)
    state_space = (200, 200)  # Discretized state space size
    action_space = 21  # Discretized actions (-1.0 to 1.0 in steps of 0.1)
    agent = QLearningAgent(state_space, action_space)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        q_values_per_episode = []  # List to store Q[0, 0] for each step

        for step in range(env.max_steps):
            # Choose an action
            action_idx = agent.choose_action(state)
            action = (action_idx - 10) / 10.0  # Map action index to continuous value (-1.0 to 1.0)

            # Take a step in the environment
            next_state, reward, done = env.step(action)

            # Update Q-table
            agent.update_q_table(state, action_idx, reward, next_state, done)

            # Update state and accumulate reward
            state = next_state
            total_reward += reward
            
            # Store the value of Q[0, 0] for this step
            q_values_per_episode.append(env.Q[0, 0])

            if done:
                break

        # Plotting Q[0, 0] for this episode
        if episode > 900:
            plt.figure(figsize=(10, 5))
            plt.plot(q_values_per_episode, label='Q[0,0] over Steps')
            plt.xlabel('Steps')
            plt.ylabel('Q[0,0]')
            plt.title(f'Episode {episode + 1}: Q[0,0] Progression')
            plt.legend()
            plt.grid()

            # Save the plot for this episode
            plt.savefig(os.path.join(output_dir, f'episode_{episode + 1}.png'))
            plt.close()  # Close the plot to free memory

        # Print progress
        print(f"Episode {episode + 1}/{episodes}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

    print("Training complete!")
    return agent

trained_agent = train_agent(Q_target=500)


evaluation_results = evaluate_agent(trained_agent, Q_target=10, num_episodes=10)
print("Evaluation Results:")
for metric, value in evaluation_results.items():
    print(f"{metric}: {value:.2f}")
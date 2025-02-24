import gymnasium as gym 
import numpy as np
import math

class Ventilator(gym.Env):
    def __init__(self):
        super(Ventilator, self).__init__()
        self.env = gym.make
        # self.R = 5    # Sức cản đường thở (cmH2O·s/L)
        # self.C = 10  # Độ giãn nở (mL/cmH2O)
        # self.PEEP = 5 # Áp suất cuối kỳ (cmH2O)
        self.time_step = 0.001
        self.A = np.array([[0, 1], [-2500 / 3, -175 / 3]])
        self.B = np.array([[0], [442500]])
        self.error_prev = 0
        self.prev_error = 0
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)
        self.integral_error = 0
        self.error = 6
        # self.total_episodes = 200
        self.reset()


    def get_optimal_action(self, Q_target, Q_t):
        B_Q = 442500
        A_Q = -2500 / 3

        # Compute the proportional error
        error = Q_target - Q_t

        # PID Controller Gains (Needs tuning)
        Kp = 0.06  # Proportional gain (increased)
        Ki = 0.003  # Integral gain (newly added)
        Kd = 0.01  # Derivative gain (fine-tuned)

        self.integral_error = self.integral_error + error * self.time_step
        derivative = (error - self.prev_error) / self.time_step
        optimal_u = Kp * error + Ki * self.integral_error + Kd * derivative
        optimal_u = np.clip(optimal_u, self.action_space.low[0], self.action_space.high[0])

        # Update previous error for next step
        self.prev_error = error

        return np.array([optimal_u], dtype=np.float32)
        
    def step(self, action,Q_target):
        self.V += self.Q[0,0] * self.time_step
        # print(self.Q[0,0])
        # error = abs(self.V - V_target)
        # derivative = (error - self.error_prev) / self.time_step
        # self.error_prev = error
        u = action
        optimal_u = self.get_optimal_action(Q_target, self.Q[0,0])
        self.Q = self.A.dot(self.Q) * self.time_step+ self.B * u * self.time_step+ self.Q
        if self.Q[0,0] > Q_target:
            u = min(action + 0.1, self.action_space.high[0])  # Increase u
        else:
            u = max(action - 0.1, self.action_space.low[0])  

        self.error = Q_target - self.Q[0,0]
        reward = 50 - 10 * np.log(1 + abs(self.error))
        reward -= 0.2 * abs(self.Q[0, 0] - Q_target)        
        if abs(self.error) < 10:
            reward += 100
        elif abs(self.error) < 50:
            reward += 50
        # print(f"Step Debug - Q: {self.Q[0,0]:.2f}, Q_target: {Q_target}, Error: {error:.2f}, Reward: {reward:.2f}")
        self.prev_error = self.error

        done = False
        self.state = np.array([self.V, Q_target])
        return self.state,self.Q[0,0], reward, done, {"optimal_action": optimal_u}
    
    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.V = 0 
        self.Q = np.array([[0.0], [0.0]])
        self.Q_target = 300
        self.P = 0  
        self.state = np.array([self.V, self.Q_target])
        return self.state

    def render(self, mode="human"):
        pass
    def close(self):
        pass
        
import gymnasium as gym 
import numpy as np
import math

class Ventilator(gym.Env):
    def __init__(self):
        super(Ventilator, self).__init__()
        self.env = gym.make
        self.R = 5    # Sức cản đường thở (cmH2O·s/L)
        self.C = 10  # Độ giãn nở (mL/cmH2O)
        self.PEEP = 5 # Áp suất cuối kỳ (cmH2O)
        self.time_step = 0.001
        self.A = np.array([[0, 1], [-2500 / 3, -175 / 3]])
        self.B = np.array([[0], [442500]])
        self.error_prev = 0
        self.prev_error = 0
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)
        self.reset()
    
    def get_optimal_action(self, Q_target, Q_t):
        """
        Computes the optimal action (u) needed to move Q towards Q_target.
        """
        B_Q = 442500
        A_Q = -2500 / 3
        time_step = 0.001

        optimal_u = (Q_target - Q_t + (A_Q * Q_t * time_step)) / (B_Q * time_step)
        
        # Ensure action is within valid range
        optimal_u = np.clip(optimal_u, self.action_space.low[0], self.action_space.high[0])
        
        return np.array([optimal_u], dtype=np.float32)

    def step(self, action,Q_target):
        self.V += self.Q[0,0] * self.time_step
        # error = abs(self.V - V_target)
        # derivative = (error - self.error_prev) / self.time_step
        # self.error_prev = error
        u = action
        self.Q = self.A.dot(self.Q) * self.time_step+ self.B * u * self.time_step+ self.Q

        optimal_u = self.get_optimal_action(Q_target, self.Q[0,0])
        error = Q_target - self.Q[0,0]
        reward = 50 - 10 * np.log(1 + abs(error))
        reward -= 0.2 * abs(self.Q[0, 0] - Q_target)        
        if abs(error) < 10:
            reward += 100  # Strong reward for being very close
        elif abs(error) < 50:
            reward += 50
        # print(f"Step Debug - Q: {self.Q[0,0]:.2f}, Q_target: {Q_target}, Error: {error:.2f}, Reward: {reward:.2f}")
        self.prev_error = error

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
        

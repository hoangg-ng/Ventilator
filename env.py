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

    def step(self, action,Q_target):
        self.V += self.Q[0,0] * self.time_step
        u = action
        self.Q = self.A.dot(self.Q) * self.time_step+ self.B * u * self.time_step+ self.Q

        error = abs(Q_target - self.Q[0,0])
        # reward = 0
        reward = 50 - 10 * np.log(1 + abs(error))
        reward -= 0.2 * error      
        if error < 3:
            reward += 100
        elif error < 10:
            reward += 30
        elif error < 50:
            reward += 10
        action_high = self.action_space.high[0]
        action_low = self.action_space.low[0]
        if abs(u - action_high) < 1e-3 or abs(u - action_low) < 1e-3:
            reward -= 5  # adjust the penalty strength as needed

        self.prev_error = error
        done = False
        self.state = np.array([self.V, Q_target])
        return self.state,self.Q[0,0], reward, done
    
    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.V = 0 
        self.Q = np.array([[0.0], [0.0]])
        self.Q_target = np.random.uniform(100, 400)  # Random Q_target between 100 and 400
        self.P = 0  
        self.state = np.array([self.V, self.Q_target])
        return self.state

    def render(self, mode="human"):
        pass
    def close(self):
        pass
        

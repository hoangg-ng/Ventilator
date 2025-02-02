import gymnasium as gym 
import numpy as np
import math
# from gym import spaces
# from encoder import Encoder
# import RPi.GPIO as GPIO


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
        # self.status = np.zeros((3, 1))
        # self.state = np.array([0.0, 0.0, 0.0])  # [i_a, omega]
        # Action =: u(t)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        # Observation: R, C, Q, PEEP, P
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        self.reset()

    def step(self, action,V_target):
        self.V += self.Q[0,0] * self.time_step
        error = abs(self.V - V_target)
        # derivative = (error - self.error_prev) / self.time_step
        # self.error_prev = error
        u = action * 5
        u = np.clip(action, 0, 5) 
        control_noise = np.random.normal(0, 0.05)  # Small noise for control effort
        u += control_noise # Add noise to the control signal
        self.Q = self.A.dot(self.Q) * self.time_step+ self.B * u * self.time_step+ self.Q
        # self.P = (self.V / self.C) + (self.R * self.Q) + self.PEEP

        tracking_error_cost = error * self.time_step  
        control_effort_cost = abs(u) * self.time_step  
        cost_value = tracking_error_cost + 0.1 * control_effort_cost

        reward = -cost_value
        # done = self.V >= self.target_volume 
        done = False
        self.state = np.array([self.V])
        return self.state,self.Q[0,0], reward, done, {}
    
    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.V = 0 
        self.Q = np.array([[0.0], [0.0]])
        self.P = 0  
        # self.state = np.array([self.R, self.C, self.Q, self.V, self.P])
        self.state = np.array([self.V])
        return self.state

    def render(self, mode="human"):
        pass
    def close(self):
        pass
        

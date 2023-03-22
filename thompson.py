from baselines import BaseAgent
import numpy as np
from constants import LINEAR_BANDIT_COLUMNS

class ThompsonSamplingAgent(BaseAgent):
    def __init__(self, v=1):
        self.num_features = len(LINEAR_BANDIT_COLUMNS)
        self.num_actions = 3
        self.means = np.zeros((self.num_actions, self.num_features))
        self.v = v
        self.covs = np.array([np.identity(self.num_features) for _ in range(self.num_actions)])
        self.f = np.zeros((self.num_actions, self.num_features))
        self.t = 0
    
    def act(self, observation):
        if (self.t < self.num_actions):
            return self.t
        x = observation[LINEAR_BANDIT_COLUMNS].to_numpy()
        p = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            sample = np.random.multivariate_normal(self.means[a], np.linalg.inv(self.covs[a]) * (self.v ** 2))
            p[a] = sample @ x
        return np.argmax(p)
    
    def update(self, observation, action, reward):
        self.t += 1
        x = observation[LINEAR_BANDIT_COLUMNS].to_numpy()
        self.f[action] += x * reward
        self.means[action] = np.linalg.inv(self.covs[action]) @ self.f[action]
        self.covs[action] += np.outer(x, x)

from baselines import BaseAgent
import numpy as np
from constants import LINEAR_BANDIT_COLUMNS

class ThompsonSamplingAgent(BaseAgent):
    def __init__(self):
        self.num_features = len(LINEAR_BANDIT_COLUMNS)
        self.num_actions = 3
        # Do I create means and covs for each action?
        self.means = np.zeros((self.num_actions, self.num_features))
        self.R = 1
        self.eps = 1 / np.log(5528)
        self.delta = 0.1
        self.v = self.R * np.sqrt(24 / self.eps * np.log(1 / self.delta) * self.num_features)
        self.covs = np.array([np.identity(self.num_features) for _ in range(self.num_actions)])
        self.f = np.zeros((self.num_actions, self.num_features))
        self.t = 0
    
    def act(self, observation):
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
    

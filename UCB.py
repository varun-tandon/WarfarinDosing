from baselines import BaseAgent
import numpy as np
from constants import LINEAR_BANDIT_COLUMNS

class UCBAgent(BaseAgent):
    def __init__(self):
        self.num_actions = 3
        self.Q = np.ones(self.num_actions)
        self.N = np.zeros(self.num_actions)
        self.t = 0

    def act(self, observation):
        if self.t < self.num_actions:
            return self.t
        else:
            return np.argmax(self.Q + np.sqrt(2 * np.log(self.t) / self.N))
    
    def update(self, observation, action, reward):
        self.t += 1
        self.N[action] += 1
        self.Q[action] = self.Q[action] * ((self.N[action] - 1) / self.N[action]) + reward / self.N[action]

class LinUCB(BaseAgent):
    def __init__(self):
        self.alpha = 0.1
        self.num_actions = 3
        self.num_features = len(LINEAR_BANDIT_COLUMNS)
        self.A = np.array([np.identity(self.num_features) for _ in range(self.num_actions)])
        self.b = np.zeros((self.num_actions, self.num_features))
        self.theta = np.zeros((self.num_actions, self.num_features))
        self.t = 0
    
    def act(self, observation):
        if self.t < self.num_actions:
            return self.t
        else:
            x = observation[LINEAR_BANDIT_COLUMNS].to_numpy()
            p = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                self.theta[a] = np.linalg.solve(self.A[a], self.b[a])
                p[a] = self.theta[a] @ x + self.alpha * np.sqrt(x.T @ np.linalg.solve(self.A[a], x))
            return np.argmax(p)
    
    def update(self, observation, action, reward):
        self.t += 1
        x = observation[LINEAR_BANDIT_COLUMNS].to_numpy()
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x
        self.theta[action] = np.linalg.inv(self.A[action]) @ self.b[action]

class EnsembleSamplingAgent(BaseAgent):
    def __init__(self, num_models=10):
        self.num_models = num_models
        self.models = [LinUCBAgent() for _ in range(self.num_models)]
    
    def act(self, observation):
        model = np.random.choice(self.models)
        return model.act(observation)
        # if self.t < self.num_actions:
        #     return self.t
        # else:
        #     x = observation[LINEAR_BANDIT_COLUMNS].to_numpy()
        #     p = np.zeros(self.num_actions)
        #     model = np.random.choice(self.models)
        #     for a in range(self.num_actions):
        #         model.theta[a] = np.linalg.solve(model.A[a], model.b[a])
        #         p[a] = model.theta[a] @ x
        #     return np.argmax(p)
    
    def update(self, observation, action, reward):
        for model in self.models:
            model.update(observation, action, reward)


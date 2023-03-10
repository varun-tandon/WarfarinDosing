from constants import (
    DOSAGE_BUCKET_COLUMN,
    LINEAR_BANDIT_COLUMNS,
    Actions
    )
from utils import convert_dosage_to_action
from sklearn import linear_model
from baselines import BaseAgent
import numpy as np

class SupervisedLearningAgent(BaseAgent):
    def __init__(self, model_type="linreg"):
        # we need to remember our past actions in order to update our weights
        self.feature_dim = len(LINEAR_BANDIT_COLUMNS)
        self.seen_observations = np.zeros((0, self.feature_dim))
        self.seen_labels = np.zeros(0)
        self.model_type = model_type

    def act(self, observation):
        # if we have no data, just pick the middle action!
        if self.seen_observations.shape[0] == 0:
            return Actions.MEDIUM.value
        
        if self.model_type == "linreg":
            reg = linear_model.LinearRegression()
        elif self.model_type == "ridge":
            reg = linear_model.Ridge(alpha=0.5)

        reg.fit(self.seen_observations, self.seen_labels)
        relevant_features = observation[LINEAR_BANDIT_COLUMNS].to_numpy().reshape(1, -1)
        pred = reg.predict(relevant_features)
        return convert_dosage_to_action(pred[0])

    # add this observation to our history
    def update(self, observation, *_):
        self.seen_observations = np.vstack((self.seen_observations, observation[LINEAR_BANDIT_COLUMNS].to_numpy()))
        self.seen_labels = np.append(self.seen_labels, observation[DOSAGE_BUCKET_COLUMN])
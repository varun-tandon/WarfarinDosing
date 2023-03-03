from constants import (
    CLINICAL_DOSING_COLUMNS, 
    CLINICAL_DOSING_BASELINE_WEIGHTS, 
    DOSAGE_BUCKET_COLUMN,
    LINEAR_BANDIT_COLUMNS,
    Actions
    )
from utils import convert_dosage_to_action
from sklearn import linear_model

import numpy as np

class BaseAgent:
    def act(self, observation):
        raise NotImplementedError
    
    def update(self, observation):
        pass
    

class FixedDoseAgent(BaseAgent):
    def act(self, observation):
        return Actions.MEDIUM.value
    

class LinearAgent(BaseAgent):
    def act(self, observation):
        relevant_features = observation[CLINICAL_DOSING_COLUMNS].to_numpy()
        relevant_features = np.insert(relevant_features, 0, 1)
        return convert_dosage_to_action(np.dot(relevant_features, CLINICAL_DOSING_BASELINE_WEIGHTS) ** 2)

class LinearBanditAgent(BaseAgent):
    def __init__(self):
        # we need to remember our past actions in order to update our weights
        self.feature_dim = len(LINEAR_BANDIT_COLUMNS)
        self.seen_observations = np.zeros((0, self.feature_dim))
        self.seen_labels = np.zeros(0)

    def act(self, observation):
        # if we have no data, just pick the middle action!
        if self.seen_observations.shape[0] == 0:
            return Actions.MEDIUM.value
        
        reg = linear_model.LinearRegression()
        reg.fit(self.seen_observations, self.seen_labels)
        relevant_features = observation[LINEAR_BANDIT_COLUMNS].to_numpy().reshape(1, -1)
        pred = reg.predict(relevant_features)
        return convert_dosage_to_action(pred[0])

    # add this observation to our history
    def update(self, observation):
        self.seen_observations = np.vstack((self.seen_observations, observation[LINEAR_BANDIT_COLUMNS].to_numpy()))
        self.seen_labels = np.append(self.seen_labels, observation[DOSAGE_BUCKET_COLUMN])

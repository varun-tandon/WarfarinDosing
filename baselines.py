from constants import (
    CLINICAL_DOSING_COLUMNS, 
    CLINICAL_DOSING_BASELINE_WEIGHTS, 
    Actions
    )
from utils import convert_dosage_to_action

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

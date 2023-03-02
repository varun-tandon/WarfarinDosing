import clean_dataset
import metrics
from constants import (
    DOSAGE_COLUMN, AGE_COLUMN, HEIGHT_COLUMN, WEIGHT_COLUMN, 
    RACE_COLUMN, ENZYME_INDUCER_COLUMNS, CLINICAL_DOSING_COLUMNS,
    CLINICAL_DOSING_BASELINE_WEIGHTS, Actions)

import numpy as np

class BaseAgent:
    def act(self, observation):
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError
    

class FixedDoseAgent(BaseAgent):
    def act(self, observation):
        return Actions.MEDIUM
    

class LinearAgent(BaseAgent):
    def act(self, observation):
        return Actions.MEDIUM
        

def fixed_dose_baseline(data):
    return [35 for _ in range(len(data))]

def clinical_dosing_baseline(data):
    data_matrix = data[CLINICAL_DOSING_COLUMNS].to_numpy()
    data_matrix = np.insert(data_matrix, 0, 1, axis=1)
    return (data_matrix @ np.array(CLINICAL_DOSING_BASELINE_WEIGHTS)) ** 2

def main():
    # Load the dataset, it will be cleaned
    data = clean_dataset.read_data()

    fixed_dose_results = fixed_dose_baseline(data)
    fixed_dose_accuracy = metrics.compute_accuracy(data[DOSAGE_COLUMN], fixed_dose_results)
    
    print('Accuracy of fixed dose baseline: {}'.format(fixed_dose_accuracy))
    clinical_dosing_results = clinical_dosing_baseline(data)
    clinical_dose_accuracy = metrics.compute_accuracy(data[DOSAGE_COLUMN], clinical_dosing_results)
    print('Accuracy of clinical dosing baseline: {}'.format(clinical_dose_accuracy))

if __name__ == '__main__':
    main()
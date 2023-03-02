import dataset
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
    # Start by transforming the data with the features needed for the algo per S1f
    data['Age in decades'] = data[AGE_COLUMN].apply(lambda x: int(x[0]))
    data['Enzyme inducer status'] = (data[ENZYME_INDUCER_COLUMNS].sum(axis=1) > 0).astype(int)
    data['Amiodarone status'] = data['Amiodarone (Cordarone)'].fillna(0)
    data['Asian race'] = data[RACE_COLUMN].apply(lambda x: 1 if x == 'Asian' else 0)
    data['Black or African American'] = data[RACE_COLUMN].apply(lambda x: 1 if x == 'Black or African American' else 0)
    data['Missing or Mixed race'] = data[RACE_COLUMN].apply(lambda x: 1 if x == 'Unknown' else 0)

    data_matrix = data[CLINICAL_DOSING_COLUMNS].to_numpy()
    data_matrix = np.insert(data_matrix, 0, 1, axis=1)
    return (data_matrix @ np.array(CLINICAL_DOSING_BASELINE_WEIGHTS)) ** 2

def main():
    # Load the dataset, it will be cleaned
    data = dataset.read_data()

    fixed_dose_results = fixed_dose_baseline(data)
    fixed_dose_accuracy = metrics.compute_accuracy(data[DOSAGE_COLUMN], fixed_dose_results)
    
    print('Accuracy of fixed dose baseline: {}'.format(fixed_dose_accuracy))
    clinical_dosing_results = clinical_dosing_baseline(data)
    clinical_dose_accuracy = metrics.compute_accuracy(data[DOSAGE_COLUMN], clinical_dosing_results)
    print('Accuracy of clinical dosing baseline: {}'.format(clinical_dose_accuracy))

if __name__ == '__main__':
    main()
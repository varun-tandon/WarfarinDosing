from baselines import BaseAgent
import numpy as np
from constants import LINEAR_BANDIT_COLUMNS, Actions, DOSAGE_BUCKET_COLUMN
from sklearn import linear_model
from utils import convert_dosage_to_action


class EnsembleSamplingAgent(BaseAgent):
    def __init__(self, num_models=10, sigma_w=1):
        self.num_models = num_models
        self.feature_dim = len(LINEAR_BANDIT_COLUMNS)
        self.seen_observations = [np.zeros((0, self.feature_dim)) for _ in range(self.num_models)]
        self.seen_labels = [np.zeros(0) for _ in range(self.num_models)]
        # hyperparameter to tune
        self.sigma_w = sigma_w
        self.t = 0

    def act(self, observation):
        # if we have no data, just pick the middle action!
        if self.t < 3:
            return Actions.MEDIUM.value
        
        # now we'll pick a random model and use it to choose the action
        model_id = np.random.choice(self.num_models)
        
        # train a model on the specified data
        model = linear_model.LinearRegression()
        model.fit(self.seen_observations[model_id], self.seen_labels[model_id])

        # predict on this new data point
        relevant_features = observation[LINEAR_BANDIT_COLUMNS].to_numpy().reshape(1, -1)
        pred = model.predict(relevant_features)

        return convert_dosage_to_action(pred[0])

    def update(self, observation, *_):
        # we need to train all of the models with this new data point!
        for i in range(0, self.num_models):
            new_features = observation[LINEAR_BANDIT_COLUMNS].to_numpy().reshape(1, -1)
            # we also need to add some noise to the features
            noise = np.random.normal(0, self.sigma_w)

            self.seen_observations[i] = np.vstack((self.seen_observations[i], new_features))
            self.seen_labels[i] = np.append(self.seen_labels[i], observation[DOSAGE_BUCKET_COLUMN] + noise)
        self.t += 1

import argparse
import numpy as np
from config import Config
import random
import pandas as pd
from baselines import FixedDoseAgent, LinearAgent
from supervised import SupervisedLearningAgent
from thompson import ThompsonSamplingAgent
from UCB import UCBAgent, LinUCBAgent
from ensemble import EnsembleSamplingAgent
from utils import get_reward
import os
from tqdm import tqdm
from constants import DOSAGE_COLUMN, DOSAGE_BUCKET_COLUMN

parser = argparse.ArgumentParser()
parser.add_argument(
    "--agent", required=True, type=str, choices=[
        "fixed", "linear", "ucb", "linucb",
        "supervised-lin", "supervised-ridge", "supervised-ridge-0.01", "supervised-ridge-0.05", 
        "supervised-ridge-0.1", "supervised-ridge-0.5", "supervised-ridge-1", "supervised-ridge-5", 
        "thompson-0", "thompson-0.01", "thompson-0.1", "thompson-0.3", "thompson-0.5", "thompson-0.7", "thompson-1", "thompson-2", "thompson-5",
        "thompson-10", "thompson-20", "thompson-100",
        "ensemble-0", "ensemble-0.01", "ensemble-0.1", "ensemble-0.5", "ensemble-1", "ensemble-2", "ensemble-5", 
        "ensemble-10", "ensemble-20", "ensemble-100",
    ]
)

if __name__ == "__main__":
    args = parser.parse_args()

    # first get all of the data
    df = pd.read_csv('data/warfarin_clean.csv')
    print(len(df.columns))

    accuracies = []
    regrets = []

    # we need to run our big boi 20 times! 
    for seed in range(0, 20):
        # now select the agent
        agent = None
        if args.agent == 'fixed':
            agent = FixedDoseAgent()
        elif args.agent == 'linear':
            agent = LinearAgent()
        elif args.agent == 'ucb':
            agent = UCBAgent()
        elif args.agent == 'linucb':
            agent = LinUCBAgent()
        elif args.agent == 'supervised-lin':
            agent = SupervisedLearningAgent()
        # this one represnts the optimal ridge regression
        elif args.agent == 'supervised-ridge':
            agent = SupervisedLearningAgent(model_type='ridge', alpha=0.5)
        elif args.agent == 'supervised-ridge-0.01':
            agent = SupervisedLearningAgent(model_type='ridge', alpha=0.01)
        elif args.agent == 'supervised-ridge-0.05':
            agent = SupervisedLearningAgent(model_type='ridge', alpha=0.05)
        elif args.agent == 'supervised-ridge-0.1':
            agent = SupervisedLearningAgent(model_type='ridge', alpha=0.1)
        elif args.agent == 'supervised-ridge-0.5':
            agent = SupervisedLearningAgent(model_type='ridge', alpha=0.5)
        elif args.agent == 'supervised-ridge-1':
            agent = SupervisedLearningAgent(model_type='ridge', alpha=1)
        elif args.agent == 'supervised-ridge-5':
            agent = SupervisedLearningAgent(model_type='ridge', alpha=5)
        elif args.agent == 'thompson-0':
            agent = ThompsonSamplingAgent(v=0)
        elif args.agent == 'thompson-0.01':
            agent = ThompsonSamplingAgent(v=0.01)
        elif args.agent == 'thompson-0.1':
            agent = ThompsonSamplingAgent(v=0.1)
        elif args.agent == 'thompson-0.3':
            agent = ThompsonSamplingAgent(v=0.3)
        elif args.agent == 'thompson-0.5':
            agent = ThompsonSamplingAgent(v=0.5)
        elif args.agent == 'thompson-0.7':
            agent = ThompsonSamplingAgent(v=0.7)
        elif args.agent == 'thompson-1':
            agent = ThompsonSamplingAgent(v=1)
        elif args.agent == 'thompson-2':
            agent = ThompsonSamplingAgent(v=2)
        elif args.agent == 'thompson-5':
            agent = ThompsonSamplingAgent(v=5)
        elif args.agent == 'thompson-10':
            agent = ThompsonSamplingAgent(v=10)
        elif args.agent == 'thompson-20':
            agent = ThompsonSamplingAgent(v=20)
        elif args.agent == 'thompson-100':
            agent = ThompsonSamplingAgent(v=100)
        elif args.agent == 'ensemble-0':
            agent = EnsembleSamplingAgent(sigma_w=0)
        elif args.agent == 'ensemble-0.01':
            agent = EnsembleSamplingAgent(sigma_w=0.01)
        elif args.agent == 'ensemble-0.1':
            agent = EnsembleSamplingAgent(sigma_w=0.1)
        elif args.agent == 'ensemble-0.5':
            agent = EnsembleSamplingAgent(sigma_w=0.5)
        elif args.agent == 'ensemble-1':
            agent = EnsembleSamplingAgent(sigma_w=1)
        elif args.agent == 'ensemble-2':
            agent = EnsembleSamplingAgent(sigma_w=2)
        elif args.agent == 'ensemble-5':
            agent = EnsembleSamplingAgent(sigma_w=5)
        elif args.agent == 'ensemble-10':
            agent = EnsembleSamplingAgent(sigma_w=10)
        elif args.agent == 'ensemble-20':
            agent = EnsembleSamplingAgent(sigma_w=20)
        elif args.agent == 'ensemble-100':
            agent = EnsembleSamplingAgent(sigma_w=100)
        else:
            raise ValueError("Agent type not recognized")
        # set our seeds
        np.random.seed(seed)
        random.seed(seed)
        config = Config(args.agent, seed)

        # reorder the data
        cur_df = df.sample(frac=1, random_state=seed)
        num_wrong = 0
        accuracy = np.zeros(len(cur_df))
        regret = np.zeros(len(cur_df))
        i = 1

        for _, observation in tqdm(cur_df.iterrows(), total=len(cur_df)):
            # action is the dosage bucket
            action = agent.act(observation)
            reward = get_reward(observation, action)
            agent.update(observation, action, reward)

            # record the performance and the loss
            if reward == -1:
                num_wrong += 1

            if i - 1 == 0:
                regret[i - 1] = -reward
            else:
                regret[i - 1] = regret[i - 2] - reward
            accuracy[i - 1] = num_wrong / i
            i += 1
        
        accuracies.append(1-accuracy[-1])
        regrets.append(regret[-1])
        print("Accuracy: {}".format(1 - accuracy[-1]))
        print("Regret: {}".format(regret[-1]))
        
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        np.save(config.accuracy_output, accuracy)
        np.save(config.regret_output, regret)
    print(f"Average FIDD: {1 - sum(accuracies)/len(accuracies)}")
    print(f"Average regret: {sum(regrets)/len(regrets)}")
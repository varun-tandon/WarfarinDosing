import argparse
import numpy as np
from config import Config
import random
import pandas as pd
from baselines import FixedDoseAgent, LinearAgent, StochasticLinearBanditAgent
from utils import get_reward
import os


parser = argparse.ArgumentParser()
parser.add_argument(
    "--agent", required=True, type=str, choices=["fixed", "linear", "slb"]
)


if __name__ == "__main__":
    args = parser.parse_args()

    # first get all of the data
    df = pd.read_csv('data/warfarin_clean.csv')

    # now select the agent
    agent = None
    if args.agent == 'fixed':
        agent = FixedDoseAgent()
    elif args.agent == 'linear':
        agent = LinearAgent()
    elif args.agent == 'slb':
        agent = StochasticLinearBanditAgent()
    else:
        raise ValueError("Agent type not recognized")

    # we need to run our big boi 20 times! 
    for seed in range(0, 1):
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

        for _, observation in cur_df.iterrows():
            action = agent.act(observation)
            agent.update()

            reward = get_reward(observation, action)
            # record the performance and the loss
            if reward == -1:
                num_wrong += 1

            if i - 1 == 0:
                regret[i - 1] = reward
            else:
                regret[i - 1] = regret[i - 2] + reward
            accuracy[i - 1] = num_wrong / i
            i += 1
        
        print("Accuracy: {}".format(1 - accuracy[-1]))
        print("Regret: {}".format(regret[-1]))
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        np.save(config.accuracy_output, accuracy)
        np.save(config.regret_output, regret)
        
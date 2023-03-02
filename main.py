import argparse
import numpy as np
from config import Config
import random
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument(
    "--agent", required=True, type=str, choices=["fixed", "linear", "slb"]
)


if __name__ == "__main__":
    args = parser.parse_args()

    # first get all of the data
    df = pd.read_csv('data/warfarin_clean.csv')
    print(df.head())

    # now select the agent
    agent = None
    if args.agent == 'fixed':
        pass
    elif args.agent == 'linear':
        pass
    elif args.agent == 'slb':
        pass
    else:
        raise ValueError("Agent type not recognized")

    # we need to run our big boi 20 times! 
    for seed in range(1, 2):
        # set our seeds
        np.random.seed(seed)
        random.seed(seed)
        config = Config(args.agent, seed)

        # reorder the data
        cur_df = df.sample(frac=1, random_state=seed)

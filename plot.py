from collections import defaultdict
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import scipy.stats as stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_combined(name, results):
    results = np.array(results)
    xs = np.arange(results.shape[1])
    ys = np.mean(results, axis=0)
    yerrs = stats.sem(results, axis=0)
    plt.fill_between(xs, ys - yerrs, ys + yerrs, alpha=0.25)
    plt.plot(xs, ys, label=name)


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("--agent", required=True, type=str, choices=["fixed", "linear", "linearbandit"])
    # args = parser.parse_args()


    all_results = {"Fixed baseline": [], "Linear baseline": [], "Linear Bandits": []}
    for seed in range(0, 2):
        format_str = f"results-{{}}-seed={seed}"
        all_results["Baseline"].append(
            np.load(directory / format_str.format("baseline") / "scores.npy")
        )
        all_results["No baseline"].append(
            np.load(directory / format_str.format("no_baseline") / "scores.npy")
        )
        all_results["PPO"].append(
            np.load(directory / format_str.format("ppo") / "scores.npy")
        )
    
    print(all_results.shape)

    # plt.figure()
    # plt.title(args.env_name)
    # plt.xlabel("Iteration")
    # for name, results in all_results.items():
    #     plot_combined(name, results)
    # plt.legend()
    # plt.savefig(directory / f"results-{args.env_name}", bbox_inches="tight")

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
    # get the 95% confidence interval
    yerrs = stats.sem(results, axis=0) * stats.t.ppf((1 + 0.95) / 2, results.shape[0] - 1)

    plt.fill_between(xs, ys - yerrs, ys + yerrs, alpha=0.25)
    plt.ylim(0.3, 0.6)
    plt.plot(xs, ys, label=name)


if __name__ == "__main__":
    # all_results = {"Fixed baseline": [], "Linear baseline": [], 
    #                "UCB": [], "linUCB": [], "Supervised Linear Regression": [], 
    #                "Supervised Ridge Regression": [], "Thompson Sampling": []}
    
    all_results = {"Fixed baseline": [], "Linear baseline": [], 
                   "UCB": []}
    
    for seed in range(0, 20):
        format_str = f"{{}}-seed={seed}"
        all_results["Fixed baseline"].append(
            np.load("results/" + format_str.format("fixed") + "/accuracy.npy")
        )
        all_results["Linear baseline"].append(
            np.load("results/" + format_str.format("linear") + "/accuracy.npy")
        )
        all_results["UCB"].append(
            np.load("results/" + format_str.format("ucb") + "/accuracy.npy")
        )

    plt.figure()
    plt.title("Accuracy")
    plt.xlabel("Iteration")
    for name, results in all_results.items():
        plot_combined(name, results)
    plt.legend()
    plt.savefig("results-accuracy", bbox_inches="tight")

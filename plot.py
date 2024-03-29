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
    plt.plot(xs, ys, label=name)


def plot_accuracy(all_results, title, filename):
    plt.figure()
    plt.title(title)
    plt.xlabel("Iteration")
    for name, results in all_results.items():
        plot_combined(name, results)

    plt.ylim(0.2, 0.8)
    plt.legend()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def plot_regret(all_results, title, filename):
    plt.figure()
    plt.title(title)
    plt.xlabel("Iteration")
    for name, results in all_results.items():
        plot_combined(name, results)
    
    plt.legend()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--agents", required=True, help="Comma-separated list of agents to plot"
    )
    args = parser.parse_args()
    agents = args.agents.split(",")

    agent_mapping = {
        "fixed": "Fixed baseline",
        "linear": "Linear baseline",
        "ucb": "UCB",
        "linucb": "LinUCB",
        "supervised-lin": "Supervised Linear Regression",
        # this one represnts the optimal ridge regression
        "supervised-ridge": "Supervised Ridge Regression (alpha=0.5)",
        "supervised-ridge-0.01": "alpha=0.01",
        "supervised-ridge-0.05": "alpha=0.05",
        "supervised-ridge-0.1": "alpha=0.1",
        "supervised-ridge-0.5": "alpha=0.5",
        "supervised-ridge-1": "alpha=1",
        "supervised-ridge-5": "alpha=5",
        "thompson-0": "variance=0",
        "thompson-0.01": "variance=0.01",
        "thompson-0.1": "variance=0.1",
        "thompson-0.3": "variance=0.3",
        "thompson-0.5": "variance=0.5",
        "thompson-0.7": "variance=0.7",
        "thompson-1": "variance=1",
        "thompson-2": "variance=2",
        "thompson-5": "variance=5",
        "thompson-10": "Thompson (variance=10)",
        "thompson-20": "variance=20",
        "thompson-100": "variance=100",
        "ensemble-0": "noise_std = 0",
        "ensemble-0.01": "noise_std = 0.01",
        "ensemble-0.1": "Ensemble (noise_std = 0.1)",
        "ensemble-0.5": "noise_std = 0.5",
        "ensemble-1": "Ensemble (noise_std = 1)",
        "ensemble-2": "noise_std = 2",
        "ensemble-5": "noise_std = 5",
        "ensemble-10": "noise_std = 10",
        "ensemble-20": "noise_std = 20",
        "ensemble-100": "noise_std = 100",
    }

    all_results_accuracy = {}
    all_results_regret = {}
    
    for seed in range(0, 20):
        format_str = f"{{}}-seed={seed}"
        for agent in agents:
            if agent not in agent_mapping:
                raise ValueError(f"Agent {agent} not recognized")
            name = agent_mapping[agent]

            if name not in all_results_accuracy:
                all_results_accuracy[name] = []
                all_results_regret[name] = []

            all_results_accuracy[name].append(
                np.load("results/" + format_str.format(agent) + "/accuracy.npy")
            )
            all_results_regret[name].append(
                np.load("results/" + format_str.format(agent) + "/regret.npy")
            )

    accuracy_title = '_'.join(agents) + '_accuracy.png'
    regret_title = '_'.join(agents) + '_regret.png'

    plot_accuracy(all_results_accuracy, "Fraction of Incorrect Dosing Decisions", f"plots/{accuracy_title}")
    plot_regret(all_results_regret, "Cumulative Regret", f"plots/{regret_title}")
    

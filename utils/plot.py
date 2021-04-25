from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from pymc3 import plot_posterior_predictive_glm


def traceplot(trace):
    plt.figure(figsize=(7, 7))
    traceplot(trace[100:])
    plt.tight_layout()


def plot_posterior_predictive_samples(trace:np.array, data:Callable, samples=100):

    x, y = data()
    plt.figure(figsize=(7, 7))
    plt.plot(x, y, "x", label="data")
    plot_posterior_predictive_glm(
        trace, samples=samples, label="posterior predictive regression lines"
    )
    plt.title("Posterior predictive regression lines")
    plt.legend(loc=0)
    plt.xlabel("x")
    plt.ylabel("y")
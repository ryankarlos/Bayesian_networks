import pymc3 as pm
from data.data import generate_data
from typing import Callable

from pymc3_.map_and_sampling import map_estimation
from pymc3 import Normal, HalfCauchy, plot_posterior_predictive_glm
import matplotlib.pyplot as plt

def linear_regression(data:Callable)->pm.Model:

    X1, X2,Y = data()
    basic_model = pm.Model()
    with basic_model:
        # Define priors
        sigma = HalfCauchy('sigma', beta=10, testval=1.)
        intercept = Normal('Intercept', 0, sigma=20)
        x_coeff = Normal('x', 0, sigma=20)
        likelihood = Normal('y', mu=intercept + x_coeff * x,
                            sigma=sigma, observed=y)

    return basic_model


def logistic_regression():
    pass



def traceplot(trace)
    plt.figure(figsize=(7, 7))
    traceplot(trace[100:])
    plt.tight_layout();


def plot_posterior_predictive_samples(data:Callable, samples=100):

    plt.figure(figsize=(7, 7))
    plt.plot(x, y, 'x', label='data')
    plot_posterior_predictive_glm(trace, samples=samples,
                                  label='posterior predictive regression lines')

    plt.title('Posterior predictive regression lines')
    plt.legend(loc=0)
    plt.xlabel('x')
    plt.ylabel('y')

if __name__ == "__main__":

    basic_model:pm.Model = linear_regression(generate_data)
    map = map_estimation(basic_model, method='powell')
    print(map)



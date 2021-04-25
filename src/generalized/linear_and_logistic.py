import pymc3 as pm
from data.data import simulate_data_lr
from typing import Callable
from utils.plot import traceplot
from utils.map_and_sampling import map_estimation
from pymc3 import Normal, HalfCauchy
from functools import partial
from utils.constants import LIN_REG_PARAMS
import logging

log = logging.getLogger("linear_and_logistic")


def linear_regression(data: Callable, samples=None) -> pm.Model:

    x, y = data()
    basic_model = pm.Model()
    with basic_model:
        # Define priors
        sigma = HalfCauchy("sigma", beta=10, testval=1.0)
        intercept = Normal("Intercept", 0, sigma=20)
        x_coeff = Normal("x", 0, sigma=20)
        likelihood = Normal("y", mu=intercept + x_coeff * x, sigma=sigma, observed=y)
        map = map_estimation(basic_model, method="powell")
        if samples is not None:
            if not isinstance(samples, int):
                raise (ValueError, "samples arg must be int")
            elif samples < 50:
                raise (ValueError, "samples, must be greater than 50")
            else:
                trace = pm.sample(samples)
                traceplot(trace)
    return basic_model, trace


def logistic_regression():
    pass


if __name__ == "__main__":
    data_gen = partial(simulate_data_lr, **LIN_REG_PARAMS)
    basic_model: pm.Model = linear_regression(data_gen)
    log.info("Running MAP mode")

    print(map)
    log.info("Running NUTS")
    basic_model: pm.Model = linear_regression(data_gen, samples=500)
    ppc = pm.sample_posterior_predictive(trace, samples=500, model=model)

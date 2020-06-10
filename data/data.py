import numpy as np


def simulate_data_lr(size, intercept, slope):
    size = size
    c = intercept
    m = slope
    # Initialize random number generator
    np.random.seed(123)
    # Predictor variable
    x = np.linspace(0, 1, size)
    true_regression_line = m * x + c
    # add noise
    sigma = np.random.normal(scale=0.5, size=size)
    # Outcome variable
    y = true_regression_line + sigma
    return x, y

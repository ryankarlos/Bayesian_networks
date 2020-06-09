import numpy as np


def generate_data(size=100):
    # Initialize random number generator
    np.random.seed(123)
    # Predictor variable
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2
    # True parameter values
    alpha, sigma = 1, 1
    beta = [1, 2.5]
    # Simulate outcome variable
    Y = alpha + beta[0] * X1 + beta[1] * X2 + np.random.randn(size) * sigma

    return X1, X2, Y

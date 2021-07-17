import bnlearn as bn
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


def bayesian_network_datasets(name="asia", samples=10000):
    """
    Generates well known sample/toy datasets for bayesian networks,
    by sampling from existing graph model.

    Parameters
    ----------
    name: str, (default:'asia')
        Name of the model to sample from
    samples: int, (default: 10000)
        Number of observations for our dataset
    Returns
    -------
    pd.DataFrame
    """
    model = bn.import_DAG(name)
    df = bn.sampling(model, n=samples)
    return df

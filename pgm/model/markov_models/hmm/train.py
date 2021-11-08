import warnings

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import prefect
from hmmlearn.hmm import GaussianHMM, MultinomialHMM
from matplotlib.dates import MonthLocator, YearLocator
from prefect import task

warnings.filterwarnings("ignore")


logger = prefect.context.get("logger")


@task
def train_discrete_hmm(
    X,
    lengths,
    start_probability,
    transition_probability,
    emission_probability,
    components=3,
    iterations=15,
    verbose=True,
):
    model = MultinomialHMM(components, iterations, verbose, init_params="mc")
    model.startprob_ = start_probability
    model.transmat_ = transition_probability
    model.emissionprob_ = emission_probability
    logger.info(
        f"commencing hmm model training with parameters- "
        f"Iterations: {iterations}, components: {components}"
    )
    model.fit(X, lengths)
    if is_converged(model):
        logger.info("model has converged")
        print(model.transmat_)
        print(model.emissionprob_)
        print(model.startprob_)
        return model
    else:
        logger.warning(
            "Model failed to converge. Try increasing number of iterations "
            "and/or initialise transition matrix"
        )
        return model


@task
def train_gaussian_hmm(X, components=4, iter=1000):
    logger.info("fitting to HMM and decoding ...")
    model = GaussianHMM(
        n_components=components, covariance_type="diag", n_iter=iter
    ).fit(X)
    return model


def is_converged(hmm_model):
    """
    Checks whether the EM algo has converged after training finishes
    Parameters
    ----------
    hmm_model

    Returns
    -------

    """
    return hmm_model.monitor_.converged


def trained_model_distribution_params(hmm_model):
    return {"mean": hmm_model.means_, "cov_matrix": hmm_model.covars_}


@task
def plot_trained_parameters(model, hidden_states, dates, end_val):
    fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
    colours = cm.rainbow(np.linspace(0, 1, model.n_components))
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        # Use fancy indexing to plot data in each state.
        mask = hidden_states == i
        ax.plot_date(dates[mask], end_val[mask], ".-", c=colour)
        ax.set_title("{0}th hidden state".format(i))

        # Format the ticks.
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.grid(True)
    plt.show()


@task
def plot_likelihood_per_iteration(hmm_model):
    fig, ax = plt.subplots(1, 1)
    ax.plot(hmm_model.monitor_.history)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("log likelihood")
    plt.show()

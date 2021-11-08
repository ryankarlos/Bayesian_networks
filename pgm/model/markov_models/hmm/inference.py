import numpy as np
import prefect
from prefect import task

logger = prefect.context.get("logger")


@task(nout=2)
def decode_hidden_state_for_discrete_hmm(encoded_obs_seq, observations, states, model):
    logprob, hidden_states = model.decode(encoded_obs_seq, algorithm="viterbi")
    print(
        "Observed behaviour:",
        ", ".join(map(lambda x: observations[x], encoded_obs_seq.T[0])),
    )
    print("Inferred hidden states:", ", ".join(map(lambda x: states[x], hidden_states)))
    return logprob, hidden_states


@task
def decode_hidden_states_time_series(X, model):
    """
    Computes hidden state regimes for observations in time series using the
    vertibi algorithm
    Parameters
    ----------
    X
    model

    Returns
    -------

    """
    hidden_states = model.predict(X)
    logger.info("Transition matrix:")
    print(model.transmat_)
    return hidden_states


@task
def compute_mean_and_vars_hidden_state(model):
    for i in range(model.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", model.means_[i])
        print("var = ", np.diag(model.covars_[i]))

import random

import numpy as np
import prefect
from prefect import Flow, task

from pgm.model.markov_models.hmm.inference import decode_hidden_state_for_discrete_hmm
from pgm.model.markov_models.hmm.train import train_discrete_hmm

logger = prefect.context.get("logger")


states = ["Rainy", "Sunny"]
n_states = len(states)
observations = ["walk", "shop", "clean"]
n_observations = len(observations)
start_probability = np.array([0.6, 0.4])
transition_probability = np.array([[0.7, 0.3], [0.4, 0.6]])
emission_probability = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])


def _get_markov_edges(Q):
    """
    function that maps transition probability dataframe
    to markov edges and weights
    """
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx, col)] = Q.loc[idx, col]
    return edges


@task(nout=2)
def generate_random_seq_of_observations():
    seq = []
    lengths = []
    for _ in range(100):
        length = random.randint(5, 10)
        lengths.append(length)
        for _ in range(length):
            r = random.random()
            if r < 0.2:
                seq.append(0)  # walk
            elif r < 0.6:
                seq.append(1)  # shop
            else:
                seq.append(2)  # clean
    seq = np.array([seq]).T
    return seq, lengths


if __name__ == "__main__":
    with Flow("hmm-stocks") as flow:
        # predict a sequence of hidden states based on visible states
        seq, lengths = generate_random_seq_of_observations()
        model = train_discrete_hmm(
            seq,
            lengths,
            start_probability,
            transition_probability,
            emission_probability,
            components=n_states,
            iterations=30,
            verbose=True,
        )
        obs_states = np.array([[0, 2, 1, 1, 2, 0]]).T
        decode_hidden_state_for_discrete_hmm(obs_states, observations, states, model)
    flow_state = flow.run()
    flow.visualize(flow_state=flow_state)

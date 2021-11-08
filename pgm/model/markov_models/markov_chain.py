import networkx as nx
import pandas as pd
from pomegranate import ConditionalProbabilityTable, DiscreteDistribution, MarkovChain

prior_prob = {"S": 0.5, "I": 0.2, "R": 0.3}
cpd = [
    ["S", "S", 0.10],
    ["S", "I", 0.50],
    ["S", "R", 0.30],
    ["I", "S", 0.10],
    ["I", "I", 0.40],
    ["I", "R", 0.40],
    ["R", "S", 0.05],
    ["R", "I", 0.45],
    ["R", "R", 0.45],
]


def initialise_markov_chain(prior_prob, cpd):
    d1 = DiscreteDistribution(prior_prob)
    d2 = ConditionalProbabilityTable(cpd, [d1])
    clf = MarkovChain([d1, d2])
    return clf


def compute_probability_sequence(seq, clf):
    print(clf.log_probability(seq))


def generate_random_sample_from_model(clf, length, num_seqs):
    df = pd.DataFrame(columns=["sequences"])
    for i in range(num_seqs):
        obs = clf.sample(length)
        df = df.append({"sequences": obs}, ignore_index=True)
    return df


def build_markov_chain_from_data(df):
    seq = list(df["sequences"])
    model = MarkovChain.from_samples(seq)
    return model


def create_markov_networkx_object(model):
    cpd = model.distributions[1].to_dict()["table"]
    states = list(model.distributions[0].to_dict()["parameters"][0].keys())
    G = nx.MultiDiGraph()
    G.add_nodes_from(states)
    for k, l, v in cpd:
        tmp_origin, tmp_destination = k, l
        G.add_edge(tmp_origin, tmp_destination, weight=v, label=v[0:4])
    return G


if __name__ == "__main__":
    clf = initialise_markov_chain(prior_prob, cpd)
    df = generate_random_sample_from_model(clf, length=4, num_seqs=1000)
    clf = build_markov_chain_from_data(df)
    print(clf.distributions[0])
    print(clf.distributions[1])

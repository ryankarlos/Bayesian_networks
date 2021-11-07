import pandas as pd
from pomegranate import ConditionalProbabilityTable, DiscreteDistribution, MarkovChain


def initialise_markov_chain():
    d1 = DiscreteDistribution({"S": 0.5, "I": 0.2, "R": 0.3})
    d2 = ConditionalProbabilityTable(
        [
            ["S", "S", 0.10],
            ["S", "I", 0.50],
            ["S", "R", 0.30],
            ["I", "S", 0.10],
            ["I", "I", 0.40],
            ["I", "R", 0.40],
            ["R", "S", 0.05],
            ["R", "I", 0.45],
            ["R", "R", 0.45],
        ],
        [d1],
    )
    clf = MarkovChain([d1, d2])
    print(clf.distributions[0])
    print(clf.distributions[1])
    return clf


def compute_probability_sequence(seq, clf):
    print(clf.log_probability(seq))


def generate_random_sample_from_model(clf, length, num_seqs):
    df = pd.DataFrame(columns=["sequence"])
    for i in range(num_seqs):
        obs = clf.sample(length)
        df = df.append({"sequence": obs}, ignore_index=True)
    return df


if __name__ == "__main__":
    clf = initialise_markov_chain()
    df = generate_random_sample_from_model(clf, length=5, num_seqs=3)
    print(df)

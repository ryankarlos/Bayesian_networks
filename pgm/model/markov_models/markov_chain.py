from pomegranate import ConditionalProbabilityTable, DiscreteDistribution, MarkovChain

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
print(clf.log_probability(list("SSIR")))

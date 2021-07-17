import bnlearn


def plot_bayesian_network(DAG):
    bnlearn.plot(DAG)


def compare_network_plot(model, DAG):
    bnlearn.compare_networks(model, DAG)

import networkx as nx
import numpy as np
from networkx.algorithms import tree, community
from networkx.generators.ego import ego_graph
from sklearn.metrics import f1_score
from utils.log import get_logger

LOG = get_logger(__name__)


def get_f1_score(estimated_model, true_model):
    nodes = estimated_model.nodes()
    est_adj = nx.to_numpy_matrix(
        estimated_model.to_undirected(), nodelist=nodes, weight=None
    )
    true_adj = nx.to_numpy_matrix(
        true_model.to_undirected(), nodelist=nodes, weight=None
    )

    f1 = f1_score(np.ravel(true_adj), np.ravel(est_adj))
    print("F1-score for the model skeleton: ", f1)


def is_independent(X, Y, model):
    Zs = []
    return model.test_conditional_independence(X, Y, Zs, significance_level=0.05)


def compute_metrics(G):
    """
    degree, betweeness, communities
    """
    degree_dict = dict(G.degree(G.nodes()))
    betweenness_dict = nx.betweenness_centrality(G)  # Run betweenness centrality
    # eigenvector_dict = nx.eigenvector_centrality(G) # Run eigenvector centrality
    communities = community.greedy_modularity_communities(G)
    modularity_dict = {}  # Create a blank dictionary
    for i, c in enumerate(
        communities
    ):  # Loop through the list of communities, keeping track of the number for the community
        for name in c:  # Loop through each person in a community
            modularity_dict[
                name
            ] = i  # Create an entry in the dictionary for the person, where the value is which group they belong to.

    return communities, {
        "degree": degree_dict,
        "betweeness": betweenness_dict,
        "modularity": modularity_dict,
    }

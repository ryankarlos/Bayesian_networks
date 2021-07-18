import networkx as nx
import numpy as np
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

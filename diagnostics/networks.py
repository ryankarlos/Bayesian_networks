import networkx as nx
import numpy as np
from pgmpy.estimators import K2Score
from sklearn.metrics import f1_score

from utils.logconfig import module_logger

LOG = module_logger()


def get_f1_score(estimated_model, true_model):
    """
    Funtion to evaluate the learned model structures.
    Parameters
    ----------
    estimated_model
    true_model

    Returns
    -------

    """
    nodes = estimated_model.nodes()
    est_adj = nx.to_numpy_matrix(estimated_model.to_undirected(), nodelist=nodes, weight=None)
    true_adj = nx.to_numpy_matrix(true_model.to_undirected(), nodelist=nodes, weight=None)

    return f1_score(np.ravel(true_adj), np.ravel(est_adj))


def is_independent(X, Y, model):
    Zs = []
    return model.test_conditional_independence(X, Y, Zs, significance_level=0.05)

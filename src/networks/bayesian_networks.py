import bnlearn
from utils.plot import compare_network_plot
import pomegranate
import pandas as pd
import numpy as np


def structured_learning(data, dag_compare=None):
    """
    Perform structure learning on the data set,
    Also, plots differences between expert-DAG and the computed-DAG
    if compare arg set to True

    Parameters
    ----------
    data (pd.DataFrame): Datset containing root nodes, target and features to
    to perform structured learning on.
    dag_compare(Optional). DAG to compare network to. Defaults to None

    Returns
    -------

    """
    model = bnlearn.structure_learning.fit(data)
    if dag_compare is not None:
        compare_network_plot(model, dag_compare)

    return model

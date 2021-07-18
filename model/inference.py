from pgmpy.inference import VariableElimination


def infer_variable_cpd(model):
    """
    Compute probability distribution for a variable in
    the network, given a set of variables(evidence).
    Parameters
    ----------
    model

    Returns
    -------

    """
    infer = VariableElimination(model)
    g_dist = infer.query(["G"])
    return g_dist


def predict_values_from_new_data(model):
    """
    Predicting values from new data points is quite similar to computing the
    conditional probabilities. We need to query for the variable that we need to predict
    given all the other features. The only difference is that rather than getting
    the probabilitiy distribution we are interested in getting the most probable state of the variable.
    Parameters
    ----------
    model

    Returns
    -------

    """
    infer = VariableElimination(model)
    return infer.map_query(["G"])

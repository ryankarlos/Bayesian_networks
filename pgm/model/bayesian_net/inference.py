from pgmpy.inference import VariableElimination

from pgm.utils.logging_conf import get_logger

LOG = get_logger()


def infer_variable_cpd(model, variable: list, evidence: dict):
    """
    Compute probability distribution for a variable in
    the network, given a set of variables(evidence).
    Parameters
    ----------
    model
    variable
    evidence

    Returns
    -------

    """
    infer = VariableElimination(model)
    return infer.query(variables=variable, evidence=evidence)


def predict_values_from_new_data(model, variable: list):
    """
    Predicting values from new data points is quite similar to computing the
    conditional probabilities. We need to query for the variable that we need to
    predict given all the other features. The only difference is that rather
    than getting the probabilitiy distribution we are interested in getting
    the most probable state of the variable.
    Parameters
    ----------
    model
    variable

    Returns
    -------

    """
    infer = VariableElimination(model)
    return infer.map_query(variable)

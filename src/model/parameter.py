from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD

from ..utils.logging_conf import get_logger

LOG = get_logger()


def learn_model_parameters(
    model_struct,
    data,
    estimator: str,
    prior_type="BDeu",
    eq_sample_size=1000,
    get_cpd=None,
):
    if estimator == "MLE":
        model_struct.fit(data=data, estimator=MaximumLikelihoodEstimator)
    elif estimator == "BE":
        model_struct.fit(
            data=data,
            estimator=BayesianEstimator,
            prior_type=prior_type,
            equivalent_sample_size=eq_sample_size,
        )
    if get_cpd is not None:
        cpd = get_cpd_from_model(model_struct, get_cpd)

        return model_struct, cpd
    return model_struct


def get_cpd_from_model(model_struct, node):
    cpd = model_struct.get_cpds(node)
    return cpd


def define_cpd_from_scratch(variable_name, variable_card, values):
    return TabularCPD(
        variable=variable_name, variable_card=variable_card, values=values
    )


def add_cpd_to_model(model, *args):
    model.add_cpds(*args)
    return model

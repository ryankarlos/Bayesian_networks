from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
import pgmpy
from pgmpy.factors.discrete import TabularCPD


def learn_model_parameters(
    model_struct, data, estimator: str, prior_type="BDeu", eq_sample_size=1000
):
    if estimator == "MLE":
        model_struct.fit(data=data, estimator=MaximumLikelihoodEstimator)
        return model_struct
    elif estimator == "BE":
        model_struct.fit(
            data=data,
            estimator=BayesianEstimator,
            prior_type=prior_type,
            equivalent_sample_size=eq_sample_size,
        )
        return model_struct


def get_cpd_from_model(model_struct, node):
    cpd = model_struct.get_cpds(node)
    print(cpd)
    return cpd


def define_cpd_from_scratch(variable_name, variable_card, values):
    return TabularCPD(
        variable=variable_name, variable_card=variable_card, values=values
    )


def add_cpd_to_model(model, *args):
    model.add_cpds(*args)
    return model

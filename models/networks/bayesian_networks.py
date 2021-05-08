import pomegranate
from pgmpy.estimators import PC, BayesianEstimator, BDeuScore, BicScore, HillClimbSearch, K2Score
from pgmpy.models import BayesianModel
from pgmpy.sampling import BayesianModelSampling

from data.generate_sample_data import bayesian_network_datasets
from diagnostics.networks import get_f1_score, is_independent
from utils.logconfig import module_logger

LOG = module_logger()


def structured_learning(data, scoring=K2Score, algo=HillClimbSearch, expert_model=None, **kwargs):

    """
    Perform structure learning on the data set,
    Also, plots differences between expert-DAG and the computed-DAG
    if compare arg set to True
    Parameters
    ----------
    data (pd.DataFrame): Datset containing root nodes, target and features to
    to perform structured learning on.
    scoring
    scoring metric to use e.g. K2, BIC
    algo
    Algorithm used to carry out structured learning e.g. Hill Climbing, PC, Exhaustive Search
    expert_model
    expert model object to be used to compare structured learning and compute F1 score
    kwargs
    Other parameters to pass into the estimator
    Returns
    -------
    """
    scoring_method = scoring(data)
    est = algo(data)
    estimated_model = est.estimate(scoring_method, **kwargs)
    if expert_model is not None:
        LOG.info("F1-score for the model skeleton: ", get_f1_score(estimated_model, expert_model))
    print(scoring_method.score(estimated_model))
    print(estimated_model.edges())
    # print(is_independent(edges[0][0], edges[0][1], estimated_model))
    return estimated_model


def parameter_learning(data, model, estimator, prior_type="dirichlet"):
    est = estimator(model, data)
    est.get_parameters(prior_type=prior_type)
    return model


if __name__ == "__main__":
    df = bayesian_network_datasets(name="alarm", samples=1000)
    kwargs = {"max_iter": 1e5, "epsilon": 1e-4}
    LOG.info("Running structured learning")
    model = structured_learning(df, **kwargs)
    # model = parameter_learning(df, model, estimator=BayesianEstimator)

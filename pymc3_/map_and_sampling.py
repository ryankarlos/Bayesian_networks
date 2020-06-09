import pymc3 as pm


def map_estimation(model:pm.Model, method=None)->pm.find_MAP:
    if method is None:
        map = pm.find_MAP(model=model)
    else:
        map = pm.find_MAP(model=model, method=method)

    return map


def hmc_nuts(model:pm.Model, samples=500, step=None, start=None)->pm.sample:

    with model:
        # draw n posterior samples
        if step is None:
            trace = pm.samples(samples)
        else:
            trace = pm.samples(samples,step=step)

    return trace

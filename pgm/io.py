import pickle

import networkx as nx


def export_graph_gexf(G, path):
    nx.write_gexf(G, path)


def serialise_to_pickle(path, **estimators):
    for k, v in estimators.items():
        with open(f"{path}/{k}.pkl", "wb") as file:
            pickle.dump(v, file)


def load_from_pickle(path):
    with open(f"{path}", "rb") as file:
        model = pickle.load(file)
        return model

import networkx as nx
import pandas as pd


def read_network_from_data_csv(path):
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "source", df.columns[1]: "target"})
    Graphtype = nx.Graph()
    G = nx.from_pandas_edgelist(df, create_using=Graphtype)
    return G

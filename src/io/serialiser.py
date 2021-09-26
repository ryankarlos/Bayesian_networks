import networkx as nx


def export_graph(G, path):
    nx.write_gexf(G, path)

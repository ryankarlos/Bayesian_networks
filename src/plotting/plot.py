import networkx as nx
from pyvis.network import Network


def draw_network(G, ax, edge_list=None, color="red"):
    # draw from existing nx object
    if edge_list is not None:
        G = nx.Graph()
        G.add_edges_from(edge_list)  # using a list of edge tuples
        # pruned network after Max weighted spanning tree algo
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos=pos, node_color=color, ax=ax)
        print(nx.info(G))
        return G
    else:
        nx.draw_networkx(G, node_color=color)
        print(nx.info(G))
        return G


def plot_interactive_network(model, title):
    net = Network(notebook=True)
    net.from_nx(model)
    net.show_buttons(filter=["physics"])
    filename = f"{title}.html"
    net.show(filename)
    return net

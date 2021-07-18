import pandas as pd
from operator import itemgetter
import networkx as nx
from iteration_utilities import duplicates

from utils.log import get_logger

LOG = get_logger(__name__)


def get_edge_list(G, sort=True):
    if sort:
        # sort by node 1
        sorted_edges = sorted(G.edges(), key=lambda x: x[0], reverse=True)
        return sorted_edges
    else:
        return list(G.edges())


def list_duplicate_edges(G):
    dup = list(duplicates(get_edge_list(G)))
    if len(dup) > 0:
        LOG.info(f"{len(dup)} duplicates found")
    return dup


def remove_nodes_with_low_degree(G, n):
    degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    # keep top n nodes
    node_list_remove = [node for node, v in degree_sorted[n::]]
    G.remove_nodes_from(node_list_remove)
    return G


def set_attributes(G, **kwargs):
    for key in kwargs.keys():
        nx.set_node_attributes(G, kwargs[key], key)
    return G


def shortest_path(G, source_id, target_id):
    shortest_path = nx.shortest_path(G, source=source_id, target=target_id)
    LOG.info("Shortest path between user1 and user2:", shortest_path)
    return shortest_path


def top_betweeness_nodes_with_attributes(G, top_n=20):
    betweenness_dict = nx.get_node_attributes(G, "betweeness")
    sorted_betweenness = sorted(
        betweenness_dict.items(), key=itemgetter(1), reverse=True
    )
    top_betweenness = sorted_betweenness[:top_n]
    node_id = []
    betweenness_list = []

    for tb in top_betweenness:  # Loop through top_betweenness
        betweenness_list.append(tb[1])
        node_id.append(tb[0])
    return pd.DataFrame(
        {
            "id": node_id,
            "betweeness": betweenness_list,
        }
    )

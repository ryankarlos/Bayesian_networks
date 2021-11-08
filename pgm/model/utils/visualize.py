from pyvis.network import Network


def interactive_network_vis(
    dag, *widgets, options=None, weights=False, notebook=True, directed=True
):
    nt = Network("800px", "800px", directed=directed, notebook=notebook)

    nt.from_nx(dag)
    if weights:
        for edge in nt.edges:
            edge["value"] = edge["weight"]
        if options is not None:
            nt.set_options(options=options)
            return nt
        else:
            nt.show_buttons(filter=widgets)
            return nt

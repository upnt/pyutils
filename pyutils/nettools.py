import matplotlib.pyplot as plt
import networkx as nx


def collect_layout(graph: "nx.Graph"):
    pos = {}
    for node, data in graph.nodes(data=True):
        pos[node] = data["pos"]

    return pos


def collect_labels(graph: "nx.Graph"):
    labels = {}
    for node, data in graph.nodes(data=True):
        labels[node] = data["labels"]

    return labels


def collect_node_weight(graph: "nx.Graph"):
    return [abs(data["weight"]) for _, data in graph.nodes(data=True)]


def collect_edge_weight(graph: "nx.Graph"):
    return [abs(data["weight"]) for _, _, data in graph.edges(data=True)]


def draw(graph: "nx.Graph", node_size=500, border=100, width=5, with_labels=False):
    node_color = collect_node_weight(graph)
    edge_color = collect_edge_weight(graph)
    pos = collect_layout(graph)
    nx.draw_networkx_nodes(
        graph, node_size=node_size + border, pos=pos, node_color="black"
    )
    nx.draw_networkx_nodes(
        graph,
        node_size=node_size,
        pos=pos,
        node_color=node_color,
        cmap=plt.cm.Blues,
    )
    labels = collect_labels(graph)
    nx.draw_networkx_labels(graph, pos=pos, labels=labels)
    nx.draw_networkx_edges(
        graph, pos=pos, width=width, edge_color=edge_color, edge_cmap=plt.cm.Reds
    )


if __name__ == "__main__":
    main()

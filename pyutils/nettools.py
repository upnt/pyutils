import matplotlib
import matplotlib.pyplot as plt
import networkx as nx


def graph_minmax(graph: nx.Graph, param: str):
    it = iter(graph.nodes)
    min_value = max_value = graph.nodes[next(it)][param]

    for node in graph.nodes:
        if min_value > graph.nodes[node][param]:
            min_value = graph.nodes[node][param]

        if max_value < graph.nodes[node][param]:
            max_value = graph.nodes[node][param]

    return min_value, max_value


def lock_value(graph, node, value):
    for neighbor in graph.neighbors(node):
        graph.edges[node, neighbor]["weight"] *= value

    graph.remove_node(node)

    return graph


def edge(graph, in_node, out_node):
    graph.add_edge(in_node, out_node, weight=-1)
    return graph


def fold(graph, fold_node, folded_node):
    graph.nodes[fold_node]["weight"] += graph.nodes[folded_node]["weight"]
    for n in nx.all_neighbors(graph, folded_node):
        graph.add_edge(fold_node, n, weight=graph[folded_node][n]["weight"])
    graph.remove_node(folded_node)
    return graph


def move_graph(graph: "nx.Graph", displacement: tuple):
    for node in graph.nodes:
        graph.nodes[node]["pos"][0] += displacement[0]
        graph.nodes[node]["pos"][1] += displacement[1]

    return graph


def collect_layout(graph: "nx.Graph"):
    pos = {}
    for node, data in graph.nodes(data=True):
        pos[node] = data["pos"]

    return pos


def collect_color(graph: "nx.Graph"):
    pos = []
    for _, data in graph.nodes(data=True):
        try:
            pos.append(data["color"])
        except:
            pos.append("blue")

    return pos


def node_weight_gradation(graph: "nx.Graph", value, minus_color, plus_color):
    minus_colors = fix_gradation(minus_color, value + 1)
    plus_colors = fix_gradation(plus_color, value + 1)

    result = []
    for node in graph.nodes:
        weight = graph.nodes[node]["weight"]
        if weight > 0:
            result.append(plus_colors[weight])
        else:
            result.append(minus_colors[-weight])

    return result


def fix_gradation(start_color, length):
    rgb = matplotlib.colors.to_rgb(start_color)
    hsv = matplotlib.colors.rgb_to_hsv(rgb)
    hsv[1] = 1
    step = hsv[1] / (length - 1)

    result = []
    for i in range(length):
        hsv[1] = i * step
        result.append(matplotlib.colors.hsv_to_rgb(hsv))

    return result


def collect_labels(graph: "nx.Graph"):
    labels = {}
    for node, data in graph.nodes(data=True):
        labels[node] = data["labels"]

    return labels


def collect_node_weight(graph: "nx.Graph"):
    return [abs(data["weight"]) for _, data in graph.nodes(data=True)]


def collect_edge_weight(graph: "nx.Graph"):
    return [abs(data["weight"]) for _, _, data in graph.edges(data=True)]


def to_deep_color(color):
    pass


if __name__ == "__main__":
    graph = nx.Graph()
    graph.add_node(4, weight=-3)
    graph.add_node(9, weight=4)
    graph.add_node("15", weight=-7)
    graph.add_node("2", weight=9)
    graph.add_node("5", weight=0)
    for color in node_weight_gradation(graph, 10, "#e41a1c", "#377eb8"):
        print((color * 255).astype(int))

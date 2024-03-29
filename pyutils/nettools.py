import math
from collections import Counter, defaultdict
from typing import Any

import matplotlib
import networkx as nx
import numpy as np


def graph_minmax(graph: nx.Graph, param: str) -> tuple[float, float]:
    it = iter(graph.nodes)
    min_value = max_value = graph.nodes[next(it)][param]

    for node in graph.nodes:
        if min_value > graph.nodes[node][param]:
            min_value = graph.nodes[node][param]

        if max_value < graph.nodes[node][param]:
            max_value = graph.nodes[node][param]

    return min_value, max_value


def lock_value(graph: nx.Graph, node: Any, value: float) -> nx.Graph:
    for neighbor in graph.neighbors(node):
        graph.edges[node, neighbor]["weight"] *= value

    graph.remove_node(node)

    return graph


def edge(graph: nx.Graph, in_node: Any, out_node: Any) -> nx.Graph:
    graph.add_edge(in_node, out_node, weight=-1)
    return graph


def fold(graph: nx.Graph, fold_node: Any, folded_node: Any) -> nx.Graph:
    graph.nodes[fold_node]["weight"] += graph.nodes[folded_node]["weight"]
    for n in nx.all_neighbors(graph, folded_node):
        graph.add_edge(fold_node, n, weight=graph[folded_node][n]["weight"])
    graph.remove_node(folded_node)
    return graph


def move_graph(graph: nx.Graph, displacement: tuple[float, float]) -> nx.Graph:
    for node in graph.nodes:
        graph.nodes[node]["pos"][0] += displacement[0]
        graph.nodes[node]["pos"][1] += displacement[1]

    return graph


def scale_graph(graph: nx.Graph, scale: tuple[float, float]) -> nx.Graph:
    for node in graph.nodes:
        graph.nodes[node]["pos"][0] *= scale[0]
        graph.nodes[node]["pos"][1] *= scale[1]

    return graph


def rotate_graph(graph: nx.Graph, degree: float) -> nx.Graph:
    for node in graph.nodes:
        x = graph.nodes[node]["pos"][0]
        y = graph.nodes[node]["pos"][1]
        rad = math.radians(degree)
        graph.nodes[node]["pos"][0] = x * math.cos(rad) - y * math.sin(rad)
        graph.nodes[node]["pos"][1] = x * math.sin(rad) + y * math.cos(rad)

    return graph


def collect_layout(graph: nx.Graph) -> dict:
    pos = {}
    for node, data in graph.nodes(data=True):
        pos[node] = data["pos"]

    return pos


def collect_color(graph: nx.Graph) -> list:
    pos = []
    for _, data in graph.nodes(data=True):
        try:
            pos.append(data["color"])
        except:
            pos.append("blue")

    return pos


def node_weight_gradation(graph: nx.Graph, value: int, minus_color: int, plus_color: int) -> list:
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


def fix_gradation(start_color: int, length: int) -> list:
    rgb = matplotlib.colors.to_rgb(start_color)
    hsv = matplotlib.colors.rgb_to_hsv(rgb)
    hsv[1] = 1
    step = hsv[1] / (length - 1)

    result = []
    for i in range(length):
        hsv[1] = i * step
        result.append(matplotlib.colors.hsv_to_rgb(hsv))

    return result


def collect_labels(graph: nx.Graph) -> dict:
    labels = {}
    for node, data in graph.nodes(data=True):
        labels[node] = data["labels"]

    return labels


def collect_node_weight(graph: nx.Graph) -> list[float]:
    return [abs(data["weight"]) for _, data in graph.nodes(data=True)]


def collect_edge_weight(graph: nx.Graph) -> list[float]:
    return [abs(data["weight"]) for _, _, data in graph.edges(data=True)]


def convert_graph_to_expression(graph: nx.Graph, expr_type: type) -> Any:
    expr = 0
    for node in graph.nodes:
        expr += graph.nodes[node]["weight"] * expr_type(node)

    for edge in graph.edges:
        expr += graph.edges[edge]["weight"] * expr_type(edge[0]) * expr_type(edge[1])

    return expr


def _analize(items: Any, is_detail: bool = True) -> int:
    diags: defaultdict = defaultdict(int)
    diag_count = 0
    for item in items:
        weight = items[item]["weight"]
        if weight != 0:
            diags[weight] += 1
            diag_count += 1
    if is_detail:
        key, cnt = "key", "cnt"
        print(f"{key:>8} {cnt:>8}")
        for key, cnt in sorted(diags.items()):
            print(f"{key:>8} {cnt:>8}")

    return diag_count


def check_graph(graph: nx.Graph, is_detail: bool = True):
    print(f"var count: {len(graph.nodes)}")
    print("------------------------------")
    diag_count = _analize(graph.nodes, is_detail)
    print(f"1-d count: {diag_count}")
    print("------------------------------")
    not_diag_count = _analize(graph.edges, is_detail)
    print(f"2-d count: {not_diag_count}")
    print("------------------------------")
    degrees = list(dict(nx.degree(graph)).values())
    if is_detail:
        key, cnt = "key", "cnt"
        print(f"{key:>8} {cnt:>8}")
        for key, cnt in sorted(Counter(degrees).items()):
            print(f"{key:>8} {cnt:>8}")
    print(f"mean degree: {np.mean(degrees)}")
    print("------------------------------")
    E = graph.number_of_edges()
    N = graph.number_of_nodes()
    print(f"density: {(2 * E) / (N * (N - 1))}")
    print(f"all count: {diag_count + not_diag_count}")

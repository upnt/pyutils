import math
from collections import Counter, defaultdict
from pprint import pprint
from typing import Any, Callable, Dict, Self, Tuple

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import abc

class GateGraphFactory():
    def __init__(self: Self, node_dict: Dict[str, Dict[str, Any]], edge_dict: Dict[Tuple[str, str], Dict[str, Any]], offset: int):
        self._node_dict = node_dict
        self._edge_dict = edge_dict
        self._offset = offset
        self._id = 0

    @staticmethod
    def from_sympy() -> "GateGraphFactory":
        node_dict = {"a": {"a": 0}}
        edge_dict = {("a", "b"): {"a": 0}}
        offset = 0
        return GateGraphFactory(node_dict, edge_dict, offset)
        

    def generate(self: Self) -> nx.Graph:
        graph = nx.Graph()
        for key, val in self._node_dict.items():
            gkey = f"{key}_{self._id}"
            if gkey in graph.nodes:
                val['weight'] += graph.nodes[gkey]["weight"]
            graph.add_node(gkey, weight=val["weight"], pos=val["pos"], color=val["color"])
        for key, val in self._edge_dict.items():
            gkey = (f"{key[0]}_{self._id}", f"{key[1]}_{self._id}")
            if gkey in graph.edges:
                val['weight'] += graph.edges[gkey]["weight"]
            graph.add_edge(*gkey, weight=val["weight"])
        self._id += 1
        return graph

        #     def get_variable_dict(self, gen):
        #         q = gen.array(nx.number_of_nodes(self._graph))
        #         variables = {}
        #         for i, node in enumerate(self._graph.nodes):
        #             if "value" in self._graph.nodes[node]:
        #                 variables[node] = self._graph.nodes[node]["value"]
        #             else:
        #                 variables[node] = q[i]
        #         return variables
        # 
        #     def expression(self, variables: dict):
        #         expr = convert_graph_to_expression(self._graph, variables)
        #         return expr + self._offset


class Circuit:
    def __init__(self, map_shape: tuple):
        self._circuit_graph: nx.DiGraph = nx.DiGraph()
        self.pos_map: np.ndarray = np.full(map_shape, None)
        self.nodes: dict[str, str] = {}
        self.elems: dict[str, Element] = {}
        self._variable_funcs: dict[Element, Callable] = {}

    def get_variable_dict(self, gen):
        return {elm: func(gen) for elm, func in self._variable_funcs.items()}

    def expression(self, variables: dict):
        for edge in self._circuit_graph.edges:
            in_element = self._circuit_graph.nodes[edge[0]]["element"]
            out_element = self._circuit_graph.nodes[edge[1]]["element"]
            in_node = self._circuit_graph.edges[edge]["in_node"]
            out_node = self._circuit_graph.edges[edge]["out_node"]
            if out_element not in variables:
                print(f"key lengths: {len(variables.keys())}")
                raise KeyError(f"variables has no attribute out_element {out_element}")
            elif in_element not in variables:
                print(f"key lengths: {len(variables.keys())}")
                raise KeyError(f"variables has no attribute in_element {in_element}")
            variables[out_element][out_node] = variables[in_element][in_node]

        # pprint(variables)
        expr = 0
        for node in self._circuit_graph.nodes:
            element = self._circuit_graph.nodes[node]["element"]
            buf = element.expression(variables[element])
            # print(variables[element])
            # print(buf)
            expr += buf

        # pprint(variables)
        return expr

    def add_circuit(self, circuit: "Circuit", base: tuple):
        self._circuit_graph = nx.compose(self._circuit_graph, circuit._circuit_graph)
        self._variable_funcs.update(circuit._variable_funcs)
        width, height = circuit.pos_map.shape
        for i in range(width):
            for j in range(height):
                self.pos_map[i + base[0]][j + base[1]] = circuit.pos_map[i][j]

    def save_to_nodes(self, element, mapping: dict):
        if isinstance(element, Element):
            for key, before in mapping.items():
                self.nodes[key] = element.nodes[before]
                self.elems[key] = element
        elif isinstance(element, Circuit):
            for key, before in mapping.items():
                self.nodes[key] = element.nodes[before]
                self.elems[key] = element.elems[before]
        else:
            raise ValueError

    def append_input_element(self, element: Element):
        root = hash(element)
        self._circuit_graph.add_node(root, element=element)
        self._variable_funcs[element] = element.get_variable_dict

    def connect(self, in_element: Element, out_element: Element, in_node: str, out_node: str):
        in_hash = hash(in_element)
        out_hash = hash(out_element)
        if in_hash not in self._circuit_graph:
            self._circuit_graph.add_node(in_hash, element=in_element)
            self._variable_funcs[in_element] = in_element.get_variable_dict
        if out_hash not in self._circuit_graph:
            self._circuit_graph.add_node(out_hash, element=out_element)
            self._variable_funcs[out_element] = out_element.get_variable_dict
        self._circuit_graph.add_edge(in_hash, out_hash, in_node=in_node, out_node=out_node)

    def connects_from(
        self,
        in_circuit: "Circuit",
        out_circuit: "Circuit",
        in_nodes: list[str],
        out_nodes: list[str],
    ):
        for in_node, out_node in zip(in_nodes, out_nodes):
            in_element = in_circuit.elems[in_node]
            in_node = in_circuit.nodes[in_node]
            out_element = out_circuit.elems[out_node]
            out_node = out_circuit.nodes[out_node]
            self.connect(
                out_element,
                in_element,
                out_node,
                in_node,
            )

    def rot90(self, k: int = 1):
        self.pos_map = np.rot90(self.pos_map, k, axes=(1, 0))


def generate_graph(
    circuit: Circuit, connect_func: Callable, scale: tuple = (1, 1), gap: tuple = (0.5, 0.5)
) -> nx.Graph:
    graph = nx.Graph()
    width, height = circuit.pos_map.shape
    for i in range(width):
        for j in range(height):
            node = circuit.pos_map[i][j]
            if node is not None:
                buf = circuit._circuit_graph.nodes[hash(node)]["element"]._graph
                buf = move_graph(buf, (scale[0] * i * (1 + gap[0]), scale[1] * -j * (1 + gap[1])))
                graph = nx.compose(graph, buf)

    for edge in circuit._circuit_graph.edges:
        in_node = circuit._circuit_graph.edges[edge]["in_node"]
        out_node = circuit._circuit_graph.edges[edge]["out_node"]
        connect_func(graph, in_node, out_node)

    return graph


def generate_expression(circuit: Circuit, connect_type: str):
    width, height = circuit.pos_map.shape
    result = 0
    num_nodes = 0
    for i in range(width):
        for j in range(height):
            node = circuit.pos_map[i][j]
            if node is not None:
                buf = circuit._circuit_graph.nodes[hash(node)]["element"]._graph
                num_nodes += nx.number_of_nodes(buf)

    gen = BinarySymbolGenerator()
    q = gen.array(num_nodes)

    for i in range(width):
        for j in range(height):
            node = circuit.pos_map[i][j]
            if node is not None:
                buf = circuit._circuit_graph.nodes[hash(node)]["element"]._graph
                result += convert_graph_to_expression(buf, lambda x: q[i][j][x])

    for edge in circuit._circuit_graph.edges:
        in_node = circuit._circuit_graph.edges[edge]["in_node"]
        out_node = circuit._circuit_graph.edges[edge]["out_node"]
        in_node = circuit._circuit_graph.nodes[edge[0]]["element"].nodes[in_node]
        out_node = circuit._circuit_graph.nodes[edge[1]]["element"].nodes[out_node]
        out_graph = circuit._circuit_graph.nodes[edge[1]]["element"]._graph
        if connect_type == "fold":
            weight = out_graph.nodes[out_node]["weight"]
            result -= weight * expr_type(out_node)
            result += weight * expr_type(in_node)
            for n in nx.all_neighbors(out_graph, out_node):
                weight = out_graph.edges[out_node, n]["weight"]
                result -= weight * expr_type(out_node) * expr_type(n)
                result += weight * expr_type(in_node) * expr_type(n)
        elif connect_type == "connect":
            result += -expr_type(in_node) * expr_type(out_node)
        else:
            raise ValueError

    return result


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


def all_lock(graph: nx.Graph):
    for node in list(graph.nodes):
        if "value" in graph.nodes[node]:
            graph = lock_value(graph, node, graph.nodes[node]["value"])

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


def move_graph(graph: nx.Graph, displacement: Tuple[float, float]) -> nx.Graph:
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
        try:
            pos[node] = data["pos"]
        except KeyError as e:
            raise KeyError(f"node {node} has no 'pos'") from e
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


def convert_graph_to_expression(graph: nx.Graph, variables: dict) -> Any:
    expr = 0
    for node in graph.nodes:
        if "value" in graph.nodes[node]:
            variables[node] = graph.nodes[node]["value"]
        expr += graph.nodes[node]["weight"] * variables[node]

    for edge in graph.edges:
        expr += graph.edges[edge]["weight"] * variables[edge[0]] * variables[edge[1]]

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

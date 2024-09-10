import abc
import math
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Tuple

import amplify
import matplotlib
import networkx as nx
import numpy as np


class GateGraph:
    """
    論理ゲート用QUBO/Isingモデル
    """

    def __init__(
        self: "GateGraph", in_mapping: Dict[str, str], out_mapping: Dict[str, str], graph: nx.Graph
    ):
        self._in_mapping = in_mapping
        self._out_mapping = out_mapping
        self._graph = graph
        self._connection: Dict[str, List[Tuple["GateGraph", str]]] = {
            key: [] for key in out_mapping
        }

    def get_graph(self: "GateGraph") -> nx.Graph:
        """keyの値をvalueで固定する. 入力keyのみ設定可能

        Args:

        Returns:
            nx.Graph: 内部状態graphを出力する
        """
        return self._graph

    def bind_input(self: "GateGraph", key: str, value: int) -> None:
        """keyの値をvalueで固定する. 入力keyのみ設定可能

        Args:
            key (str): 入力するキー
            value (int): 固定したい値

        Returns:
            None: 内部状態graphのkeyに対応する値をvalueで固定する. 戻り値はNone

        Examples:
            >>> G = nx.Graph({"a0": {"weight": 0}, "b0": {"weight": 0}, "s0": {"weight": 0}})
            >>> gate = Gate({"a": "a0", "b": "b0"}, {"s": "s0"}, G)
            >>> gate.set_value("a", 4)
            >>> print(circuit.get_graph)
            nx.Graph({"a0": {"weight": 0, "value": 4}, "b0": {"weight": 0}, "s0": {"weight": 0}})
        """
        try:
            self._graph.nodes[self._in_mapping[key]]["value"] = value
        except KeyError as e:
            print(f"current in_mapping: {self._in_mapping}")
            raise e
        del self._in_mapping[key]

    def bind_output(self: "GateGraph", key: str, value: int) -> None:
        """keyの値をvalueで固定する. 出力keyのみ設定可能

        Args:
            key (str): 出力キー
            value (int): 固定したい値

        Returns:
            None: 内部状態graphのkeyに対応する値をvalueで固定する. 戻り値はNone

        Examples:
            >>> G = nx.Graph({"a0": {"weight": 0}, "b0": {"weight": 0}, "s0": {"weight": 0}})
            >>> gate = Gate({"a": "a0", "b": "b0"}, {"s": "s0"}, G)
            >>> gate.set_value("s", 4)
            >>> print(gate.get_graph())
            nx.Graph({"a0": {"weight": 0}, "b0": {"weight": 0}, "s0": {"weight": 0, "value": 4}})
        """
        try:
            self._graph.nodes[self._out_mapping[key]]["value"] = value
        except KeyError as e:
            print(f"current out_mapping: {self._out_mapping}")
            raise e
        del self._out_mapping[key]

    def get_input(self: "GateGraph", key: str) -> str:
        """入力keyのノード名を取得する

        Args:
            key (str): 入力キー

        Returns:
            str: 内部状態graphのkeyに対応するノード名

        Examples:
            >>> G = nx.Graph({"a0": {"weight": 0}, "b0": {"weight": 0}, "s0": {"weight": 0}})
            >>> gate = GateGraph({"a": "a0", "b": "b0"}, {"s": "s0"}, G)
            >>> gate.get_input("a")
            "a0"
        """
        try:
            return self._in_mapping[key]
        except KeyError as e:
            print(f"current in_mapping: {self._in_mapping}")
            raise e

    def get_output(self: "GateGraph", key: str) -> str:
        """出力keyのノード名を取得する

        Args:
            key (str): 出力キー

        Returns:
            str: 内部状態graphのkeyに対応するノード名

        Examples:
            >>> G = nx.Graph({"a0": {"weight": 0}, "b0": {"weight": 0}, "s0": {"weight": 0}})
            >>> gate = GateGraph({"a": "a0", "b": "b0"}, {"s": "s0"}, G)
            >>> gate.get_output("s")
            "s0"
        """
        try:
            return self._out_mapping[key]
        except KeyError as e:
            print(f"current out_mapping: {self._out_mapping}")
            raise e

    # class Circuit:
    #     def __init__(self: "Circuit"):
    #         self.nodes: dict[str, str] = {}
    #         self.elems: dict[str, Element] = {}
    #         self._variable_funcs: dict[Element, Callable] = {}
    #
    #     def get_variable_dict(self, gen):
    #         return {elm: func(gen) for elm, func in self._variable_funcs.items()}
    #
    #     def expression(self, variables: dict):
    #         for edge in self._circuit_graph.edges:
    #             in_element = self._circuit_graph.nodes[edge[0]]["element"]
    #             out_element = self._circuit_graph.nodes[edge[1]]["element"]
    #             in_node = self._circuit_graph.edges[edge]["in_node"]
    #             out_node = self._circuit_graph.edges[edge]["out_node"]
    #             if out_element not in variables:
    #                 print(f"key lengths: {len(variables.keys())}")
    #                 raise KeyError(f"variables has no attribute out_element {out_element}")
    #             elif in_element not in variables:
    #                 print(f"key lengths: {len(variables.keys())}")
    #                 raise KeyError(f"variables has no attribute in_element {in_element}")
    #             variables[out_element][out_node] = variables[in_element][in_node]
    #
    #         expr = 0
    #         for node in self._circuit_graph.nodes:
    #             element = self._circuit_graph.nodes[node]["element"]
    #             buf = element.expression(variables[element])
    #             # print(variables[element])
    #             # print(buf)
    #             expr += buf
    #
    #         return expr
    #
    #     def add_circuit(self, circuit: "Circuit", base: tuple):
    #         self._circuit_graph = nx.compose(self._circuit_graph, circuit._circuit_graph)
    #         self._variable_funcs.update(circuit._variable_funcs)
    #         width, height = circuit.pos_map.shape
    #         for i in range(width):
    #             for j in range(height):
    #                 self.pos_map[i + base[0]][j + base[1]] = circuit.pos_map[i][j]
    #
    #     def save_to_nodes(self, element, mapping: dict):
    #         if isinstance(element, Element):
    #             for key, before in mapping.items():
    #                 self.nodes[key] = element.nodes[before]
    #                 self.elems[key] = element
    #         elif isinstance(element, Circuit):
    #             for key, before in mapping.items():
    #                 self.nodes[key] = element.nodes[before]
    #                 self.elems[key] = element.elems[before]
    #         else:
    #             raise ValueError
    #
    #     def append_input_element(self, element: Element):
    #         root = hash(element)
    #         self._circuit_graph.add_node(root, element=element)
    #         self._variable_funcs[element] = element.get_variable_dict
    #
    #     def connect(self, in_element: Element, out_element: Element, in_node: str, out_node: str):
    #         in_hash = hash(in_element)
    #         out_hash = hash(out_element)
    #         if in_hash not in self._circuit_graph:
    #             self._circuit_graph.add_node(in_hash, element=in_element)
    #             self._variable_funcs[in_element] = in_element.get_variable_dict
    #         if out_hash not in self._circuit_graph:
    #             self._circuit_graph.add_node(out_hash, element=out_element)
    #             self._variable_funcs[out_element] = out_element.get_variable_dict
    #         self._circuit_graph.add_edge(in_hash, out_hash, in_node=in_node, out_node=out_node)
    #
    #     def connects_from(
    #         self,
    #         in_circuit: "Circuit",
    #         out_circuit: "Circuit",
    #         in_nodes: list[str],
    #         out_nodes: list[str],
    #     ):
    #         for in_node, out_node in zip(in_nodes, out_nodes):
    #             in_element = in_circuit.elems[in_node]
    #             in_node = in_circuit.nodes[in_node]
    #             out_element = out_circuit.elems[out_node]
    #             out_node = out_circuit.nodes[out_node]
    #             self.connect(
    #                 out_element,
    #                 in_element,
    #                 out_node,
    #                 in_node,
    #             )
    #
    #     def rot90(self, k: int = 1):
    #         self.pos_map = np.rot90(self.pos_map, k, axes=(1, 0))
    #


class GateGraphFactory:
    """論理ゲート用Ising/QUBOモデルのファクトリクラス

    GateGraphをidによってナンバリングすることで同一のキーを持つグラフの生成を抑制する
    """

    def __init__(
        self: "GateGraphFactory",
        node_dict: Dict[str, Dict[str, Any]],
        edge_dict: Dict[Tuple[str, str], Dict[str, Any]],
        in_nodes: List[str],
        out_nodes: List[str],
        offset: float,
    ):
        self._node_dict = node_dict
        self._edge_dict = edge_dict
        self._in_nodes = in_nodes
        self._out_nodes = out_nodes
        self._offset = offset
        self._id = 0

    @staticmethod
    def from_poly(
        poly: amplify.Poly, in_vars: List[amplify.Variable], out_vars: List[amplify.Variable]
    ) -> "GateGraphFactory":
        """Amplify.Polyによる二次式に基づいたファクトリ生成

        Examples:
            >>> gen = VariableGenerator()
            >>> x = gen.scalar("Binary", name="x")
            >>> y = gen.scalar("Binary", name="y")
            >>> z = gen.scalar("Binary", name="z")
            >>> s = gen.scalar("Binary", name="s")
            >>> poly = x * y + x * z + y * z - 2 * s * (x + y + z) + 3
            >>> factory = GateGraphFactory.from_poly(poly, [x, y, z], [s])
            >>> gate = factory.generate()
        """
        if poly.degree() > 2:
            raise ValueError(f"The degree of poly is higher than 2 (degree: {poly.degree()})")

        node_dict: Dict[str, Dict[str, Any]] = {}
        edge_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}
        offset = 0.0
        for var in poly.variables:
            node_dict[var.name] = {
                "weight": 0,
                "pos": [0, 0],
                "color": "blue",
            }
        for var, coeff in poly:
            match len(var):
                case 0:
                    offset += coeff
                case 1:
                    node_dict[var[0].name] = {
                        "weight": coeff,
                        "pos": (0, 0),
                        "color": "blue",
                    }
                case 2:
                    edge_dict[(var[0].name, var[1].name)] = {"weight": coeff}
                case _:
                    raise ValueError(f"poly is higher degree (degree: {poly.degree()})")

        n = len(node_dict)
        pos_list = [
            ((math.cos((i / n) * 2 * math.pi) - 1) / 2, (math.sin((i / n) * 2 * math.pi) - 1) / 2)
            for i in range(n)
        ]
        for pos, (key, val) in zip(pos_list, node_dict.items()):
            val["pos"] = pos
            node_dict[key] = val
        in_nodes = [var.name for var in in_vars]
        out_nodes = [var.name for var in out_vars]
        return GateGraphFactory(node_dict, edge_dict, in_nodes, out_nodes, offset)

    def generate(self: "GateGraphFactory") -> GateGraph:
        """
        __init__で与えられた情報からGateGraphインスタンスを生成する
        """
        graph = nx.Graph()
        mapping: Dict[str, str] = {}
        for key_node, val in self._node_dict.items():
            mapping[key_node] = f"{key_node}_{self._id}"
            if mapping[key_node] in graph.nodes:
                val["weight"] += graph.nodes[mapping[key_node]]["weight"]
            if "color" not in val:
                val["color"] = "blue"
            graph.add_node(
                mapping[key_node], weight=val["weight"], pos=val["pos"], color=val["color"]
            )

        for key_edge, val in self._edge_dict.items():
            new_key_edge = mapping[key_edge[0]], mapping[key_edge[1]]
            if new_key_edge in graph.edges:
                val["weight"] += graph.edges[new_key_edge]["weight"]
            graph.add_edge(*new_key_edge, weight=val["weight"])

        in_mapping: Dict[str, str] = {}
        out_mapping: Dict[str, str] = {}
        for node in self._in_nodes:
            in_mapping[node] = mapping[node]
        for node in self._out_nodes:
            out_mapping[node] = mapping[node]
        self._id += 1
        return GateGraph(in_mapping, out_mapping, graph)


def generate_graph(
    circuit_graph: nx.DiGraph,
    connect_func: Callable[[nx.Graph, str, str], None],
    gap: Tuple[float, float] = (0.5, 0.5),
) -> nx.Graph:
    """keyの値をvalueで固定する. 出力keyのみ設定可能

    Args:
        circuit_graph (nx.DiGraph): 回路の接続関係が書かれた有向グラフ
        connect_func (Callable[[nx.Graph, str, str]]): グラフの二点間を結ぶ関数
        gap (Tuple[float, float]): ゲートの同士の間隔

    Returns:
        nx.Graph: 接続関係に基づいて構築された論理回路を表現する無向グラフ

    Examples:
        >>> import networkx as nx
        >>> node_dict = {"a": {"weight": 1, "pos": (0, 0)}, "b": {"weight": 1, "pos": (0, 1)}, "s": {"weight": 2, "pos": (1, 1)}}
        >>> edge_dict={("a", "b"): {"weight": -1}}
        >>> factory = GateGraphFactory(node_dict, edge_dict, ["a", "b"], ["s"], 0)
        >>> circuit_graph = nx.DiGraph()
        >>> circuit_graph.add_node("GT0", gate=factory.generate(), pos=(0, 0))
        >>> circuit_graph.add_node("GT1", gate=factory.generate(), pos=(0, 1))
        >>> circuit_graph.add_node("GT2", gate=factory.generate(), pos=(1, 0))
        >>> circuit_graph.add_node("GT3", gate=factory.generate(), pos=(1, 1))
        >>> circuit_graph.add_edge("GT0", "GT3", from_key="s", to_key="a")
        >>> circuit_graph.add_edge("GT1", "GT2", from_key="s", to_key="b")
        >>> G = generate_graph(circuit_graph, fold)
        >>> G.nodes(data=True)
        NodeDataView({
            "a0": {"weight": 1, "pos": (0, 0)},     "b0": {"weight": 0, "pos": (0, 1)},     "s0": {"weight": 3, "pos": (1, 1)},
            "a1": {"weight": 1, "pos": (0, 1.5)},   "b1": {"weight": 0, "pos": (0, 2.5)},   "s1": {"weight": 3, "pos": (1, 2.5)},
            "a2": {"weight": 1, "pos": (1.5, 0)},                                           "s2": {"weight": 2, "pos": (2.5, 1)},
                                                    "b3": {"weight": 0, "pos": (1.5, 2.5)}, "s3": {"weight": 2, "pos": (2.5, 2.5)},
        })
        >>> G.edges(data=True)
        EdgeDataView({
            ("a0", "b0"): {"weight": -1}, ("a1", "b1"): {"weight": -1}, ("a2", "s1"): {"weight": -1}, ("s0", "b3"): {"weight": -1},
        })
    """
    graph = nx.Graph()

    for _, data in circuit_graph.nodes(data=True):
        buf: nx.Graph = data["gate"].get_graph()
        i = data["pos"][0]
        j = data["pos"][1]
        buf = move_graph(buf, (i * (1 + gap[0]), j * (1 + gap[1])))
        print(buf.nodes(data=True))
        graph = nx.compose(graph, buf)
    print(graph.nodes(data=True))

    for from_node, to_node, data in circuit_graph.edges(data=True):
        from_node = circuit_graph.nodes[from_node]["gate"].get_output(data["from_key"])
        to_node = circuit_graph.nodes[to_node]["gate"].get_input(data["to_key"])
        connect_func(graph, from_node, to_node)

    return graph


def generate_expression(circuit: GateGraph, connect_type: str):
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


def graph_param_minmax(graph: nx.Graph, param: str) -> Tuple[float, float]:
    """graphのparamの最大・最小値を同時に求める

    Args:
        graph (nx.Graph): 探索対象のグラフ
        param (str): 探索したいパラメータ

    Returns:
        min_value (float): 最小値
        max_value (float): 最大値

    Examples:
        >>> graph = nx.Graph()
        >>> graph.add_node("a", weight=0, pos=(0, 0))
        >>> graph.add_node("b", weight=-2, pos=(0, 1), color="green")
        >>> graph.add_node("c", weight=5, pos=(1, 0), color="red")
        >>> graph_param_minmax(graph, "weight")
        (-2, 5)
    """
    it = iter(graph.nodes)
    min_value = max_value = graph.nodes[next(it)][param]

    for node in graph.nodes:
        if min_value > graph.nodes[node][param]:
            min_value = graph.nodes[node][param]

        if max_value < graph.nodes[node][param]:
            max_value = graph.nodes[node][param]

    return min_value, max_value


def edge(graph: nx.Graph, from_node: Any, to_node: Any) -> nx.Graph:
    """graphのfrom_nodeとto_nodeを重み-1で接続する

    2つのGateグラフがgraph上に配置されているとき、2つのGateグラフをつなぐ辺の重みは-1で十分である。
    """
    graph.add_edge(from_node, to_node, weight=-1)
    return graph


def fold(graph: nx.Graph, fold_node: Any, folded_node: Any) -> nx.Graph:
    """graphのfrom_nodeとto_nodeを1つの頂点に圧縮する"""
    graph.nodes[fold_node]["weight"] += graph.nodes[folded_node]["weight"]
    for n in nx.all_neighbors(graph, folded_node):
        graph.add_edge(fold_node, n, weight=graph[folded_node][n]["weight"])
    graph.remove_node(folded_node)
    return graph


def move_graph(graph: nx.Graph, displacement: Tuple[float, float]) -> nx.Graph:
    """graphを並行移動する

    Args:
        graph (nx.Graph): 移動したいグラフ
        displacement (Tuple[float, float]): 移動距離

    Returns:
        graph (nx.Graph): 移動後のグラフ
    """
    for node in graph.nodes:
        i = graph.nodes[node]["pos"][0]
        i += displacement[0]
        j = graph.nodes[node]["pos"][1]
        j += displacement[1]
        graph.nodes[node]["pos"] = (i, j)

    return graph


def scale_graph(graph: nx.Graph, scale: Tuple[float, float]) -> nx.Graph:
    """graphを拡大・縮小する

    Args:
        graph (nx.Graph): 拡大・縮小したいグラフ
        scale (Tuple[float, float]): x, y方向の拡大率・縮小率

    Returns:
        graph (nx.Graph): 拡大・縮小後のグラフ
    """
    for node in graph.nodes:
        graph.nodes[node]["pos"][0] *= scale[0]
        graph.nodes[node]["pos"][1] *= scale[1]

    return graph


def rotate_graph(graph: nx.Graph, degree: float) -> nx.Graph:
    """graphを回転する

    Args:
        graph (nx.Graph): 回転したいグラフ
        degree (float): 角度(rad)

    Returns:
        graph (nx.Graph): 回転後のグラフ
    """
    for node in graph.nodes:
        x = graph.nodes[node]["pos"][0]
        y = graph.nodes[node]["pos"][1]
        rad = math.radians(degree)
        graph.nodes[node]["pos"][0] = x * math.cos(rad) - y * math.sin(rad)
        graph.nodes[node]["pos"][1] = x * math.sin(rad) + y * math.cos(rad)

    return graph


def collect_labels(graph: nx.Graph) -> Dict[str, str]:
    """graphのlabel情報を収集しnode_labelに変換する"""
    labels = {}
    for node, data in graph.nodes(data=True):
        labels[node] = data["labels"]

    return labels


def collect_node_weight(graph: nx.Graph) -> List[float]:
    """graphの頂点の重み絶対値を収集しnode_labelに変換する"""
    return [abs(data["weight"]) for _, data in graph.nodes(data=True)]


def collect_edge_weight(graph: nx.Graph) -> List[float]:
    """graphの辺の重み絶対値を収集しedge_labelに変換する"""
    return [abs(data["weight"]) for _, _, data in graph.edges(data=True)]


def collect_layout(graph: nx.Graph) -> Dict[str, Tuple[int, int]]:
    """graphのpos情報を収集しlayoutに変換する

    Args:
        graph (nx.Graph): 入力グラフ

    Returns:
        dict: {node: (x, y)}

    Examples:
        >>> graph = nx.Graph()
        >>> graph.add_node("a", weight=0, pos=(0, 0))
        >>> graph.add_node("b", weight=0, pos=(0, 1), color="green")
        >>> graph.add_node("c", weight=0, pos=(1, 0), color="red")
        >>> pos = collect_layout(graph)
        {"a": (0, 0), "b": (0, 1), "c": (1, 0)}
        >>> nx.draw_networkx(graph, pos=pos)
    """
    pos = {}
    for node, data in graph.nodes(data=True):
        try:
            pos[node] = data["pos"]
        except KeyError as e:
            raise KeyError(f"node {node} has no 'pos'") from e
    return pos


def collect_color(graph: nx.Graph) -> List[str]:
    """graphのcolor情報を収集しnode_colorに変換する

    Args:
        graph (nx.Graph): 入力グラフ

    Returns:
        List[str]: ノードの色リスト

    Examples:
        >>> import networkx as nx
        >>> import matplotlib.pyplot as plt
        >>> graph = nx.Graph()
        >>> graph.add_node("a", weight=0, pos=(0, 0))
        >>> graph.add_node("b", weight=0, pos=(0, 1), color="green")
        >>> graph.add_node("c", weight=0, pos=(1, 0), color="red")
        >>> node_color = collect_layout(graph)
        ["blue", "green", "red"]
        >>> nx.draw_networkx(graph, node_color=node_color)
        >>> plt.show()
    """
    pos = []
    for _, data in graph.nodes(data=True):
        try:
            pos.append(data["color"])
        except:
            pos.append("blue")

    return pos


# TODO: remove value parameter
def node_weight_gradation(
    graph: nx.Graph, value: int, minus_color: int | str, plus_color: int | str
) -> List[int]:
    """graphのweight情報をもとにnode_colorを生成する

    Args:
        graph (nx.Graph): 入力グラフ
        value (int): 色の分割数. valueはすべての重みの総数にしてください
        minus_color (int | str): 最大絶対値の負数の色
        minus_color (int | str): 最大絶対値の正数の色

    Returns:
        List[int]: ノードの色リスト

    Examples:
        >>> import networkx as nx
        >>> import matplotlib.pyplot as plt
        >>> graph = nx.Graph()
        >>> graph.add_node("a", weight=0, pos=(0, 0))
        >>> graph.add_node("b", weight=-2, pos=(0, 1), color="green")
        >>> graph.add_node("c", weight=5, pos=(1, 0), color="red")
        >>> node_color = node_weight_gradation(graph, minus_color="blue", plus_color="red")
        ["blue", "green", "red"]
        >>> nx.draw_networkx(graph, node_color=node_color)
        >>> plt.show()
    """
    minus_colors = _fix_gradation(minus_color, value + 1)
    plus_colors = _fix_gradation(plus_color, value + 1)

    result = []
    for node in graph.nodes:
        weight = graph.nodes[node]["weight"]
        if weight > 0:
            result.append(plus_colors[weight])
        else:
            result.append(minus_colors[-weight])

    return result


def _fix_gradation(start_color: int | str, length: int) -> List[int]:
    rgb = matplotlib.colors.to_rgb(start_color)
    hsv = matplotlib.colors.rgb_to_hsv(rgb)
    hsv[1] = 1
    step = hsv[1] / (length - 1)

    result = []
    for i in range(length):
        hsv[1] = i * step
        result.append(matplotlib.colors.hsv_to_rgb(hsv))

    return result


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
        print(f"{'key':>8} {'cnt':>8}")
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
        print(f"{'key':>8} {'cnt':>8}")
        for key, cnt in sorted(Counter(degrees).items()):
            print(f"{key:>8} {cnt:>8}")
    print(f"mean degree: {np.mean(degrees)}")
    print("------------------------------")
    E = graph.number_of_edges()
    N = graph.number_of_nodes()
    print(f"density: {(2 * E) / (N * (N - 1))}")
    print(f"all count: {diag_count + not_diag_count}")

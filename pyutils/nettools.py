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
        self: "GateGraph",
        graph: nx.Graph,
        offset: int | float,
    ):
        self._graph = graph
        self._offset = offset

    @classmethod
    def from_circuit(
        cls,
        circuit: nx.DiGraph,
    ) -> "GateGraph":
        graph, offset = generate_graph(circuit, fold)
        return cls(graph, offset)

    @classmethod
    def from_dict(
        cls,
        node_dict: Dict[str, Dict[str, Any]],
        edge_dict: Dict[Tuple[str, str], Dict[str, Any]],
        in_nodes: List[str],
        out_nodes: List[str],
        offset: int | float,
    ):
        """
        dict形式で書かれた情報からGateGraphインスタンスを生成する
        """
        graph = nx.Graph()
        mapping: Dict[str, str] = {}
        for key_node, val in node_dict.items():
            mapping[key_node] = key_node
            if mapping[key_node] in graph.nodes:
                val["weight"] += graph.nodes[mapping[key_node]]["weight"]
            if "color" not in val:
                val["color"] = "blue"
            graph.add_node(
                mapping[key_node], weight=val["weight"], pos=val["pos"], color=val["color"]
            )

        for key_edge, val in edge_dict.items():
            new_key_edge = mapping[key_edge[0]], mapping[key_edge[1]]
            if new_key_edge in graph.edges:
                val["weight"] += graph.edges[new_key_edge]["weight"]
            graph.add_edge(*new_key_edge, weight=val["weight"])

        in_mapping: Dict[str, str] = {}
        out_mapping: Dict[str, str] = {}
        for node in in_nodes:
            in_mapping[node] = mapping[node]
        for node in out_nodes:
            out_mapping[node] = mapping[node]
        return cls(graph, offset)

    @classmethod
    def from_poly(
        cls, poly: amplify.Poly, in_vars: List[amplify.Variable], out_vars: List[amplify.Variable]
    ) -> "GateGraph":
        """Amplify.Polyによる二次式に基づいたファクトリ生成

        Examples:
            >>> gen = VariableGenerator()
            >>> x = gen.scalar("Binary", name="x")
            >>> y = gen.scalar("Binary", name="y")
            >>> z = gen.scalar("Binary", name="z")
            >>> s = gen.scalar("Binary", name="s")
            >>> poly = x * y + x * z + y * z - 2 * s * (x + y + z) + 3
            >>> gate = GateGraph.from_poly(poly, [x, y, z], [s])
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
            ((math.cos((i / n) * 2 * math.pi) + 1) / 2, (math.sin((i / n) * 2 * math.pi) + 1) / 2)
            for i in range(n)
        ]
        for pos, (key, val) in zip(pos_list, node_dict.items()):
            val["pos"] = pos
            node_dict[key] = val
        in_nodes = [var.name for var in in_vars]
        out_nodes = [var.name for var in out_vars]
        return cls.from_dict(node_dict, edge_dict, in_nodes, out_nodes, offset)

    @classmethod
    def symbol(
        cls,
        var: amplify.Variable,
    ) -> "GateGraph":
        """頂点生成用のファクトリ

        Examples:
            >>> gen = VariableGenerator()
            >>> x = gen.scalar("Binary", name="x")
            >>> gate = GateGraph.symbol_factory(x)
        """
        node_dict: Dict[str, Dict[str, Any]] = {}
        edge_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}
        offset = 0.0
        node_dict[var.name] = {
            "weight": 0,
            "pos": [0, 0],
            "color": "blue",
        }
        in_nodes = [var.name]
        out_nodes = [var.name]
        return cls.from_dict(node_dict, edge_dict, in_nodes, out_nodes, offset)

    def get_graph(self: "GateGraph") -> nx.Graph:
        """内部graphを出力する

        Args:

        Returns:
            nx.Graph: 内部状態graphを出力する
        """
        return self._graph

    def get_offset(self: "GateGraph") -> int | float:
        """内部offsetを出力する

        Args:

        Returns:
            nx.Graph: 内部状態offsetを出力する
        """
        return self._offset

    def bind(self: "GateGraph", key: str, value: int) -> None:
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
            self._graph.nodes[key]["value"] = value
        except KeyError as e:
            print(f"current graph: {self._graph.nodes}")
            raise e


def generate_graph(
    circuit_graph: nx.DiGraph,
    connect_func: Callable[[nx.Graph, str, str], nx.Graph],
    gap: Tuple[float, float] = (0.5, 0.5),
) -> Tuple[nx.Graph, int | float]:
    """接続関係を表す有向グラフから完全な重み付き無向グラフを生成する

    Args:
        circuit_graph (nx.DiGraph): 回路の接続関係が書かれた有向グラフ
        connect_func (Callable[[nx.Graph, str, str], nx.Graph]): グラフの二点間を結ぶ関数
        gap (Tuple[float, float]): ゲートの同士の間隔

    Returns:
        graph (nx.Graph): 接続関係に基づいて構築された論理回路を表現する無向グラフ
        offset (int | float): オフセット

    Examples:
        >>> import networkx as nx
        >>> node_dict = {"a": {"weight": 1, "pos": (0, 0)}, "b": {"weight": 1, "pos": (0, 1)}, "s": {"weight": 2, "pos": (1, 1)}}
        >>> edge_dict={("a", "b"): {"weight": -1}}
        >>> gate = GateGraph(node_dict, edge_dict, ["a", "b"], ["s"], 4)
        >>> circuit_graph = nx.DiGraph()
        >>> circuit_graph.add_node("GT0", gate=copy(gate), pos=(0, 0))
        >>> circuit_graph.add_node("GT1", gate=copy(gate), pos=(0, 1))
        >>> circuit_graph.add_node("GT2", gate=copy(gate), pos=(1, 0))
        >>> circuit_graph.add_node("GT3", gate=copy(gate), pos=(1, 1))
        >>> circuit_graph.add_edge("GT0", "GT3", from_key="s", to_key="a")
        >>> circuit_graph.add_edge("GT1", "GT2", from_key="s", to_key="b")
        >>> G, offset = generate_graph(circuit_graph, fold)
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
        >>> offset
        4
    """
    graph = nx.Graph()
    offset = 0

    for node, data in circuit_graph.nodes(data=True):
        buf: nx.Graph = data["gate"].get_graph()
        buf = nx.relabel_nodes(buf, {n: f"{node}_{n}" for n in buf.nodes})
        offset += data["gate"].get_offset()
        i = data["pos"][0]
        j = data["pos"][1]
        buf = move_graph(buf, (i * (1 + gap[0]), j * (1 + gap[1])))
        graph = nx.compose(graph, buf)

    for from_node, to_node, data in circuit_graph.edges(data=True):
        from_node = f"{from_node}_{data['from_key']}"
        to_node = f"{to_node}_{data['to_key']}"
        try:
            connect_func(graph, from_node, to_node)
        except KeyError as e:
            print(graph.nodes())
            raise e

    return graph, offset


def generate_expression(
    graph: nx.Graph,
    vartype: amplify.VariableType,
    offset: int | float = 0,
) -> Tuple[Dict[str, amplify.Poly], int | float | amplify.Poly]:
    """無向グラフからAmplify.PolyによるQUBO/Isingモデルを生成する

    Args:
        graph (nx.Graph): 重み付き無向グラフ
        vartype (str): Amplify.Variableの型

    Returns:
        nx.Graph: 接続関係に基づいて構築された論理回路を表現する無向グラフ

    Examples:
        >>> import networkx as nx
        >>> node_dict = {"a": {"weight": 1, "pos": (0, 0)}, "b": {"weight": 1, "pos": (0, 1)}, "s": {"weight": 2, "pos": (1, 1)}}
        >>> edge_dict={("a", "b"): {"weight": -1}}
        >>> gate = GateGraph(node_dict, edge_dict, ["a", "b"], ["s"], 0)
        >>> circuit_graph = nx.DiGraph()
        >>> circuit_graph.add_node("GT0", gate=copy(gate), pos=(0, 0))
        >>> circuit_graph.add_node("GT1", gate=copy(gate), pos=(0, 1))
        >>> circuit_graph.add_node("GT2", gate=copy(gate), pos=(1, 0))
        >>> circuit_graph.add_node("GT3", gate=copy(gate), pos=(1, 1))
        >>> circuit_graph.add_edge("GT0", "GT3", from_key="s", to_key="a")
        >>> circuit_graph.add_edge("GT1", "GT2", from_key="s", to_key="b")
        >>> G = generate_graph(circuit_graph, fold)
        >>> generate_expression(G, "Binary")
        a0 + a1 + a2 + 3 * s0 + 3 * s1 + 2 * s2 + 2 * s3 - a0 * b0 - a1 * b1 - a2 * s1 - b3 * s0
    """
    poly = 0
    variables: Dict[str, amplify.Poly] = {}
    gen = amplify.VariableGenerator()

    for node, data in graph.nodes(data=True):
        if "value" in data:
            variables[node] = data["value"]
        else:
            variables[node] = gen.scalar(vartype, name=node)
        poly += data["weight"] * variables[node]

    for from_node, to_node, data in graph.edges(data=True):
        poly += data["weight"] * variables[from_node] * variables[to_node]
    return variables, poly + offset


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


def fold(graph: nx.Graph, from_node: Any, to_node: Any) -> nx.Graph:
    """graphのfrom_nodeとto_nodeを1つの頂点に圧縮する"""
    graph.nodes[from_node]["weight"] += graph.nodes[to_node]["weight"]
    for n in nx.all_neighbors(graph, to_node):
        graph.add_edge(from_node, n, weight=graph[to_node][n]["weight"])
    graph.remove_node(to_node)
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

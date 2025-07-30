import bidict
import pytest
import rustworkx as rx

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import IndexedRXGraph


def test_index_rx_graph_with_mapping() -> None:
    index_map = {1: 0, 5_000: 1, 3: 2}
    attrs = [
        {DEFAULT_ATTR_KEYS.T: 0, "a": 1},
        {DEFAULT_ATTR_KEYS.T: 1, "a": 2},
        {DEFAULT_ATTR_KEYS.T: 2, "a": 3},
    ]
    rx_graph = rx.PyDiGraph()
    rx_graph.add_nodes_from(attrs)
    rx_graph.add_edges_from([(0, 1, {}), (1, 2, {})])

    graph = IndexedRXGraph(
        rx_graph=rx_graph,
        node_id_map=index_map,
    )

    assert graph.node_ids() == [1, 5_000, 3]
    assert set(graph.node_attr_keys) == {DEFAULT_ATTR_KEYS.T, "a"}


def test_duplicate_index_map() -> None:
    index_map = {0: 1, 1: 5_000, 2: 3, 3: 1}

    attrs = [
        {DEFAULT_ATTR_KEYS.T: 0, "a": 1, "b": 1},
        {DEFAULT_ATTR_KEYS.T: 1, "a": 2, "b": 2},
        {DEFAULT_ATTR_KEYS.T: 2, "a": 3, "b": 3},
        {DEFAULT_ATTR_KEYS.T: 3, "a": 4, "b": 4},
    ]

    rx_graph = rx.PyDiGraph()
    rx_graph.add_nodes_from(attrs)
    rx_graph.add_edges_from([(0, 1, {}), (1, 2, {}), (2, 3, {}), (3, 0, {})])

    with pytest.raises(bidict.ValueDuplicationError):
        IndexedRXGraph(rx_graph=rx_graph, node_id_map=index_map)

    graph = IndexedRXGraph()

    graph.add_node({"t": 0}, index=3)

    with pytest.raises(bidict.KeyDuplicationError):
        graph.add_node({"t": 5}, index=3)

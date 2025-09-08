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
    assert set(graph.node_attr_keys) == {DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.T, "a"}


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


def test_add_node_with_none_index_avoids_collision() -> None:
    """Test that add_node with index=None uses next available ID to avoid collision."""
    graph = IndexedRXGraph()

    # Add nodes with specific indices
    idx1 = graph.add_node({"t": 0}, index=2)
    idx2 = graph.add_node({"t": 1}, index=10)

    assert idx1 == 2
    assert idx2 == 10

    # Add nodes with index=None - should get next available IDs
    # Since max existing external ID is 10, next should be 11
    idx3 = graph.add_node({"t": 2}, index=None)
    idx4 = graph.add_node({"t": 3}, index=None)

    assert idx3 == 11
    assert idx4 == 12

    # Test bulk_add_nodes with all explicit indices
    bulk_nodes = [{"t": 4}, {"t": 5}, {"t": 6}]
    bulk_indices = [30, 31, 32]  # All explicit
    bulk_result = graph.bulk_add_nodes(bulk_nodes, indices=bulk_indices)

    # Should get exactly what we specified
    assert bulk_result == [30, 31, 32]

    # Verify all nodes exist and are accessible
    node_ids = graph.node_ids()
    assert set(node_ids) == {2, 10, 11, 12, 30, 31, 32}

    # Verify node attributes can be retrieved
    node_attrs = graph.node_attrs()
    assert len(node_attrs) == 7

    # Test adding another explicit index works
    idx5 = graph.add_node({"t": 7}, index=20)
    assert idx5 == 20

    # Test that next None index continues from the max (should be 33)
    idx6 = graph.add_node({"t": 8}, index=None)
    assert idx6 == 33


def test_add_node_none_index_empty_graph() -> None:
    """Test that add_node with index=None works correctly on empty graph."""

    # test with empty graph and empty node_id_map
    graph = IndexedRXGraph(rx_graph=rx.PyDiGraph(), node_id_map={})

    # test with empty graph and no node_id_map
    graph = IndexedRXGraph()

    # First node with None index should get ID 0
    idx1 = graph.add_node({"t": 0}, index=None)
    assert idx1 == 0

    # Second node with None index should get ID 1
    idx2 = graph.add_node({"t": 1}, index=None)
    assert idx2 == 1

    # Test bulk_add_nodes with all None indices on existing graph
    bulk_nodes = [{"t": 2}, {"t": 3}, {"t": 4}]
    bulk_result = graph.bulk_add_nodes(bulk_nodes, indices=None)  # None means use defaults

    # Should get [2, 3, 4] - continuing from counter
    assert bulk_result == [2, 3, 4]

    # Verify all nodes exist
    node_ids = graph.node_ids()
    assert set(node_ids) == {0, 1, 2, 3, 4}


def test_counter_updates_with_explicit_high_indices() -> None:
    """Test that counter updates correctly when explicit high indices are added."""
    graph = IndexedRXGraph()

    # Add some nodes with None index (should get 0, 1)
    idx1 = graph.add_node({"t": 0}, index=None)
    idx2 = graph.add_node({"t": 1}, index=None)
    assert idx1 == 0
    assert idx2 == 1

    # Add a node with explicit high index
    idx3 = graph.add_node({"t": 2}, index=100)
    assert idx3 == 100

    # Next None index should be 101 (counter updated)
    idx4 = graph.add_node({"t": 3}, index=None)
    assert idx4 == 101

    # Test bulk_add_nodes with all explicit indices including high values
    bulk_nodes = [{"t": 4}, {"t": 5}, {"t": 6}, {"t": 7}]
    bulk_indices = [50, 200, 201, 202]  # All explicit, with high values
    bulk_result = graph.bulk_add_nodes(bulk_nodes, indices=bulk_indices)

    # Should get exactly what we specified
    assert bulk_result == [50, 200, 201, 202]

    # Next None index should be 203 (counter updated to max + 1)
    idx5 = graph.add_node({"t": 8}, index=None)
    assert idx5 == 203


def test_indexed_rx_graph_initialization_with_existing_mapping() -> None:
    """Test that counter initializes correctly when graph is created with existing mapping."""
    import rustworkx as rx

    attrs = [
        {"t": 0, "a": 1},
        {"t": 1, "a": 2},
        {"t": 2, "a": 3},
    ]
    rx_graph = rx.PyDiGraph()
    rx_graph.add_nodes_from(attrs)

    # External IDs: 5, 100, 25 (mapping {external: local})
    node_id_map = {5: 0, 100: 1, 25: 2}

    graph = IndexedRXGraph(rx_graph=rx_graph, node_id_map=node_id_map)

    # Counter should start after max existing external ID (100)
    idx1 = graph.add_node({"t": 3, "a": 4}, index=None)
    assert idx1 == 101

    idx2 = graph.add_node({"t": 4, "a": 5}, index=None)
    assert idx2 == 102

    # Test bulk_add_nodes with all None indices (auto-generated)
    bulk_nodes_auto = [{"t": 5, "a": 6}, {"t": 6, "a": 7}, {"t": 7, "a": 8}]
    bulk_result_auto = graph.bulk_add_nodes(bulk_nodes_auto, indices=None)

    # Should get [103, 104, 105] - continuing from counter
    assert bulk_result_auto == [103, 104, 105]

    # Test bulk_add_nodes with all explicit indices
    bulk_nodes_explicit = [{"t": 8, "a": 9}]
    bulk_indices_explicit = [200]  # All explicit, high value
    bulk_result_explicit = graph.bulk_add_nodes(bulk_nodes_explicit, indices=bulk_indices_explicit)

    # Should get exactly what we specified
    assert bulk_result_explicit == [200]

    # Verify all nodes exist in the mapping
    all_node_ids = graph.node_ids()
    assert set(all_node_ids) == {5, 100, 25, 101, 102, 103, 104, 105, 200}

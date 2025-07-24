import pytest

from tracksdata.graph import RustWorkXGraph
from tracksdata.graph.filters._spatial_filter import SpatialFilter


@pytest.fixture
def sample_graph() -> RustWorkXGraph:
    """Create a sample graph with nodes for testing."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("z", 0)
    graph.add_node_attr_key("y", 0)
    graph.add_node_attr_key("x", 0)

    # Add some nodes with spatial coordinates
    nodes = [
        {"t": 0, "z": 0, "y": 10, "x": 20},
        {"t": 0, "z": 0, "y": 30, "x": 40},
        {"t": 1, "z": 1, "y": 50, "x": 60},
        {"t": 2, "z": 2, "y": 90, "x": 100},
    ]

    for node_attrs in nodes:
        graph.add_node(node_attrs)

    return graph


def test_spatial_filter_initialization(sample_graph: RustWorkXGraph) -> None:
    """Test SpatialFilter initialization with default and custom attributes."""
    # Test default attributes
    spatial_filter = SpatialFilter(sample_graph)
    assert spatial_filter._df_filter._attrs_keys == ["t", "z", "y", "x"]
    assert spatial_filter._df_filter._sg_graph is not None

    # Test custom attributes
    custom_attrs = ["t", "y", "x"]
    spatial_filter = SpatialFilter(sample_graph, attrs_keys=custom_attrs)
    assert spatial_filter._df_filter._attrs_keys == custom_attrs


def test_spatial_filter_querying(sample_graph: RustWorkXGraph) -> None:
    """Test spatial querying with different bounds and dimensions."""
    spatial_filter = SpatialFilter(sample_graph)

    # Test valid bounds that include nodes
    result = spatial_filter[0:2, 0:2, 0:100, 0:100]
    node_attrs = result.node_attrs()
    assert not node_attrs.is_empty()

    # Test narrow bounds that exclude most nodes
    result = spatial_filter[0:1, 0:1, 0:15, 0:25]
    node_attrs = result.node_attrs()
    assert len(node_attrs) <= 1

    # Test bounds that exclude all nodes
    result = spatial_filter[10:20, 10:20, 200:300, 200:300]
    node_attrs = result.node_attrs()
    assert node_attrs.is_empty()


def test_spatial_filter_dimensions() -> None:
    """Test SpatialFilter with different coordinate dimensions."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("z", 0)
    graph.add_node_attr_key("y", 0)
    graph.add_node_attr_key("x", 0)
    graph.add_node({"t": 0, "z": 0, "y": 10, "x": 20})

    # Test 2D coordinates
    spatial_filter_2d = SpatialFilter(graph, attrs_keys=["y", "x"])
    assert spatial_filter_2d._df_filter._attrs_keys == ["y", "x"]
    result = spatial_filter_2d[0:50, 0:50]
    assert not result.node_attrs().is_empty()

    # Test 3D coordinates
    spatial_filter_3d = SpatialFilter(graph, attrs_keys=["z", "y", "x"])
    assert spatial_filter_3d._df_filter._attrs_keys == ["z", "y", "x"]
    result = spatial_filter_3d[0:2, 0:100, 0:100]
    assert not result.node_attrs().is_empty()


def test_spatial_filter_error_handling(sample_graph: RustWorkXGraph) -> None:
    """Test error handling for invalid slice counts."""
    spatial_filter = SpatialFilter(sample_graph)

    # Test wrong number of slices
    with pytest.raises(ValueError, match="Expected 4 keys, got 3"):
        spatial_filter[0:2, 0:1, 0:50]


def test_spatial_filter_with_edges() -> None:
    """Test SpatialFilter preserves edges in subgraphs."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("y", 0)
    graph.add_node_attr_key("x", 0)
    graph.add_edge_attr_key("weight", 0.0)

    # Add nodes and edge
    node1_id = graph.add_node({"t": 0, "y": 10, "x": 20})
    node2_id = graph.add_node({"t": 1, "y": 30, "x": 40})
    graph.add_edge(node1_id, node2_id, {"weight": 1.0})

    spatial_filter = SpatialFilter(graph, attrs_keys=["y", "x"])
    result = spatial_filter[0:50, 0:50]

    # Should preserve both nodes and the edge
    assert len(result.node_attrs()) == 2
    assert len(result.edge_attrs()) == 1

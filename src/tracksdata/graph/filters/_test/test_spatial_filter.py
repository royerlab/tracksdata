import pytest

from tracksdata.graph import RustWorkXGraph
from tracksdata.graph.filters._spatial_filter import BBoxSpatialFilter, SpatialFilter


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


@pytest.fixture
def sample_bbox_graph() -> RustWorkXGraph:
    """Create a sample graph with nodes for bounding box testing."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("bbox", [0, 0, 0, 0, 0, 0])

    # Add some nodes with bounding box coordinates
    nodes = [
        {"t": 0, "bbox": [0, 10, 20, 1, 15, 25]},
        {"t": 0, "bbox": [0, 30, 40, 1, 35, 45]},
        {"t": 1, "bbox": [1, 50, 60, 2, 55, 65]},
        {"t": 2, "bbox": [2, 90, 100, 3, 95, 105]},
    ]

    for node_attrs in nodes:
        graph.add_node(node_attrs)

    return graph


def test_spatial_filter_initialization(sample_graph: RustWorkXGraph) -> None:
    """Test SpatialFilter initialization with default and custom attributes."""
    # Test default attributes
    spatial_filter = SpatialFilter(sample_graph)
    assert spatial_filter._df_filter._attr_keys == ["t", "z", "y", "x"]

    # Test custom attributes
    custom_attrs = ["t", "y", "x"]
    spatial_filter = SpatialFilter(sample_graph, attr_keys=custom_attrs)
    assert spatial_filter._df_filter._attr_keys == custom_attrs


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
    spatial_filter_2d = SpatialFilter(graph, attr_keys=["y", "x"])
    assert spatial_filter_2d._df_filter._attr_keys == ["y", "x"]
    result = spatial_filter_2d[0:50, 0:50]
    assert not result.node_attrs().is_empty()

    # Test 3D coordinates
    spatial_filter_3d = SpatialFilter(graph, attr_keys=["z", "y", "x"])
    assert spatial_filter_3d._df_filter._attr_keys == ["z", "y", "x"]
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

    spatial_filter = SpatialFilter(graph, attr_keys=["y", "x"])
    result = spatial_filter[0:50, 0:50]

    # Should preserve both nodes and the edge
    assert len(result.node_attrs()) == 2
    assert len(result.edge_attrs()) == 1


def test_bbox_spatial_filter_overlaps() -> None:
    """Test BoundingBoxSpatialFilter overlaps with existing nodes."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("bbox", [0, 0, 0, 0])
    # Add nodes with bounding boxes
    bboxes = [
        [0, 20, 10, 30],  # Node 1
        [5, 25, 15, 35],  # Node 2
        [10, 30, 20, 40],  # Node 3
        [15, 35, 25, 45],  # Node 4
    ]
    node_ids = graph.bulk_add_nodes([{"t": 0, "bbox": bbox} for bbox in bboxes])

    spatial_filter = BBoxSpatialFilter(graph, frame_attr_key="t", bbox_attr_key="bbox")
    result = spatial_filter[0:0, 15:20, 0:40]
    assert set(result.node_ids()) == set(node_ids[1:])
    result = spatial_filter[0:0, 15:20, 36:40]
    assert set(result.node_ids()) == set(node_ids[2:])

    # Should return both nodes
    assert len(result.node_attrs()) == 2


def test_bbox_spatial_filter_with_edges() -> None:
    """Test SpatialFilter preserves edges in subgraphs."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("bbox", [0, 0, 0, 0])
    graph.add_edge_attr_key("weight", 0.0)

    # Add nodes and edge
    node1_id = graph.add_node({"t": 0, "bbox": [10, 20, 15, 25]})
    node2_id = graph.add_node({"t": 1, "bbox": [30, 40, 35, 45]})
    graph.add_edge(node1_id, node2_id, {"weight": 1.0})

    spatial_filter = BBoxSpatialFilter(graph, frame_attr_key="t", bbox_attr_key="bbox")
    result = spatial_filter[0:1, 0:50, 0:50]

    # Should preserve both nodes and the edge
    assert len(result.node_attrs()) == 2
    assert len(result.edge_attrs()) == 1


def test_bbox_spatial_filter_initialization(sample_bbox_graph: RustWorkXGraph) -> None:
    """Test BoundingBoxSpatialFilter initialization with default and custom attributes."""
    spatial_filter = BBoxSpatialFilter(sample_bbox_graph)
    assert spatial_filter._node_rtree is not None

    spatial_filter = BBoxSpatialFilter(sample_bbox_graph, frame_attr_key="t", bbox_attr_key="bbox")
    assert spatial_filter._node_rtree is not None


def test_bbox_spatial_filter_querying(sample_bbox_graph: RustWorkXGraph) -> None:
    """Test bounding box spatial querying with different bounds and dimensions."""
    spatial_filter = BBoxSpatialFilter(sample_bbox_graph, frame_attr_key="t", bbox_attr_key="bbox")

    # Test valid bounds that include nodes
    result = spatial_filter[0:3, 0:3, 0:100, 0:110]
    node_attrs = result.node_attrs()
    assert not node_attrs.is_empty()

    # Test narrow bounds that should include specific nodes
    result = spatial_filter[0:1, 0:2, 10:20, 20:30]
    node_attrs = result.node_attrs()
    assert len(node_attrs) >= 1
    assert len(node_attrs) < sample_bbox_graph.num_nodes

    # Test bounds that exclude all nodes
    result = spatial_filter[10:20, 10:20, 200:300, 200:300]
    node_attrs = result.node_attrs()
    assert node_attrs.is_empty()


def test_bbox_spatial_filter_dimensions() -> None:
    """Test BoundingBoxSpatialFilter with different coordinate dimensions."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("bbox", [0, 0, 0, 0, 0, 0])
    graph.add_node({"t": 0, "bbox": [0, 10, 20, 1, 15, 25]})

    # Test 3D coordinates
    spatial_filter = BBoxSpatialFilter(graph, frame_attr_key="t", bbox_attr_key="bbox")
    result = spatial_filter[0:2, 0:50, 0:50, 0:50]
    assert not result.node_attrs().is_empty()

    with pytest.raises(ValueError, match="Expected 4 keys, got 3"):
        # Test wrong number of slices
        spatial_filter[0:2, 0:50, 0:50]

    # Only spatial coordinates
    spatial_filter = BBoxSpatialFilter(graph, frame_attr_key=None, bbox_attr_key="bbox")
    result = spatial_filter[0:50, 0:50, 0:50]
    assert not result.node_attrs().is_empty()


def test_bbox_spatial_filter_error_handling() -> None:
    """Test error handling for mismatched min/max attribute lengths."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("bbox", [0, 0, 0, 0])
    graph.add_node({"t": 0, "bbox": [10, 20, 15, 25, 14]})
    # Test mismatched min/max attributes length
    with pytest.raises(ValueError, match="Bounding box coordinates must have even number of dimensions"):
        BBoxSpatialFilter(graph, frame_attr_key="t", bbox_attr_key="bbox")

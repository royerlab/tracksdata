import numpy as np
import pytest

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges import IoUEdgeAttr
from tracksdata.graph import RustWorkXGraph
from tracksdata.nodes import Mask
from tracksdata.options import get_options, options_context


def test_iou_edges_init_default() -> None:
    """Test IoUEdgesOperator initialization with default parameters."""
    operator = IoUEdgeAttr(output_key="iou_score")

    assert operator.output_key == "iou_score"
    assert operator.attr_keys == DEFAULT_ATTR_KEYS.MASK
    assert operator.func == Mask.iou


def test_iou_edges_init_custom() -> None:
    """Test IoUEdgesOperator initialization with custom parameters."""
    operator = IoUEdgeAttr(output_key="custom_iou", mask_key="custom_mask")

    assert operator.output_key == "custom_iou"
    assert operator.attr_keys == "custom_mask"
    assert operator.func == Mask.iou


@pytest.mark.parametrize("n_workers", [1, 2])
def test_iou_edges_add_weights(n_workers: int) -> None:
    """Test adding IoU weights to edges with different worker counts."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_WEIGHT, 0.0)

    # Create test masks
    mask1_data = np.array([[True, True], [True, False]], dtype=bool)
    mask1 = Mask(mask1_data, bbox=np.array([0, 0, 2, 2]))

    mask2_data = np.array([[True, False], [False, False]], dtype=bool)
    mask2 = Mask(mask2_data, bbox=np.array([0, 0, 2, 2]))

    mask3_data = np.array([[True, True], [True, True]], dtype=bool)
    mask3 = Mask(mask3_data, bbox=np.array([0, 0, 2, 2]))

    # Add nodes with masks
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask1})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, DEFAULT_ATTR_KEYS.MASK: mask2})
    node3 = graph.add_node({DEFAULT_ATTR_KEYS.T: 2, DEFAULT_ATTR_KEYS.MASK: mask3})

    # Add edge
    edge_id_1 = graph.add_edge(node1, node2, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: 0.0})
    edge_id_2 = graph.add_edge(node2, node3, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: 0.0})

    # Create operator and add weights
    operator = IoUEdgeAttr(output_key="iou_score")
    with options_context(n_workers=n_workers):
        operator.add_edge_attrs(graph)

    # Check that IoU weights were added
    edges_df = graph.edge_attrs()
    assert "iou_score" in edges_df.columns

    # checking default returned columns
    assert DEFAULT_ATTR_KEYS.EDGE_SOURCE in edges_df.columns
    assert DEFAULT_ATTR_KEYS.EDGE_TARGET in edges_df.columns
    assert DEFAULT_ATTR_KEYS.EDGE_ID in edges_df.columns
    assert len(edges_df) == 2

    # Calculate expected IoU: intersection = 1, union = 3, IoU = 1/3
    expected_iou_1 = 1.0 / 3.0
    expected_iou_2 = 1.0 / 4.0
    edge_iou = dict(zip(edges_df[DEFAULT_ATTR_KEYS.EDGE_ID], edges_df["iou_score"], strict=True))
    assert abs(edge_iou[edge_id_1] - expected_iou_1) < 1e-6
    assert abs(edge_iou[edge_id_2] - expected_iou_2) < 1e-6


def test_iou_edges_no_overlap() -> None:
    """Test IoU calculation with non-overlapping masks."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_WEIGHT, 0.0)

    # Create non-overlapping masks
    mask1_data = np.array([[True, True], [False, False]], dtype=bool)
    mask1 = Mask(mask1_data, bbox=np.array([0, 0, 2, 2]))

    mask2_data = np.array([[False, False], [True, True]], dtype=bool)
    mask2 = Mask(mask2_data, bbox=np.array([0, 0, 2, 2]))

    # Add nodes with masks
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask1})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask2})

    # Add edge
    edge_id = graph.add_edge(node1, node2, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: 0.0})

    # Create operator and add weights
    operator = IoUEdgeAttr(output_key="iou_score")
    operator.add_edge_attrs(graph)

    # Check that IoU is 0 for non-overlapping masks
    edges_df = graph.edge_attrs()

    # checking default returned columns
    assert DEFAULT_ATTR_KEYS.EDGE_SOURCE in edges_df.columns
    assert DEFAULT_ATTR_KEYS.EDGE_TARGET in edges_df.columns
    assert DEFAULT_ATTR_KEYS.EDGE_ID in edges_df.columns
    assert len(edges_df) == 1

    edge_iou = dict(zip(edges_df[DEFAULT_ATTR_KEYS.EDGE_ID], edges_df["iou_score"], strict=False))
    assert edge_iou[edge_id] == 0.0


def test_iou_edges_perfect_overlap() -> None:
    """Test IoU calculation with perfectly overlapping masks."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_WEIGHT, 0.0)

    # Create identical masks
    mask_data = np.array([[True, True], [True, False]], dtype=bool)
    mask1 = Mask(mask_data, bbox=np.array([0, 0, 2, 2]))
    mask2 = Mask(mask_data, bbox=np.array([0, 0, 2, 2]))

    # Add nodes with masks
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask1})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask2})

    # Add edge
    edge_id = graph.add_edge(node1, node2, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: 0.0})

    # Create operator and add weights
    operator = IoUEdgeAttr(output_key="iou_score")
    operator.add_edge_attrs(graph)

    edges_df = graph.edge_attrs()

    # checking default returned columns
    assert DEFAULT_ATTR_KEYS.EDGE_SOURCE in edges_df.columns
    assert DEFAULT_ATTR_KEYS.EDGE_TARGET in edges_df.columns
    assert DEFAULT_ATTR_KEYS.EDGE_ID in edges_df.columns
    assert len(edges_df) == 1

    edge_iou = dict(zip(edges_df[DEFAULT_ATTR_KEYS.EDGE_ID], edges_df["iou_score"], strict=False))
    assert edge_iou[edge_id] == 1.0


def test_iou_edges_custom_mask_key() -> None:
    """Test IoU edges operator with custom mask key."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("custom_mask", None)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_WEIGHT, 0.0)

    # Create test masks
    mask1_data = np.array([[True, True], [True, True]], dtype=bool)
    mask1 = Mask(mask1_data, bbox=np.array([0, 0, 2, 2]))

    mask2_data = np.array([[True, True], [False, False]], dtype=bool)
    mask2 = Mask(mask2_data, bbox=np.array([0, 0, 2, 2]))

    # Add nodes with custom mask key
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "custom_mask": mask1})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "custom_mask": mask2})

    # Add edge
    edge_id = graph.add_edge(node1, node2, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: 0.0})

    # Create operator with custom mask key
    operator = IoUEdgeAttr(output_key="iou_score", mask_key="custom_mask")
    operator.add_edge_attrs(graph)

    # Check that IoU weights were calculated
    edges_df = graph.edge_attrs()
    assert "iou_score" in edges_df.columns

    # checking default returned columns
    assert DEFAULT_ATTR_KEYS.EDGE_SOURCE in edges_df.columns
    assert DEFAULT_ATTR_KEYS.EDGE_TARGET in edges_df.columns
    assert DEFAULT_ATTR_KEYS.EDGE_ID in edges_df.columns
    assert len(edges_df) == 1

    # Expected IoU: intersection = 2, union = 4, IoU = 0.5
    expected_iou = 0.5
    edge_iou = dict(zip(edges_df[DEFAULT_ATTR_KEYS.EDGE_ID], edges_df["iou_score"], strict=False))
    assert abs(edge_iou[edge_id] - expected_iou) < 1e-6


def test_iou_edges_multiprocessing_isolation() -> None:
    """Test that multiprocessing options don't affect subsequent tests."""
    # Verify default n_workers is 1
    assert get_options().n_workers == 1

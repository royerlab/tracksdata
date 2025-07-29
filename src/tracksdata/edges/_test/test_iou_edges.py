import numpy as np
import pytest

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges import IoUEdgeAttr
from tracksdata.graph import RustWorkXGraph
from tracksdata.options import get_options, options_context
from tracksdata.utils._test_utils import (
    setup_custom_node_attr,
    setup_edge_distance_attr,
    setup_mask_attrs,
)


def test_iou_edges_init_default() -> None:
    """Test IoUEdgesOperator initialization with default parameters."""
    operator = IoUEdgeAttr(output_key="iou_score")

    assert operator.output_key == "iou_score"
    assert operator.attr_keys == [DEFAULT_ATTR_KEYS.MASK, DEFAULT_ATTR_KEYS.BBOX]


def test_iou_edges_init_custom() -> None:
    """Test IoUEdgesOperator initialization with custom parameters."""
    operator = IoUEdgeAttr(output_key="custom_iou", mask_key="custom_mask", bbox_key="custom_bbox")

    assert operator.output_key == "custom_iou"
    assert operator.attr_keys == ["custom_mask", "custom_bbox"]


@pytest.mark.parametrize("n_workers", [1, 2])
def test_iou_edges_add_weights(n_workers: int) -> None:
    """Test adding IoU weights to edges with different worker counts."""
    graph = RustWorkXGraph()

    # Set up graph attributes
    setup_mask_attrs(graph)
    setup_edge_distance_attr(graph)

    # Create test masks and bboxes
    mask1_data = np.array([[True, True], [True, False]], dtype=bool)
    bbox1 = np.array([0, 0, 2, 2])

    mask2_data = np.array([[True, False], [False, False]], dtype=bool)
    bbox2 = np.array([0, 0, 2, 2])

    mask3_data = np.array([[True, True], [True, True]], dtype=bool)
    bbox3 = np.array([0, 0, 2, 2])

    # Add nodes with masks and bboxes
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask1_data, DEFAULT_ATTR_KEYS.BBOX: bbox1})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, DEFAULT_ATTR_KEYS.MASK: mask2_data, DEFAULT_ATTR_KEYS.BBOX: bbox2})
    node3 = graph.add_node({DEFAULT_ATTR_KEYS.T: 2, DEFAULT_ATTR_KEYS.MASK: mask3_data, DEFAULT_ATTR_KEYS.BBOX: bbox3})

    # Add edge
    edge_id_1 = graph.add_edge(node1, node2, {DEFAULT_ATTR_KEYS.EDGE_DIST: 0.0})
    edge_id_2 = graph.add_edge(node2, node3, {DEFAULT_ATTR_KEYS.EDGE_DIST: 0.0})

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

    # Set up graph attributes
    setup_mask_attrs(graph)
    setup_edge_distance_attr(graph)

    # Create non-overlapping masks
    mask1_data = np.array([[True, True], [False, False]], dtype=bool)
    bbox1 = np.array([0, 0, 2, 2])

    mask2_data = np.array([[False, False], [True, True]], dtype=bool)
    bbox2 = np.array([0, 0, 2, 2])

    # Add nodes with masks and bboxes
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask1_data, DEFAULT_ATTR_KEYS.BBOX: bbox1})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask2_data, DEFAULT_ATTR_KEYS.BBOX: bbox2})

    # Add edge
    edge_id = graph.add_edge(node1, node2, {DEFAULT_ATTR_KEYS.EDGE_DIST: 0.0})

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

    # Set up graph attributes
    setup_mask_attrs(graph)
    setup_edge_distance_attr(graph)

    # Create identical masks
    mask_data = np.array([[True, True], [True, False]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])

    # Add nodes with masks and bboxes
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask_data, DEFAULT_ATTR_KEYS.BBOX: bbox})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask_data, DEFAULT_ATTR_KEYS.BBOX: bbox})

    # Add edge
    edge_id = graph.add_edge(node1, node2, {DEFAULT_ATTR_KEYS.EDGE_DIST: 0.0})

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

    # Set up custom attributes
    setup_custom_node_attr(graph, "custom_mask", None)
    setup_custom_node_attr(graph, "custom_bbox", None)
    setup_edge_distance_attr(graph)

    # Create test masks
    mask1_data = np.array([[True, True], [True, True]], dtype=bool)
    bbox1 = np.array([0, 0, 2, 2])

    mask2_data = np.array([[True, True], [False, False]], dtype=bool)
    bbox2 = np.array([0, 0, 2, 2])

    # Add nodes with custom mask and bbox keys
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "custom_mask": mask1_data, "custom_bbox": bbox1})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "custom_mask": mask2_data, "custom_bbox": bbox2})

    # Add edge
    edge_id = graph.add_edge(node1, node2, {DEFAULT_ATTR_KEYS.EDGE_DIST: 0.0})

    # Create operator with custom mask and bbox keys
    operator = IoUEdgeAttr(output_key="iou_score", mask_key="custom_mask", bbox_key="custom_bbox")
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

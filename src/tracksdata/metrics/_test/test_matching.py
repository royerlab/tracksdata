"""Tests for matching strategies."""

import numpy as np
import polars as pl
import pytest

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph
from tracksdata.metrics import DistanceMatching, MaskMatching
from tracksdata.nodes._mask import Mask


def create_2d_mask_from_coords(coords: list[tuple[int, int]]) -> Mask:
    """Create a 2D Mask object from a list of (y, x) coordinates."""
    if not coords:
        return Mask(np.array([[False]]), np.array([0, 0, 1, 1]))

    y_coords, x_coords = [c[0] for c in coords], [c[1] for c in coords]
    y_min, y_max = min(y_coords), max(y_coords)
    x_min, x_max = min(x_coords), max(x_coords)

    mask_arr = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=bool)
    for y, x in coords:
        mask_arr[y - y_min, x - x_min] = True

    return Mask(mask_arr, np.array([y_min, x_min, y_max + 1, x_max + 1]))


class TestMaskMatching:
    """Tests for MaskMatching strategy."""

    def test_initialization_and_validation(self):
        """Test initialization with valid and invalid parameters."""
        # Valid initialization
        matching = MaskMatching()
        assert matching.min_reference_intersection == 0.5
        assert matching.optimal is True

        matching = MaskMatching(min_reference_intersection=0.7, optimal=False)
        assert matching.min_reference_intersection == 0.7
        assert matching.optimal is False

        # Invalid threshold
        with pytest.raises(ValueError, match="min_reference_intersection must be between 0 and 1"):
            MaskMatching(min_reference_intersection=1.5)

    def test_compute_weights_various_overlaps(self):
        """Test compute_weights with perfect, partial, and no overlap scenarios."""
        graph1 = RustWorkXGraph()
        graph2 = RustWorkXGraph()

        default_mask = Mask(np.array([[False]]), np.array([0, 0, 1, 1]))
        graph1.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, dtype=pl.Object, default_value=default_mask)
        graph2.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, dtype=pl.Object, default_value=default_mask)

        # Test 1: Perfect overlap (IoU = 1.0)
        mask_perfect = create_2d_mask_from_coords([(0, 0), (0, 1), (0, 2)])
        graph1.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask_perfect})
        graph2.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask_perfect})

        ref_group = graph1.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.MASK])
        comp_group = graph2.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.MASK])

        matching = MaskMatching(min_reference_intersection=0.5)
        mapped_ref, _, _, _, weights = matching.compute_weights(
            ref_group, comp_group, DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.NODE_ID
        )

        assert len(mapped_ref) == 1
        assert weights[0] == 1.0

        # Test 2: Partial overlap (3/5 intersection, IoU = 3/7)
        graph1 = RustWorkXGraph()
        graph2 = RustWorkXGraph()
        graph1.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, dtype=pl.Object, default_value=default_mask)
        graph2.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, dtype=pl.Object, default_value=default_mask)

        mask1 = create_2d_mask_from_coords([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)])
        mask2 = create_2d_mask_from_coords([(0, 2), (0, 3), (0, 4), (0, 5), (0, 6)])

        graph1.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask1})
        graph2.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask2})

        ref_group = graph1.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.MASK])
        comp_group = graph2.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.MASK])

        mapped_ref, _, _, _, weights = matching.compute_weights(
            ref_group, comp_group, DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.NODE_ID
        )

        assert len(mapped_ref) == 1
        assert np.isclose(weights[0], 3.0 / 7.0, atol=1e-6)

        # Test 3: Below threshold (1/5 = 0.2 < 0.5) - should not match
        graph1 = RustWorkXGraph()
        graph2 = RustWorkXGraph()
        graph1.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, dtype=pl.Object, default_value=default_mask)
        graph2.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, dtype=pl.Object, default_value=default_mask)

        mask1 = create_2d_mask_from_coords([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)])
        mask2 = create_2d_mask_from_coords([(0, 4), (0, 5), (0, 6), (0, 7), (0, 8)])

        graph1.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask1})
        graph2.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask2})

        ref_group = graph1.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.MASK])
        comp_group = graph2.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.MASK])

        mapped_ref, _, _, _, _ = matching.compute_weights(
            ref_group, comp_group, DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.NODE_ID
        )

        assert len(mapped_ref) == 0


class TestDistanceMatching:
    """Tests for DistanceMatching strategy."""

    def test_initialization(self):
        """Test initialization with various parameters."""
        matching = DistanceMatching(max_distance=10.0)
        assert matching.max_distance == 10.0
        assert matching.optimal is True

        matching = DistanceMatching(max_distance=5.0, optimal=False, attr_keys=("y", "x"), scale=(2.0, 1.0))
        assert matching.max_distance == 5.0
        assert matching.optimal is False
        assert matching.attr_keys == ("y", "x")
        assert matching.scale == (2.0, 1.0)

    def test_compute_weights_2d_and_3d(self):
        """Test distance matching in 2D and 3D, including threshold behavior."""
        # Test 2D: Close nodes should match, far nodes should not
        graph1 = RustWorkXGraph()
        graph2 = RustWorkXGraph()

        graph1.add_node_attr_key("y", dtype=pl.Float64)
        graph1.add_node_attr_key("x", dtype=pl.Float64)
        graph2.add_node_attr_key("y", dtype=pl.Float64)
        graph2.add_node_attr_key("x", dtype=pl.Float64)

        # Close nodes (distance ≈ 1.414)
        node1 = graph1.add_node({DEFAULT_ATTR_KEYS.T: 0, "y": 10.0, "x": 10.0})
        graph2.add_node({DEFAULT_ATTR_KEYS.T: 0, "y": 11.0, "x": 11.0})

        ref_group = graph1.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, "y", "x"])
        comp_group = graph2.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, "y", "x"])

        matching = DistanceMatching(max_distance=3.0, attr_keys=("y", "x"))
        mapped_ref, _, _, _, weights = matching.compute_weights(
            ref_group, comp_group, DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.NODE_ID
        )

        assert len(mapped_ref) == 1
        assert 0.3 < weights[0] < 0.5  # 1/(1+sqrt(2)) ≈ 0.414

        # Far nodes (distance ≈ 141.4)
        graph1 = RustWorkXGraph()
        graph2 = RustWorkXGraph()
        graph1.add_node_attr_key("y", dtype=pl.Float64)
        graph1.add_node_attr_key("x", dtype=pl.Float64)
        graph2.add_node_attr_key("y", dtype=pl.Float64)
        graph2.add_node_attr_key("x", dtype=pl.Float64)

        graph1.add_node({DEFAULT_ATTR_KEYS.T: 0, "y": 0.0, "x": 0.0})
        graph2.add_node({DEFAULT_ATTR_KEYS.T: 0, "y": 100.0, "x": 100.0})

        ref_group = graph1.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, "y", "x"])
        comp_group = graph2.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, "y", "x"])

        matching = DistanceMatching(max_distance=10.0, attr_keys=("y", "x"))
        mapped_ref, _, _, _, _ = matching.compute_weights(
            ref_group, comp_group, DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.NODE_ID
        )

        assert len(mapped_ref) == 0

        # Test 3D
        graph1 = RustWorkXGraph()
        graph2 = RustWorkXGraph()

        for key in ["z", "y", "x"]:
            graph1.add_node_attr_key(key, dtype=pl.Float64)
            graph2.add_node_attr_key(key, dtype=pl.Float64)

        node1 = graph1.add_node({DEFAULT_ATTR_KEYS.T: 0, "z": 5.0, "y": 10.0, "x": 15.0})
        graph2.add_node({DEFAULT_ATTR_KEYS.T: 0, "z": 6.0, "y": 11.0, "x": 16.0})

        ref_group = graph1.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, "z", "y", "x"])
        comp_group = graph2.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, "z", "y", "x"])

        matching = DistanceMatching(max_distance=3.0)
        mapped_ref, _, _, _, weights = matching.compute_weights(
            ref_group, comp_group, DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.NODE_ID
        )

        assert len(mapped_ref) == 1
        assert mapped_ref[0] == node1

    def test_anisotropic_scaling(self):
        """Test that anisotropic scaling correctly affects distance calculations."""
        graph1 = RustWorkXGraph()
        graph2 = RustWorkXGraph()

        graph1.add_node_attr_key("y", dtype=pl.Float64)
        graph1.add_node_attr_key("x", dtype=pl.Float64)
        graph2.add_node_attr_key("y", dtype=pl.Float64)
        graph2.add_node_attr_key("x", dtype=pl.Float64)

        # Nodes far in y but close in x
        graph1.add_node({DEFAULT_ATTR_KEYS.T: 0, "y": 0.0, "x": 0.0})
        graph2.add_node({DEFAULT_ATTR_KEYS.T: 0, "y": 5.0, "x": 1.0})

        ref_group = graph1.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, "y", "x"])
        comp_group = graph2.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, "y", "x"])

        # Without scale: too far
        matching = DistanceMatching(max_distance=3.0, attr_keys=("y", "x"))
        mapped_ref, _, _, _, _ = matching.compute_weights(
            ref_group, comp_group, DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.NODE_ID
        )
        assert len(mapped_ref) == 0

        # With scale: within threshold
        matching = DistanceMatching(max_distance=3.0, attr_keys=("y", "x"), scale=(0.2, 1.0))
        mapped_ref, _, _, _, _ = matching.compute_weights(
            ref_group, comp_group, DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.NODE_ID
        )
        assert len(mapped_ref) == 1

    def test_auto_detection_of_coordinates(self):
        """Test that coordinate dimensions are auto-detected when attr_keys is None."""
        # Test 2D auto-detection
        graph1 = RustWorkXGraph()
        graph2 = RustWorkXGraph()

        graph1.add_node_attr_key(DEFAULT_ATTR_KEYS.Y, dtype=pl.Float64)
        graph1.add_node_attr_key(DEFAULT_ATTR_KEYS.X, dtype=pl.Float64)
        graph2.add_node_attr_key(DEFAULT_ATTR_KEYS.Y, dtype=pl.Float64)
        graph2.add_node_attr_key(DEFAULT_ATTR_KEYS.X, dtype=pl.Float64)

        node1 = graph1.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.Y: 10.0, DEFAULT_ATTR_KEYS.X: 10.0})
        graph2.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.Y: 11.0, DEFAULT_ATTR_KEYS.X: 11.0})

        ref_group = graph1.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X])
        comp_group = graph2.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X])

        matching = DistanceMatching(max_distance=3.0)
        mapped_ref, _, _, _, _ = matching.compute_weights(
            ref_group, comp_group, DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.NODE_ID
        )

        assert len(mapped_ref) == 1
        assert mapped_ref[0] == node1

    def test_scale_validation(self):
        """Test that mismatched scale length raises ValueError."""
        graph1 = RustWorkXGraph()
        graph2 = RustWorkXGraph()

        graph1.add_node_attr_key("y", dtype=pl.Float64)
        graph1.add_node_attr_key("x", dtype=pl.Float64)
        graph2.add_node_attr_key("y", dtype=pl.Float64)
        graph2.add_node_attr_key("x", dtype=pl.Float64)

        graph1.add_node({DEFAULT_ATTR_KEYS.T: 0, "y": 10.0, "x": 10.0})
        graph2.add_node({DEFAULT_ATTR_KEYS.T: 0, "y": 11.0, "x": 11.0})

        ref_group = graph1.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, "y", "x"])
        comp_group = graph2.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, "y", "x"])

        matching = DistanceMatching(max_distance=3.0, attr_keys=("y", "x"), scale=(1.0, 2.0, 3.0))

        with pytest.raises(ValueError, match=r"Scale length .* must match attr_keys length"):
            matching.compute_weights(ref_group, comp_group, DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.NODE_ID)


class TestMatchingIntegration:
    """Integration tests for matching with actual graph operations."""

    def test_graph_match_integration(self):
        """Test MaskMatching and DistanceMatching integration with graph.match()."""

        # Test 1: MaskMatching with partial overlap
        graph1 = RustWorkXGraph()
        graph2 = RustWorkXGraph()

        default_mask = Mask(np.array([[False]]), np.array([0, 0, 1, 1]))
        graph1.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, dtype=pl.Object, default_value=default_mask)
        graph2.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, dtype=pl.Object, default_value=default_mask)

        mask1 = create_2d_mask_from_coords([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)])
        mask2 = create_2d_mask_from_coords([(0, 2), (0, 3), (0, 4), (0, 5), (0, 6)])

        node1 = graph1.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask1})
        node2 = graph2.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask2})

        # Add edges to avoid graph issues
        next1 = graph1.add_node({DEFAULT_ATTR_KEYS.T: 1, DEFAULT_ATTR_KEYS.MASK: mask1})
        graph1.add_edge(node1, next1, {})
        next2 = graph2.add_node({DEFAULT_ATTR_KEYS.T: 1, DEFAULT_ATTR_KEYS.MASK: mask2})
        graph2.add_edge(node2, next2, {})

        matching = MaskMatching(min_reference_intersection=0.5, optimal=True)
        graph1.match(graph2, matching=matching)

        node_attrs = graph1.node_attrs(
            attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.MATCHED_NODE_ID, DEFAULT_ATTR_KEYS.MATCH_SCORE]
        )

        matched_id = node_attrs.filter(pl.col(DEFAULT_ATTR_KEYS.NODE_ID) == node1)[
            DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
        ].item()
        match_score = node_attrs.filter(pl.col(DEFAULT_ATTR_KEYS.NODE_ID) == node1)[
            DEFAULT_ATTR_KEYS.MATCH_SCORE
        ].item()

        assert matched_id == node2
        assert np.isclose(match_score, 3.0 / 7.0, atol=1e-6)

        # Test 2: DistanceMatching
        graph1 = RustWorkXGraph()
        graph2 = RustWorkXGraph()

        graph1.add_node_attr_key("y", dtype=pl.Float64)
        graph1.add_node_attr_key("x", dtype=pl.Float64)
        graph2.add_node_attr_key("y", dtype=pl.Float64)
        graph2.add_node_attr_key("x", dtype=pl.Float64)

        node1 = graph1.add_node({DEFAULT_ATTR_KEYS.T: 0, "y": 10.0, "x": 10.0})
        node2 = graph2.add_node({DEFAULT_ATTR_KEYS.T: 0, "y": 11.0, "x": 11.0})

        next1 = graph1.add_node({DEFAULT_ATTR_KEYS.T: 1, "y": 10.0, "x": 10.0})
        graph1.add_edge(node1, next1, {})
        next2 = graph2.add_node({DEFAULT_ATTR_KEYS.T: 1, "y": 11.0, "x": 11.0})
        graph2.add_edge(node2, next2, {})

        matching = DistanceMatching(max_distance=3.0, optimal=True, attr_keys=("y", "x"))
        graph1.match(graph2, matching=matching)

        node_attrs = graph1.node_attrs(
            attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.MATCHED_NODE_ID, DEFAULT_ATTR_KEYS.MATCH_SCORE]
        )

        matched_id = node_attrs.filter(pl.col(DEFAULT_ATTR_KEYS.NODE_ID) == node1)[
            DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
        ].item()

        assert matched_id == node2

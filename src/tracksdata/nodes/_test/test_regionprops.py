import numpy as np
import pytest
from skimage.measure._regionprops import RegionProperties

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph
from tracksdata.nodes import Mask, RegionPropsNodes
from tracksdata.options import get_options, options_context


def test_regionprops_init_default() -> None:
    """Test RegionPropsNodes initialization with default parameters."""
    operator = RegionPropsNodes()

    assert operator._extra_properties == []
    assert operator._spacing is None


def test_regionprops_init_custom() -> None:
    """Test RegionPropsNodes initialization with custom parameters."""
    operator = RegionPropsNodes(extra_properties=["area", "perimeter"], spacing=(1.0, 2.0))

    assert operator._extra_properties == ["area", "perimeter"]
    assert operator._spacing == (1.0, 2.0)


def test_regionprops_attr_keys() -> None:
    """Test attr_keys method."""
    # Test with string properties
    operator = RegionPropsNodes(extra_properties=["area", "perimeter"])
    assert operator.attr_keys() == ["area", "perimeter"]

    # Test with callable properties
    def custom_prop(region: RegionProperties) -> float:
        return region.area * 2

    operator = RegionPropsNodes(extra_properties=[custom_prop, "area"])
    assert operator.attr_keys() == ["custom_prop", "area"]

    # Test with empty properties
    operator = RegionPropsNodes()
    assert operator.attr_keys() == []


def test_regionprops_add_nodes_2d() -> None:
    """Test adding nodes from 2D labels."""
    graph = RustWorkXGraph()

    # Create simple 2D labels
    labels = np.array([[[1, 1, 0], [1, 0, 2], [0, 2, 2]]], dtype=np.int32)

    operator = RegionPropsNodes(extra_properties=["area"])

    operator.add_nodes(graph, labels=labels)

    # Check that nodes were added
    assert graph.num_nodes == 2  # Two regions (labels 1 and 2)

    # Check node attributes
    nodes_df = graph.node_attrs()
    assert len(nodes_df) == 2
    assert DEFAULT_ATTR_KEYS.T in nodes_df.columns
    assert "y" in nodes_df.columns
    assert "x" in nodes_df.columns
    assert "area" in nodes_df.columns
    assert DEFAULT_ATTR_KEYS.MASK in nodes_df.columns

    # Check that all nodes have t=0
    assert all(nodes_df[DEFAULT_ATTR_KEYS.T] == 0)

    # Check areas (region 1 has 3 pixels, region 2 has 3 pixels)
    areas = sorted(nodes_df["area"])
    assert areas == [3, 3]


def test_regionprops_add_nodes_3d() -> None:
    """Test adding nodes from 3D labels."""
    graph = RustWorkXGraph()

    # Create simple 3D labels
    labels = np.array([[[[1, 1, 0], [1, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [2, 2, 0], [0, 0, 0]]]], dtype=np.int32)

    assert labels.shape == (2, 1, 3, 3)

    operator = RegionPropsNodes(extra_properties=["area"])

    operator.add_nodes(graph, labels=labels)

    # Check that nodes were added
    assert graph.num_nodes == 2  # Two regions

    # Check node attributes
    nodes_df = graph.node_attrs()
    assert len(nodes_df) == 2
    assert DEFAULT_ATTR_KEYS.T in nodes_df.columns
    assert "z" in nodes_df.columns
    assert "y" in nodes_df.columns
    assert "x" in nodes_df.columns
    assert "area" in nodes_df.columns
    assert DEFAULT_ATTR_KEYS.MASK in nodes_df.columns


def test_regionprops_add_nodes_with_intensity() -> None:
    """Test adding nodes with intensity image."""
    graph = RustWorkXGraph()

    # Create labels and intensity image
    labels = np.array([[[1, 1, 0], [1, 0, 2], [0, 2, 2]]], dtype=np.int32)

    assert labels.ndim == 3

    intensity = np.array([[[10, 20, 0], [30, 0, 40], [0, 50, 60]]], dtype=np.float32)

    assert intensity.ndim == 3

    operator = RegionPropsNodes(extra_properties=["mean_intensity"])

    operator.add_nodes(graph, labels=labels, intensity_image=intensity)

    # Check that nodes were added with intensity attributes
    nodes_df = graph.node_attrs()
    assert "mean_intensity" in nodes_df.columns

    # Check that mean intensities are calculated
    mean_intensities = sorted(nodes_df["mean_intensity"])
    # Region 1: pixels (10, 20, 30) -> mean = 20
    # Region 2: pixels (40, 50, 60) -> mean = 50
    assert abs(mean_intensities[0] - 20.0) < 1e-6
    assert abs(mean_intensities[1] - 50.0) < 1e-6


@pytest.mark.parametrize("n_workers", [1, 2])
def test_regionprops_add_nodes_timelapse(n_workers: int) -> None:
    """Test adding nodes from timelapse (t=None) with different worker counts."""
    graph = RustWorkXGraph()

    # Create timelapse labels (time x height x width)
    labels = np.array([[[1, 1], [0, 0]], [[0, 2], [2, 2]]], dtype=np.int32)  # t=0  # t=1

    assert labels.ndim == 3

    operator = RegionPropsNodes(extra_properties=["area"])

    with options_context(n_workers=n_workers):
        operator.add_nodes(graph, labels=labels)

    # Check that nodes were added for both time points
    nodes_df = graph.node_attrs()
    time_points = sorted(nodes_df[DEFAULT_ATTR_KEYS.T].unique())
    assert time_points == [0, 1]

    # Check that each time point has one region
    for t in time_points:
        nodes_at_t = nodes_df.filter(nodes_df[DEFAULT_ATTR_KEYS.T] == t)
        assert len(nodes_at_t) == 1


def test_regionprops_add_nodes_timelapse_with_intensity() -> None:
    """Test adding nodes from timelapse with intensity images."""
    graph = RustWorkXGraph()

    # Create timelapse labels and intensity
    labels = np.array([[[1, 1], [0, 0]], [[0, 2], [2, 2]]], dtype=np.int32)  # t=0  # t=1

    intensity = np.array([[[10, 20], [0, 0]], [[0, 30], [40, 50]]], dtype=np.float32)  # t=0  # t=1

    operator = RegionPropsNodes(extra_properties=["mean_intensity"])

    operator.add_nodes(graph, labels=labels, intensity_image=intensity)

    # Check that nodes were added with intensity attributes
    nodes_df = graph.node_attrs()
    assert "mean_intensity" in nodes_df.columns

    # Check mean intensities for each time point
    for t in [0, 1]:
        nodes_at_t = nodes_df.filter(nodes_df[DEFAULT_ATTR_KEYS.T] == t)
        assert len(nodes_at_t) == 1


def test_regionprops_custom_properties() -> None:
    """Test with custom property functions."""
    graph = RustWorkXGraph()

    # Create simple labels
    labels = np.array([[[1, 1, 0], [1, 0, 0], [0, 0, 0]]], dtype=np.int32)

    # Define custom property function
    def double_area(region: RegionProperties) -> float:
        return region.area * 2

    operator = RegionPropsNodes(extra_properties=[double_area, "area"])

    operator.add_nodes(graph, labels=labels, t=0)

    # Check that custom property was calculated
    nodes_df = graph.node_attrs()
    assert "double_area" in nodes_df.columns
    assert "area" in nodes_df.columns

    # Check that double_area is twice the area
    area = nodes_df["area"][0]
    double_area_val = nodes_df["double_area"][0]
    assert double_area_val == area * 2


def test_regionprops_invalid_dimensions() -> None:
    """Test error handling for invalid label dimensions."""
    graph = RustWorkXGraph()

    # Create 2D labels (invalid)
    labels = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

    operator = RegionPropsNodes()

    with pytest.raises(ValueError, match=r"`labels` must be 't \+ 2D' or 't \+ 3D'"):
        operator.add_nodes(graph, labels=labels)


def test_regionprops_mask_creation() -> None:
    """Test that masks are properly created for regions."""
    graph = RustWorkXGraph()

    # Create simple labels
    labels = np.array([[[1, 1, 0], [1, 0, 0], [0, 0, 2]]], dtype=np.int32)

    operator = RegionPropsNodes()

    operator.add_nodes(graph, labels=labels, t=0)

    # Check that masks were created
    nodes_df = graph.node_attrs()
    masks = nodes_df[DEFAULT_ATTR_KEYS.MASK]

    # All masks should be Mask objects
    for mask in masks:
        assert isinstance(mask, Mask)
        assert mask._mask is not None
        assert mask._bbox is not None


def test_regionprops_spacing() -> None:
    """Test regionprops with custom spacing."""
    graph = RustWorkXGraph()

    # Create simple labels
    labels = np.array([[[1, 1], [1, 1]]], dtype=np.int32)

    operator = RegionPropsNodes(extra_properties=["area"], spacing=(2.0, 3.0))  # Custom spacing

    operator.add_nodes(graph, labels=labels, t=0)

    # Check that nodes were added (spacing affects internal calculations)
    nodes_df = graph.node_attrs()

    assert len(nodes_df) == 1
    assert "area" in nodes_df.columns
    assert DEFAULT_ATTR_KEYS.MASK in nodes_df.columns
    assert nodes_df[DEFAULT_ATTR_KEYS.BBOX].to_numpy().ndim == 2


def test_regionprops_empty_labels() -> None:
    """Test behavior with empty labels (no regions)."""
    graph = RustWorkXGraph()

    # Create labels with no regions
    labels = np.zeros((1, 3, 3), dtype=np.int32)

    operator = RegionPropsNodes()

    operator.add_nodes(graph, labels=labels, t=0)

    # No nodes should be added
    assert graph.num_nodes == 0


def test_regionprops_multiprocessing_isolation() -> None:
    """Test that multiprocessing options don't affect subsequent tests."""
    # Verify default n_workers is 1
    assert get_options().n_workers == 1

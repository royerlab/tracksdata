import numpy as np
import polars as pl
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


SUPPORTED_PROPERTIES = [
    "area",
    "area_bbox",
    "area_convex",
    "area_filled",
    "axis_major_length",
    "axis_minor_length",
    "equivalent_diameter_area",
    "extent",
    "solidity",
]
SUPPORTED_PROPERTIES_2D = [
    "eccentricity",
    "feret_diameter_max",
    "orientation",
    "perimeter",
    "perimeter_crofton",
]
SUPPORTED_PROPERTIES_INTENSITY = [
    "intensity_max",
    "intensity_mean",
    "intensity_min",
    "intensity_std",
]


def test_regionprops_add_nodes_2d() -> None:
    """Test adding nodes from 2D labels."""
    graph = RustWorkXGraph()

    # Create simple 2D labels
    labels = np.array([[[1, 1, 0], [1, 0, 2], [0, 2, 2]]], dtype=np.int32)

    extra_properties = SUPPORTED_PROPERTIES + SUPPORTED_PROPERTIES_2D
    operator = RegionPropsNodes(extra_properties=extra_properties)
    operator.add_nodes(graph, labels=labels)

    assert "shape" in graph.metadata
    assert graph.metadata["shape"] == labels.shape

    # Check that nodes were added
    assert graph.num_nodes() == 2  # Two regions (labels 1 and 2)

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

    extra_properties = SUPPORTED_PROPERTIES
    operator = RegionPropsNodes(extra_properties=extra_properties)
    operator.add_nodes(graph, labels=labels)

    assert "shape" in graph.metadata
    assert graph.metadata["shape"] == labels.shape

    # Check that nodes were added
    assert graph.num_nodes() == 2  # Two regions

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

    extra_properties = SUPPORTED_PROPERTIES + SUPPORTED_PROPERTIES_INTENSITY
    operator = RegionPropsNodes(extra_properties=extra_properties)

    operator.add_nodes(graph, labels=labels, intensity_image=intensity)

    assert "shape" in graph.metadata
    assert graph.metadata["shape"] == labels.shape

    # Check that nodes were added with intensity attributes
    nodes_df = graph.node_attrs()
    assert "intensity_mean" in nodes_df.columns

    # Check that mean intensities are calculated
    mean_intensities = sorted(nodes_df["intensity_mean"])
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

    extra_properties = SUPPORTED_PROPERTIES + SUPPORTED_PROPERTIES_2D
    operator = RegionPropsNodes(extra_properties=extra_properties)

    with options_context(n_workers=n_workers):
        operator.add_nodes(graph, labels=labels)

    assert "shape" in graph.metadata
    assert graph.metadata["shape"] == labels.shape

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

    extra_properties = SUPPORTED_PROPERTIES + SUPPORTED_PROPERTIES_2D + SUPPORTED_PROPERTIES_INTENSITY
    operator = RegionPropsNodes(extra_properties=extra_properties)

    operator.add_nodes(graph, labels=labels, intensity_image=intensity)

    assert "shape" in graph.metadata
    assert graph.metadata["shape"] == labels.shape

    # Check that nodes were added with intensity attributes
    nodes_df = graph.node_attrs()
    assert "intensity_mean" in nodes_df.columns

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

    assert "shape" in graph.metadata
    assert graph.metadata["shape"] == labels.shape

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

    assert "shape" in graph.metadata
    assert graph.metadata["shape"] == labels.shape

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

    assert "shape" in graph.metadata
    assert graph.metadata["shape"] == labels.shape

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

    assert "shape" in graph.metadata
    assert graph.metadata["shape"] == labels.shape

    # No nodes should be added
    assert graph.num_nodes() == 0


def test_regionprops_multiprocessing_isolation() -> None:
    """Test that multiprocessing options don't affect subsequent tests."""
    # Verify default n_workers is 1
    assert get_options().n_workers == 1


TIMELAPSE_LABELS = np.array(
    [
        [[1, 1, 0], [1, 0, 2], [0, 2, 2]],  # t=0
        [[0, 3, 3], [0, 3, 0], [4, 0, 0]],  # t=1
    ],
    dtype=np.int32,
)

TIMELAPSE_INTENSITY = np.array(
    [
        [[10, 20, 0], [30, 0, 40], [0, 50, 60]],  # t=0
        [[0, 70, 80], [0, 90, 0], [100, 0, 0]],  # t=1
    ],
    dtype=np.float32,
)


def test_regionprops_attrs_init_validation() -> None:
    """Test RegionPropsNodes property validation for add_node_attrs."""
    operator = RegionPropsNodes(extra_properties=["area", "intensity_mean"], spacing=(1.0, 2.0))
    assert operator.attr_keys() == ["area", "intensity_mean"]
    assert operator._spacing == (1.0, 2.0)

    # empty properties are only rejected when (re-)computing attributes on existing nodes
    graph = RustWorkXGraph()
    RegionPropsNodes().add_nodes(graph, labels=TIMELAPSE_LABELS)
    with pytest.raises(ValueError, match="at least one region property"):
        RegionPropsNodes().add_node_attrs(graph)

    with pytest.raises(ValueError, match="`centroid` is not supported"):
        RegionPropsNodes(extra_properties=["centroid"])

    with pytest.raises(ValueError, match="`bbox` is not supported"):
        RegionPropsNodes(extra_properties=["bbox"])


@pytest.mark.parametrize("n_workers", [1, 2])
def test_regionprops_attrs_matches_nodes_operator(n_workers: int) -> None:
    """Test that recomputed properties match those computed at node creation."""
    properties = ["area", "intensity_mean", "intensity_max"]

    # ground truth: properties computed directly from the labels
    expected_graph = RustWorkXGraph()
    RegionPropsNodes(extra_properties=properties).add_nodes(
        expected_graph, labels=TIMELAPSE_LABELS, intensity_image=TIMELAPSE_INTENSITY
    )
    expected_df = expected_graph.node_attrs(attr_keys=properties)

    # recompute: nodes created without properties, then add_node_attrs
    graph = RustWorkXGraph()
    RegionPropsNodes().add_nodes(graph, labels=TIMELAPSE_LABELS)

    operator = RegionPropsNodes(extra_properties=properties)
    with options_context(n_workers=n_workers):
        operator.add_node_attrs(graph, intensity_image=TIMELAPSE_INTENSITY)

    result_df = graph.node_attrs(attr_keys=properties)

    for prop in properties:
        np.testing.assert_allclose(result_df[prop].to_numpy(), expected_df[prop].to_numpy())


def test_regionprops_attrs_callable_property() -> None:
    """Test recomputing properties with a custom callable."""

    def double_area(region: RegionProperties) -> float:
        return region.area * 2

    graph = RustWorkXGraph()
    RegionPropsNodes(extra_properties=["area"]).add_nodes(graph, labels=TIMELAPSE_LABELS)

    RegionPropsNodes(extra_properties=[double_area]).add_node_attrs(graph)

    nodes_df = graph.node_attrs(attr_keys=["area", "double_area"])
    np.testing.assert_array_equal(
        nodes_df["double_area"].to_numpy(),
        nodes_df["area"].to_numpy() * 2,
    )


def test_regionprops_attrs_single_time_point() -> None:
    """Test recomputing properties for a single time point only."""
    graph = RustWorkXGraph()
    RegionPropsNodes().add_nodes(graph, labels=TIMELAPSE_LABELS, intensity_image=TIMELAPSE_INTENSITY)

    RegionPropsNodes(extra_properties=["intensity_mean"]).add_node_attrs(
        graph, t=1, intensity_image=TIMELAPSE_INTENSITY
    )

    nodes_df = graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.T, "intensity_mean"])
    at_t1 = nodes_df.filter(nodes_df[DEFAULT_ATTR_KEYS.T] == 1)

    # region 3: pixels (70, 80, 90) -> mean = 80; region 4: pixel (100,) -> mean = 100
    np.testing.assert_allclose(sorted(at_t1["intensity_mean"]), [80.0, 100.0])


def test_regionprops_attrs_overwrites_existing_values() -> None:
    """Test that existing attribute values are overwritten by the recomputation."""
    graph = RustWorkXGraph()
    RegionPropsNodes(extra_properties=["area"]).add_nodes(graph, labels=TIMELAPSE_LABELS)

    expected_areas = graph.node_attrs(attr_keys=["area"])["area"].to_numpy().copy()

    # corrupt the stored values
    graph.update_node_attrs(node_ids=graph.node_ids(), attrs={"area": [-1] * graph.num_nodes()})

    RegionPropsNodes(extra_properties=["area"]).add_node_attrs(graph)

    np.testing.assert_array_equal(
        graph.node_attrs(attr_keys=["area"])["area"].to_numpy(),
        expected_areas,
    )


def test_regionprops_attrs_spacing() -> None:
    """Test that spacing affects the recomputed measurements."""
    graph = RustWorkXGraph()
    RegionPropsNodes(extra_properties=["area"]).add_nodes(graph, labels=TIMELAPSE_LABELS)

    pixel_areas = graph.node_attrs(attr_keys=["area"])["area"].to_numpy().copy()

    RegionPropsNodes(extra_properties=["area"], spacing=(2.0, 3.0)).add_node_attrs(graph)

    np.testing.assert_allclose(
        graph.node_attrs(attr_keys=["area"])["area"].to_numpy(),
        pixel_areas * 6.0,
    )


def test_regionprops_attrs_missing_mask_key() -> None:
    """Test error handling when the mask key is missing or invalid."""
    graph = RustWorkXGraph()

    with pytest.raises(ValueError, match="Mask key 'mask' not found"):
        RegionPropsNodes(extra_properties=["area"]).add_node_attrs(graph)

    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, pl.Object)
    graph.add_node({"t": 0, "mask": "not a mask"})

    with pytest.raises(TypeError, match="Expected `Mask` object"):
        RegionPropsNodes(extra_properties=["area"]).add_node_attrs(graph)

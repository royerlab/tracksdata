import numpy as np
import pytest
from numpy.typing import NDArray

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph
from tracksdata.nodes import GenericFuncNodeAttrs, Mask
from tracksdata.options import get_options, options_context


def test_crop_func_attrs_init_default() -> None:
    """Test CropFuncAttrs initialization with default parameters."""

    def dummy_func(value: float) -> float:
        return value * 2.0

    operator = GenericFuncNodeAttrs(
        func=dummy_func,
        output_key="test_output",
    )

    assert operator.func == dummy_func
    assert operator.output_key == "test_output"
    assert operator.attr_keys == ()


def test_crop_func_attrs_init_with_attr_keys() -> None:
    """Test CropFuncAttrs initialization with custom attr_keys."""

    def dummy_func(value: float, multiplier: int) -> float:
        return value * multiplier

    operator = GenericFuncNodeAttrs(
        func=dummy_func,
        output_key="test_output",
        attr_keys=["multiplier"],
    )

    assert operator.func == dummy_func
    assert operator.output_key == "test_output"
    assert operator.attr_keys == ["multiplier"]


def test_crop_func_attrs_init_with_sequence_output_key() -> None:
    """Test CropFuncAttrs initialization with sequence output_key."""

    def dummy_func(value: float) -> float:
        return value * 2.0

    operator = GenericFuncNodeAttrs(
        func=dummy_func,
        output_key=["test_output"],
    )

    assert operator.output_key == ["test_output"]


def test_crop_func_attrs_simple_function_no_frames() -> None:
    """Test applying a simple function without frames."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("value", 0.0)

    # Add nodes with values
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "value": 10.0})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "value": 20.0})

    def double_value(value: float) -> float:
        return value * 2.0

    # Create operator and add attributes
    operator = GenericFuncNodeAttrs(
        func=double_value,
        output_key="doubled_value",
        attr_keys=["value"],
    )

    operator.add_node_attrs(graph)

    # Check that attributes were added
    nodes_df = graph.node_attrs()
    assert "doubled_value" in nodes_df.columns

    # Check results
    doubled_values = dict(zip(nodes_df[DEFAULT_ATTR_KEYS.NODE_ID], nodes_df["doubled_value"], strict=False))
    assert doubled_values[node1] == 20.0
    assert doubled_values[node2] == 40.0


def test_crop_func_attrs_function_with_frames() -> None:
    """Test applying a function with frames."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)

    # Create test masks
    mask1_data = np.array([[True, True], [True, False]], dtype=bool)
    mask1 = Mask(mask1_data, bbox=np.array([0, 0, 2, 2]))

    mask2_data = np.array([[True, False], [False, False]], dtype=bool)
    mask2 = Mask(mask2_data, bbox=np.array([0, 0, 2, 2]))

    # Add nodes with masks
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask1})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask2})

    # Create test frames
    frames = np.array(
        [
            np.array([[100, 200], [300, 400]]),  # Frame 0
        ]
    )

    def intensity_sum(frame: NDArray, mask: Mask) -> float:
        cropped = mask.crop(frame)
        return float(np.sum(cropped[mask.mask]))

    # Create operator and add attributes
    operator = GenericFuncNodeAttrs(
        func=intensity_sum,
        output_key="intensity_sum",
        attr_keys=[DEFAULT_ATTR_KEYS.MASK],
    )

    operator.add_node_attrs(graph, t=0, frames=frames)

    # Check that attributes were added
    nodes_df = graph.node_attrs()
    assert "intensity_sum" in nodes_df.columns

    # Check results
    intensity_sums = dict(zip(nodes_df[DEFAULT_ATTR_KEYS.NODE_ID], nodes_df["intensity_sum"], strict=False))

    # Expected: mask1 has 3 True pixels, mask2 has 1 True pixel
    # Frame values: [[100, 200], [300, 400]]
    # mask1 covers [0,0], [0,1], [1,0] -> 100 + 200 + 300 = 600
    # mask2 covers [0,0] -> 100
    assert intensity_sums[node1] == 600.0
    assert intensity_sums[node2] == 100.0


def test_crop_func_attrs_function_with_frames_and_attrs() -> None:
    """Test applying a function with frames and additional attributes."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)
    graph.add_node_attr_key("multiplier", 1.0)

    # Create test masks
    mask1_data = np.array([[True, True], [True, False]], dtype=bool)
    mask1 = Mask(mask1_data, bbox=np.array([0, 0, 2, 2]))

    mask2_data = np.array([[True, False], [False, False]], dtype=bool)
    mask2 = Mask(mask2_data, bbox=np.array([0, 0, 2, 2]))

    # Add nodes with masks and multipliers
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask1, "multiplier": 2.0})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask2, "multiplier": 3.0})

    # Create test frames
    frames = np.array(
        [
            np.array([[10, 20], [30, 40]]),  # Frame 0
        ]
    )

    def intensity_sum_times_multiplier(frame: NDArray, mask: Mask, multiplier: float) -> float:
        cropped = mask.crop(frame)
        return float(np.sum(cropped[mask.mask]) * multiplier)

    # Create operator and add attributes
    operator = GenericFuncNodeAttrs(
        func=intensity_sum_times_multiplier,
        output_key="weighted_intensity",
        attr_keys=["mask", "multiplier"],
    )

    operator.add_node_attrs(graph, t=0, frames=frames)

    # Check that attributes were added
    nodes_df = graph.node_attrs()
    assert "weighted_intensity" in nodes_df.columns

    # Check results
    weighted_intensities = dict(zip(nodes_df[DEFAULT_ATTR_KEYS.NODE_ID], nodes_df["weighted_intensity"], strict=False))

    # Expected:
    # mask1: sum = 10 + 20 + 30 = 60, multiplier = 2.0 -> 120.0
    # mask2: sum = 10, multiplier = 3.0 -> 30.0
    assert weighted_intensities[node1] == 120.0
    assert weighted_intensities[node2] == 30.0


def test_crop_func_attrs_function_returns_different_types() -> None:
    """Test that functions can return different types."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)

    # Create test mask
    mask_data = np.array([[True, True], [True, False]], dtype=bool)
    mask = Mask(mask_data, bbox=np.array([0, 0, 2, 2]))

    # Add node
    graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask})

    def return_string(mask: Mask) -> str:
        return "test_string"

    def return_list(mask: Mask) -> list[int]:
        return [1, 2, 3]

    def return_dict(mask: Mask) -> dict[str, int]:
        return {"count": 3}

    def return_array(mask: Mask) -> NDArray:
        return np.asarray([1, 2, 3])

    # Test string return type
    operator_str = GenericFuncNodeAttrs(
        func=return_string,
        output_key="string_result",
        attr_keys=[DEFAULT_ATTR_KEYS.MASK],
    )
    operator_str.add_node_attrs(graph)

    # Test list return type
    operator_list = GenericFuncNodeAttrs(
        func=return_list,
        output_key="list_result",
        attr_keys=[DEFAULT_ATTR_KEYS.MASK],
    )
    operator_list.add_node_attrs(graph)

    # Test dict return type
    operator_dict = GenericFuncNodeAttrs(
        func=return_dict,
        output_key="dict_result",
        attr_keys=[DEFAULT_ATTR_KEYS.MASK],
    )
    operator_dict.add_node_attrs(graph)

    # Test array return type
    operator_array = GenericFuncNodeAttrs(
        func=return_array,
        output_key="array_result",
        attr_keys=[DEFAULT_ATTR_KEYS.MASK],
    )
    operator_array.add_node_attrs(graph)

    # Check results
    nodes_df = graph.node_attrs()
    assert nodes_df["string_result"][0] == "test_string"
    assert nodes_df["list_result"][0].to_list() == [1, 2, 3]
    assert nodes_df["dict_result"][0] == {"count": 3}
    np.testing.assert_array_equal(nodes_df["array_result"][0], np.asarray([1, 2, 3]))


def test_crop_func_attrs_error_handling_missing_attr_key() -> None:
    """Test error handling when required attr_key is missing."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)
    # Note: "value" is not registered

    # Create test mask
    mask_data = np.array([[True, True], [True, False]], dtype=bool)
    mask = Mask(mask_data, bbox=np.array([0, 0, 2, 2]))

    # Add node without the required attribute
    graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask})

    def use_value(mask: Mask, value: float) -> float:
        return value * 2.0

    # Create operator that requires "value" attribute
    operator = GenericFuncNodeAttrs(
        func=use_value,
        output_key="result",
        attr_keys=["value"],
    )

    # Should raise an error when trying to access missing attribute
    with pytest.raises(KeyError):  # Specific exception type depends on graph backend
        operator.add_node_attrs(graph)


@pytest.mark.parametrize("n_workers", [1, 2])
def test_crop_func_attrs_function_with_frames_multiprocessing(n_workers: int) -> None:
    """Test applying a function with frames using different worker counts."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)

    # Create test masks for multiple time points
    mask1_data = np.array([[True, True], [True, False]], dtype=bool)
    mask1 = Mask(mask1_data, bbox=np.array([0, 0, 2, 2]))

    mask2_data = np.array([[True, False], [False, False]], dtype=bool)
    mask2 = Mask(mask2_data, bbox=np.array([0, 0, 2, 2]))

    # Add nodes with masks at different time points
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask1})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, DEFAULT_ATTR_KEYS.MASK: mask2})

    # Create test frames for multiple time points
    frames = np.array(
        [
            np.array([[100, 200], [300, 400]]),  # Frame 0
            np.array([[10, 20], [30, 40]]),  # Frame 1
        ]
    )

    def intensity_sum(frame: NDArray, mask: Mask) -> float:
        cropped = mask.crop(frame)
        return float(np.sum(cropped[mask.mask]))

    # Create operator and add attributes
    operator = GenericFuncNodeAttrs(
        func=intensity_sum,
        output_key="intensity_sum",
        attr_keys=[DEFAULT_ATTR_KEYS.MASK],
    )

    with options_context(n_workers=n_workers):
        operator.add_node_attrs(graph, frames=frames)

    # Check that attributes were added
    nodes_df = graph.node_attrs()
    assert "intensity_sum" in nodes_df.columns

    # Check results
    intensity_sums = dict(zip(nodes_df[DEFAULT_ATTR_KEYS.NODE_ID], nodes_df["intensity_sum"], strict=False))

    # Expected calculations based on masks and frames
    assert intensity_sums[node1] == 600.0  # mask1 with frame 0
    assert intensity_sums[node2] == 10.0  # mask2 with frame 1


def test_crop_func_attrs_empty_graph() -> None:
    """Test behavior with an empty graph."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)

    def dummy_func(mask: Mask) -> float:
        return 1.0

    operator = GenericFuncNodeAttrs(
        func=dummy_func,
        output_key="result",
    )

    # Should not raise an error, just do nothing
    operator.add_node_attrs(graph)

    # Check that no attributes were added
    nodes_df = graph.node_attrs()
    assert len(nodes_df) == 0


def test_crop_func_attrs_multiprocessing_isolation() -> None:
    """Test that multiprocessing options don't affect subsequent tests."""
    # Verify default n_workers is 1
    assert get_options().n_workers == 1


def test_crop_func_attrs_batch_processing_without_frames() -> None:
    """Test batch processing with batch_size > 0 without frames."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("value", 0.0)

    # Add nodes with values
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "value": 10.0})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "value": 20.0})
    node3 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "value": 30.0})

    def batch_double_value(value: list[float]) -> list[float]:
        """Batch function that doubles each value."""
        return [v * 2.0 for v in value]

    # Create operator with batch_size = 2
    operator = GenericFuncNodeAttrs(
        func=batch_double_value,
        output_key="doubled_value",
        attr_keys=["value"],
        batch_size=2,
    )

    operator.add_node_attrs(graph)

    # Check that attributes were added
    nodes_df = graph.node_attrs()
    assert "doubled_value" in nodes_df.columns

    # Check results
    doubled_values = dict(zip(nodes_df[DEFAULT_ATTR_KEYS.NODE_ID], nodes_df["doubled_value"], strict=False))
    assert doubled_values[node1] == 20.0
    assert doubled_values[node2] == 40.0
    assert doubled_values[node3] == 60.0


def test_crop_func_attrs_batch_processing_with_frames() -> None:
    """Test batch processing with batch_size > 0 with frames."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)

    # Create test masks
    mask1_data = np.array([[True, True], [True, False]], dtype=bool)
    mask1 = Mask(mask1_data, bbox=np.array([0, 0, 2, 2]))

    mask2_data = np.array([[True, False], [False, False]], dtype=bool)
    mask2 = Mask(mask2_data, bbox=np.array([0, 0, 2, 2]))

    mask3_data = np.array([[False, True], [True, True]], dtype=bool)
    mask3 = Mask(mask3_data, bbox=np.array([0, 0, 2, 2]))

    # Add nodes with masks
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask1})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask2})
    node3 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask3})

    # Create test frames
    frames = np.array(
        [
            np.array([[100, 200], [300, 400]]),  # Frame 0
        ]
    )

    def batch_intensity_sum(frame: NDArray, mask: list[Mask]) -> list[float]:
        """Batch function that calculates intensity sum for each mask."""
        results = []
        for m in mask:
            cropped = m.crop(frame)
            results.append(float(np.sum(cropped[m.mask])))
        return results

    # Create operator with batch_size = 2
    operator = GenericFuncNodeAttrs(
        func=batch_intensity_sum,
        output_key="intensity_sum",
        attr_keys=["mask"],
        batch_size=2,
    )

    operator.add_node_attrs(graph, frames=frames)

    # Check that attributes were added
    nodes_df = graph.node_attrs()
    assert "intensity_sum" in nodes_df.columns

    # Check results
    intensity_sums = dict(zip(nodes_df[DEFAULT_ATTR_KEYS.NODE_ID], nodes_df["intensity_sum"], strict=False))

    # Expected calculations:
    # mask1 covers [0,0], [0,1], [1,0] -> 100 + 200 + 300 = 600
    # mask2 covers [0,0] -> 100
    # mask3 covers [0,1], [1,0], [1,1] -> 200 + 300 + 400 = 900
    assert intensity_sums[node1] == 600.0
    assert intensity_sums[node2] == 100.0
    assert intensity_sums[node3] == 900.0

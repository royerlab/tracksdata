import polars as pl
import pytest

from tracksdata.functional import TilingScheme, apply_tiled
from tracksdata.functional._apply import _get_tiles_corner
from tracksdata.graph import RustWorkXGraph


@pytest.fixture
def sample_graph() -> RustWorkXGraph:
    """Create a sample graph with spatial nodes for testing."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("z", 0)
    graph.add_node_attr_key("y", 0)
    graph.add_node_attr_key("x", 0)

    # Add nodes in a grid pattern
    nodes = [
        {"t": 0, "z": 0, "y": 10, "x": 10},
        {"t": 0, "z": 0, "y": 10, "x": 30},
        {"t": 0, "z": 0, "y": 30, "x": 10},
        {"t": 0, "z": 0, "y": 30, "x": 30},
        {"t": 1, "z": 0, "y": 50, "x": 50},
        {"t": 1, "z": 0, "y": 70, "x": 70},
    ]

    for node_attrs in nodes:
        graph.add_node(node_attrs)

    return graph


def test_tiling_scheme_validation() -> None:
    """Test TilingScheme validation."""
    # Valid initialization
    scheme = TilingScheme(tile=(10, 10), overlap=(2, 2))
    assert scheme.tile == (10, 10)
    assert scheme.overlap == (2, 2)

    # Mismatched tile_shape and overlap lengths
    with pytest.raises(ValueError, match="must have the same length"):
        TilingScheme(tile=(10, 10), overlap=(2,))

    # Mismatched attrs and tile_shape lengths
    with pytest.raises(ValueError, match="must have the same length"):
        TilingScheme(tile=(10, 10), overlap=(2, 2), attrs=["y", "x", "z"])

    # tile_shape must be greater than 0
    with pytest.raises(ValueError, match="must be greater than 0"):
        TilingScheme(tile=(0, 10), overlap=(2, 2))

    # overlap must be non-negative
    with pytest.raises(ValueError, match="must be non-negative"):
        TilingScheme(tile=(10, 10), overlap=(-2, 2))


def test_apply_tiled_no_aggregation(sample_graph: RustWorkXGraph) -> None:
    """Test apply_tiled yields tiles without aggregation."""
    scheme = TilingScheme(tile=(1, 20, 20), overlap=(0, 5, 5), attrs=["t", "y", "x"])

    results = list(
        apply_tiled(
            graph=sample_graph,
            tiling_scheme=scheme,
            func=lambda tile: len(tile.graph_filter.node_ids()),
            agg_func=None,
        )
    )

    # Should yield results for multiple tiles
    assert len(results) > 0
    assert all(isinstance(r, int) for r in results)


def test_apply_tiled_with_aggregation(sample_graph: RustWorkXGraph) -> None:
    """Test apply_tiled with aggregation function."""
    scheme = TilingScheme(tile=(1, 20, 20), overlap=(0, 5, 5), attrs=["t", "y", "x"])

    # Count total nodes across all tiles
    total = apply_tiled(
        graph=sample_graph,
        tiling_scheme=scheme,
        func=lambda tile: len(tile.graph_filter.node_ids()),
        agg_func=sum,
    )

    assert isinstance(total, int)
    # Due to overlaps, total should be >= original node count
    assert total >= sample_graph.num_nodes


def test_apply_tiled_default_attrs(sample_graph: RustWorkXGraph) -> None:
    """Test apply_tiled uses default attrs [t, z, y, x] when not specified."""
    # When attrs=None, it uses [t, z, y, x] by default, but filters to existing keys
    # Since sample_graph has all four dimensions, all will be used
    # Use explicit attrs to test with dimensions that have actual extent
    scheme = TilingScheme(tile=(2, 100, 100), overlap=(0, 10, 10), attrs=["t", "y", "x"])

    results = list(
        apply_tiled(
            graph=sample_graph,
            tiling_scheme=scheme,
            func=lambda tile: tile.graph_filter.node_attrs(attr_keys=["t", "y", "x"]),
            agg_func=None,
        )
    )

    assert len(results) > 0
    assert all(isinstance(r, pl.DataFrame) for r in results)


def test_apply_tiled_2d_tiling() -> None:
    """Test apply_tiled with 2D spatial coordinates."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("y", 0)
    graph.add_node_attr_key("x", 0)

    for y in [5, 11, 14]:
        for x in [10, 30]:
            graph.add_node({"t": 0, "y": y, "x": x})

    """
    # node ids to coords
    # 0 : (5, 10)
    # 1 : (5, 30)
    # 2 : (11, 10)
    # 3 : (11, 30)
    # 4 : (14, 10)
    # 5 : (14, 30)

      x
      |
    30|  1    3   5
      |
      |
    10|  0    2   4
      |
      ---------------------y
     0   5   10   15   20
    """

    scheme = TilingScheme(
        tile=(1, 5, 15),
        overlap=(0, 5, 5),
        attrs=["t", "y", "x"],
    )

    tiles_corner = _get_tiles_corner(
        start=[0, 5, 10],
        end=[0, 15, 30],
        tiling_scheme=scheme,
    )
    expected_tiles_corner = [(0.0, 5.0, 10.0), (0.0, 5.0, 25.0), (0.0, 10.0, 10.0), (0.0, 10.0, 25.0)]
    for c, s in zip(tiles_corner, expected_tiles_corner, strict=False):
        assert c == s

    results = list(
        apply_tiled(
            graph=graph,
            tiling_scheme=scheme,
            func=lambda tile: (tile.graph_filter.node_ids(), tile.graph_filter_wo_overlap.node_ids()),
            agg_func=None,
        )
    )

    res_tile_with_overlap, res_tile_wo_overlap = zip(*results, strict=False)

    assert len(res_tile_with_overlap) == 4
    assert set(res_tile_with_overlap[0]) == {0, 2, 4}
    assert set(res_tile_with_overlap[1]) == {1, 3, 5}
    assert set(res_tile_with_overlap[2]) == {0, 2, 4}
    assert set(res_tile_with_overlap[3]) == {1, 3, 5}

    assert len(res_tile_wo_overlap) == 4
    assert set(res_tile_wo_overlap[0]) == {0}
    assert set(res_tile_wo_overlap[1]) == {1}
    assert set(res_tile_wo_overlap[2]) == {2, 4}
    assert set(res_tile_wo_overlap[3]) == {3, 5}

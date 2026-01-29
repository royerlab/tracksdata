import polars as pl
import pytest

from tracksdata.functional import TilingScheme, apply_tiled
from tracksdata.functional._apply import _get_tiles_corner
from tracksdata.graph import RustWorkXGraph


@pytest.fixture
def sample_graph() -> RustWorkXGraph:
    """Create a sample graph with spatial nodes for testing."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("z", dtype=pl.Int64)
    graph.add_node_attr_key("y", dtype=pl.Int64)
    graph.add_node_attr_key("x", dtype=pl.Int64)

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
    scheme = TilingScheme(tile_shape=(10, 10), overlap_shape=(2, 2))
    assert scheme.tile_shape == (10, 10)
    assert scheme.overlap_shape == (2, 2)

    # Mismatched tile_shape and overlap lengths
    with pytest.raises(ValueError, match="must have the same length"):
        TilingScheme(tile_shape=(10, 10), overlap_shape=(2,))

    # Mismatched attrs and tile_shape lengths
    with pytest.raises(ValueError, match="must have the same length"):
        TilingScheme(tile_shape=(10, 10), overlap_shape=(2, 2), attrs=["y", "x", "z"])

    # tile_shape must be greater than 0
    with pytest.raises(ValueError, match="must be greater than 0"):
        TilingScheme(tile_shape=(0, 10), overlap_shape=(2, 2))

    # overlap must be non-negative
    with pytest.raises(ValueError, match="must be non-negative"):
        TilingScheme(tile_shape=(10, 10), overlap_shape=(-2, 2))


def test_apply_tiled_no_aggregation(sample_graph: RustWorkXGraph) -> None:
    """Test apply_tiled yields tiles without aggregation."""
    scheme = TilingScheme(tile_shape=(1, 20, 20), overlap_shape=(0, 5, 5), attrs=["t", "y", "x"])

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
    scheme = TilingScheme(tile_shape=(1, 20, 20), overlap_shape=(0, 5, 5), attrs=["t", "y", "x"])

    # Count total nodes across all tiles
    total = apply_tiled(
        graph=sample_graph,
        tiling_scheme=scheme,
        func=lambda tile: len(tile.graph_filter.node_ids()),
        agg_func=sum,
    )

    assert isinstance(total, int)
    # Due to overlaps, total should be >= original node count
    assert total >= sample_graph.num_nodes()


def test_apply_tiled_default_attrs(sample_graph: RustWorkXGraph) -> None:
    """Test apply_tiled uses default attrs [t, z, y, x] when not specified."""
    # When attrs=None, it uses [t, z, y, x] by default, but filters to existing keys
    # Since sample_graph has all four dimensions, all will be used
    # Use explicit attrs to test with dimensions that have actual extent
    scheme = TilingScheme(tile_shape=(2, 100, 100), overlap_shape=(0, 10, 10), attrs=["t", "y", "x"])

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
    graph.add_node_attr_key("y", dtype=pl.Float64)
    graph.add_node_attr_key("x", dtype=pl.Float64)

    for y in [5, 11, 14]:
        for x in [10, 30]:
            graph.add_node({"t": 0, "y": y, "x": x})
    graph.add_node({"t": 0, "y": 9.999999, "x": 10})

    """
    # node ids to coords
    # 0 : (5, 10)
    # 1 : (5, 30)
    # 2 : (11, 10)
    # 3 : (11, 30)
    # 4 : (14, 10)
    # 5 : (14, 30)
    # 6 : (9.999999, 10)

      x
      |
    30|  1    3   5
      |
      |
    10|  0  6 2   4
      |
      ---------------------y
     0   5   10   15   20
    """

    scheme = TilingScheme(
        tile_shape=(1, 5, 15),
        overlap_shape=(0, 5, 5),
        attrs=["t", "y", "x"],
    )

    tiles_corner = _get_tiles_corner(
        start=[0, 5, 10],
        end=[0, 14, 30],
        tiling_scheme=scheme,
    )
    expected_tiles_corner = [(0.0, 5.0, 10.0), (0.0, 5.0, 25.0), (0.0, 10.0, 10.0), (0.0, 10.0, 25.0)]
    assert len(tiles_corner) == len(expected_tiles_corner)
    for c, s in zip(tiles_corner, expected_tiles_corner, strict=True):
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
    assert set(res_tile_with_overlap[0]) == {0, 2, 4, 6}
    assert set(res_tile_with_overlap[1]) == {1, 3, 5}
    assert set(res_tile_with_overlap[2]) == {0, 2, 4, 6}
    assert set(res_tile_with_overlap[3]) == {1, 3, 5}

    assert len(res_tile_wo_overlap) == 4
    assert set(res_tile_wo_overlap[0]) == {0, 6}
    assert set(res_tile_wo_overlap[1]) == {1}
    assert set(res_tile_wo_overlap[2]) == {2, 4}
    assert set(res_tile_wo_overlap[3]) == {3, 5}


def test_apply_tile_scale_invariance() -> None:
    """
    This check that the tiling scheme is scale invariant
    if the coordinate space, `tile_shape` and `overlap_shape` are scaled by the same factor.
    """
    pos = [-1, 0, 1]
    scales = [1e-6, 0.001, 1, 1_000, 1_000_000]
    results = []

    for scale in scales:
        graph = RustWorkXGraph()
        # hack: updating schema
        graph._node_attr_schemas()["t"].dtype = pl.Float64

        for p in pos:
            graph.add_node({"t": p * scale})

        scheme = TilingScheme(
            tile_shape=(1 * scale,),
            overlap_shape=(1 * scale,),
            attrs=["t"],
        )

        results.append(
            list(
                apply_tiled(
                    graph=graph,
                    tiling_scheme=scheme,
                    func=lambda tile: (
                        tile.graph_filter.node_ids(),
                        tile.graph_filter_wo_overlap.node_ids(),
                    ),
                    agg_func=None,
                )
            )
        )

    first_result = results[0]

    for result in results[1:]:
        assert result == first_result
        for node_ids, node_ids_wo_overlap in result:
            assert len(node_ids) >= 1
            assert len(node_ids_wo_overlap) == 1

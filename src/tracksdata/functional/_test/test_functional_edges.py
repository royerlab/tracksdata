import polars as pl
import polars.testing as pl_testing

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.functional import join_node_attrs_to_edges


def test_node_attr_to_edges_basic() -> None:
    node_attrs = pl.DataFrame({DEFAULT_ATTR_KEYS.NODE_ID: [1, 2, 3], "a": [1, 2, 3], "b": [4, 5, 6]})
    edge_attrs = pl.DataFrame({DEFAULT_ATTR_KEYS.EDGE_SOURCE: [1, 2], DEFAULT_ATTR_KEYS.EDGE_TARGET: [2, 3]})

    result = join_node_attrs_to_edges(node_attrs, edge_attrs)
    expected = pl.DataFrame(
        {
            DEFAULT_ATTR_KEYS.EDGE_SOURCE: [1, 2],
            DEFAULT_ATTR_KEYS.EDGE_TARGET: [2, 3],
            "source_a": [1, 2],
            "source_b": [4, 5],
            "target_a": [2, 3],
            "target_b": [5, 6],
        }
    )

    pl_testing.assert_frame_equal(result, expected)


def test_node_attr_to_edges_custom_keys() -> None:
    node_attrs = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
    edge_attrs = pl.DataFrame({"from": [1, 2], "to": [2, 3]})

    result = join_node_attrs_to_edges(
        node_attrs,
        edge_attrs,
        node_id_key="id",
        source_key="from",
        target_key="to",
    )

    expected = pl.DataFrame({"from": [1, 2], "to": [2, 3], "source_value": [10, 20], "target_value": [20, 30]})

    pl_testing.assert_frame_equal(result, expected)


def test_node_attr_to_edges_custom_prefixes() -> None:
    node_attrs = pl.DataFrame({DEFAULT_ATTR_KEYS.NODE_ID: [1, 2], "x": [10, 20]})
    edge_attrs = pl.DataFrame({DEFAULT_ATTR_KEYS.EDGE_SOURCE: [1], DEFAULT_ATTR_KEYS.EDGE_TARGET: [2]})

    result = join_node_attrs_to_edges(node_attrs, edge_attrs, source_prefix="s_", target_prefix="t_")

    expected = pl.DataFrame(
        {DEFAULT_ATTR_KEYS.EDGE_SOURCE: [1], DEFAULT_ATTR_KEYS.EDGE_TARGET: [2], "s_x": [10], "t_x": [20]}
    )

    pl_testing.assert_frame_equal(result, expected)


def test_node_attr_to_edges_inner_join() -> None:
    node_attrs = pl.DataFrame({DEFAULT_ATTR_KEYS.NODE_ID: [1, 2], "value": [10, 20]})
    edge_attrs = pl.DataFrame({DEFAULT_ATTR_KEYS.EDGE_SOURCE: [1, 2, 3], DEFAULT_ATTR_KEYS.EDGE_TARGET: [2, 3, 4]})

    result = join_node_attrs_to_edges(node_attrs, edge_attrs, how="inner")

    expected = pl.DataFrame(
        {
            DEFAULT_ATTR_KEYS.EDGE_SOURCE: [1],
            DEFAULT_ATTR_KEYS.EDGE_TARGET: [2],
            "source_value": [10],
            "target_value": [20],
        }
    )

    pl_testing.assert_frame_equal(result, expected)

import polars as pl
from polars._typing import JoinStrategy

from tracksdata.constants import DEFAULT_ATTR_KEYS


def join_node_attrs_to_edges(
    node_attrs: pl.DataFrame,
    edge_attrs: pl.DataFrame,
    node_id_key: str = DEFAULT_ATTR_KEYS.NODE_ID,
    source_key: str = DEFAULT_ATTR_KEYS.EDGE_SOURCE,
    target_key: str = DEFAULT_ATTR_KEYS.EDGE_TARGET,
    source_prefix: str = "source_",
    target_prefix: str = "target_",
    how: JoinStrategy = "left",
) -> pl.DataFrame:
    """
    Add node attributes to edge attributes by joining on the node ID.

    Parameters
    ----------
    node_attrs : pl.DataFrame
        Node attributes.
    edge_attrs : pl.DataFrame
        Edge attributes.
    node_id_key : str, optional
        The key of the node ID column.
    source_key : str, optional
        The key of the source column.
    target_key : str, optional
        The key of the target column.
    source_prefix : str, optional
        The prefix of the source column.
    target_prefix : str, optional
        The prefix of the target column.
    how : JoinStrategy, optional
        The join type, where the "left" dataframe is the edge attributes and
        the "right" dataframe is the node attributes.

    Returns
    -------
    pl.DataFrame
        Edge attributes with node attributes added.

    Examples
    --------
    ```python
    node_attrs = pl.DataFrame({"node_id": [1, 2, 3], "a": [1, 2, 3], "b": [4, 5, 6]})
    edge_attrs = pl.DataFrame({"source": [1, 2], "target": [2, 3]})
    node_attr_to_edges(node_attrs, edge_attrs)
    # shape: (2, 5)
    # ┌──────────┬──────────┬──────────┬──────────┬────────────────┬────────────────┐
    # │ source_a ┆ source_b ┆ target_a ┆ target_b ┆ source_node_id ┆ target_node_id │
    # │ ---      ┆ ---      ┆ ---      ┆ ---      ┆ ---            ┆ ---            │
    # │ i64      ┆ i64      ┆ i64      ┆ i64      ┆ i64            ┆ i64            │
    # ╞══════════╪══════════╪══════════╪══════════╪════════════════╪════════════════╡
    # │ 1        ┆ 4        ┆ 2        ┆ 5        ┆ 1              ┆ 2              │
    # │ 2        ┆ 5        ┆ 3        ┆ 6        ┆ 2              ┆ 3              │
    # └──────────┴──────────┴──────────┴──────────┴────────────────┴────────────────┘
    ```
    """
    node_attr_keys = node_attrs.columns
    node_attr_keys.remove(node_id_key)

    source_mapping = dict(zip(node_attr_keys, [f"{source_prefix}{c}" for c in node_attr_keys], strict=False))
    target_mapping = dict(zip(node_attr_keys, [f"{target_prefix}{c}" for c in node_attr_keys], strict=False))

    edge_attrs = edge_attrs.join(
        node_attrs.select(node_id_key, *node_attr_keys).rename(source_mapping),
        left_on=source_key,
        right_on=node_id_key,
        how=how,
    ).join(
        node_attrs.select(node_id_key, *node_attr_keys).rename(target_mapping),
        left_on=target_key,
        right_on=node_id_key,
        how=how,
    )

    return edge_attrs

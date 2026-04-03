"""Utilities for shifting cell divisions in tracking graphs."""

from __future__ import annotations

import polars as pl

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import BaseGraph
from tracksdata.utils._dtypes import polars_dtype_to_numpy_dtype


def _avg_df(df: pl.DataFrame, non_numeric_idx: int = 0) -> dict:
    """
    Average a DataFrame column-wise into a single dict.

    - Numeric scalars → mean (integer-rounded when the dtype is integer).
    - Array columns  → element-wise mean, preserving the numpy dtype.
    - Everything else → value at row ``non_numeric_idx`` (0 = first, -1 = last).
    """
    result = {}
    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        if dtype.is_numeric():
            mean_val = series.mean()
            result[col] = round(mean_val) if dtype.is_integer() else mean_val
        elif isinstance(dtype, pl.Array):
            np_dtype = polars_dtype_to_numpy_dtype(dtype)
            result[col] = series.to_numpy().mean(axis=0).astype(np_dtype)
        else:
            result[col] = series[non_numeric_idx]
    return result


def _avg_node_attrs(graph: BaseGraph, node_ids: list[int], non_numeric_idx: int = 0) -> dict:
    """
    Average numeric node attributes across nodes; use first value for non-numeric.

    Parameters
    ----------
    graph : BaseGraph
        The graph to average the node attributes of.
    node_ids : list[int]
        The IDs of the nodes to average the attributes of.
    non_numeric_idx : int, optional
        The index of the non-numeric attribute to use. Defaults to 0.

    Returns
    -------
    dict
        The averaged node attributes.
    """
    df = graph.filter(node_ids=node_ids).node_attrs().drop(DEFAULT_ATTR_KEYS.NODE_ID)
    return _avg_df(df, non_numeric_idx=non_numeric_idx)


def _get_edge_custom_attrs(graph: BaseGraph, source_id: int, target_id: int) -> dict:
    """Get edge attributes excluding the system keys (edge_id, source_id, target_id)."""
    eid = graph.edge_id(source_id, target_id)
    row = (
        graph.edge_attrs()
        .filter(pl.col(DEFAULT_ATTR_KEYS.EDGE_ID) == eid)
        .drop([DEFAULT_ATTR_KEYS.EDGE_ID, DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET])
    )
    if len(row) == 0:
        return {}
    return row.rows(named=True)[0]


def _avg_edge_attrs(attrs_list: list[dict]) -> dict:
    """Average numeric edge attributes; use first value for non-numeric."""
    non_empty = [d for d in attrs_list if d]
    if not non_empty:
        return {}
    if len(non_empty) == 1:
        return dict(non_empty[0])
    return _avg_df(pl.DataFrame(non_empty))


def _shift_division_ahead_once(graph: BaseGraph, node_id: int) -> int:
    """
    Shift a division one frame ahead in-place.

    The two children are merged into a single node whose attributes are the
    average of both children. The merged node replaces the two children and
    inherits all their successors.

    Parameters
    ----------
    graph : BaseGraph
        The graph to modify in-place.
    node_id : int
        The dividing node (must have exactly 2 successors).

    Returns
    -------
    int
        The ID of the new merged node (the new dividing position).
    """
    children = graph.successors(node_id)
    if len(children) != 2:
        raise ValueError(f"Node {node_id} must have exactly 2 children to shift division ahead, found {len(children)}.")

    c1, c2 = children[0], children[1]

    # Collect grandchildren and their edge attrs before modifying the graph
    grand_c1 = graph.successors(c1)
    grand_c2 = graph.successors(c2)
    # Use a dict so duplicate grandchildren (same node in both subtrees) are de-duplicated;
    # edges from c2 take precedence for any shared grandchild.
    grandchild_edges: dict[int, dict] = {g: _get_edge_custom_attrs(graph, c1, g) for g in grand_c1}
    grandchild_edges.update({g: _get_edge_custom_attrs(graph, c2, g) for g in grand_c2})

    # Averaged edge attrs for the edge from node_id to the new merged node
    edge_node_to_merged = _avg_edge_attrs(
        [
            _get_edge_custom_attrs(graph, node_id, c1),
            _get_edge_custom_attrs(graph, node_id, c2),
        ]
    )

    # Create merged node with averaged attributes
    merged_id = graph.add_node(_avg_node_attrs(graph, [c1, c2]))

    # Connect original node → merged node.
    # Copy the dict because add_edge mutates it by adding edge_id.
    graph.add_edge(node_id, merged_id, dict(edge_node_to_merged))

    # Connect merged node → all grandchildren
    for g, attrs in grandchild_edges.items():
        graph.add_edge(merged_id, g, dict(attrs))

    # Remove children (also removes all edges incident to them)
    graph.remove_node(c1)
    graph.remove_node(c2)

    return merged_id


def _shift_division_behind_once(graph: BaseGraph, node_id: int) -> int:
    """
    Shift a division one frame behind in-place.

    The original dividing node is replaced by two new nodes whose attributes
    are linearly interpolated between the parent and each respective child.
    The parent node thereby becomes the new division point.

    Parameters
    ----------
    graph : BaseGraph
        The graph to modify in-place.
    node_id : int
        The dividing node (must have exactly 1 predecessor and 2 successors).

    Returns
    -------
    int
        The ID of the parent node, which is now the new dividing position.
    """
    parents = graph.predecessors(node_id)
    if len(parents) != 1:
        raise ValueError(f"Node {node_id} must have exactly 1 parent to shift division behind, found {len(parents)}.")
    children = graph.successors(node_id)
    if len(children) != 2:
        raise ValueError(
            f"Node {node_id} must have exactly 2 children to shift division behind, found {len(children)}."
        )

    parent = parents[0]
    c1, c2 = children[0], children[1]

    # Save edge attrs before modifying the graph
    edge_parent_to_node = _get_edge_custom_attrs(graph, parent, node_id)
    edge_node_to_c1 = _get_edge_custom_attrs(graph, node_id, c1)
    edge_node_to_c2 = _get_edge_custom_attrs(graph, node_id, c2)

    # Create two replacement nodes, each interpolated between parent and its respective child
    d1 = graph.add_node(_avg_node_attrs(graph, [parent, c1], non_numeric_idx=1))
    d2 = graph.add_node(_avg_node_attrs(graph, [parent, c2], non_numeric_idx=1))

    # Connect parent → new nodes (reusing the original parent → divider edge attrs).
    # Copy the dict before each call because add_edge mutates it by adding edge_id.
    graph.add_edge(parent, d1, dict(edge_parent_to_node))
    graph.add_edge(parent, d2, dict(edge_parent_to_node))

    # Connect new nodes → original children
    graph.add_edge(d1, c1, dict(edge_node_to_c1))
    graph.add_edge(d2, c2, dict(edge_node_to_c2))

    # Remove original dividing node (also removes parent→node, node→c1, node→c2 edges)
    graph.remove_node(node_id)

    return parent


def shift_division(graph: BaseGraph, node_id: int, frames: int) -> BaseGraph:
    """
    Move a dividing node forward or backward in time by the given number of frames.

    A dividing node is a node with exactly two successors (children).

    **Moving ahead** (positive ``frames``):
        The two children are merged into a single averaged node linked to the
        original dividing node, while the original children are removed.
        Their successors are inherited by the merged node, which becomes the new
        dividing position for the next iteration.

    **Moving behind** (negative ``frames``):
        Two new nodes replace the original dividing node.  Their attributes are
        linearly interpolated between the parent (predecessor) and each
        respective child (successor).  The parent then becomes the new dividing
        position for the next iteration.

    Parameters
    ----------
    graph : BaseGraph
        The input graph. It is not modified.
    node_id : int
        The ID of the dividing node to shift.
    frames : int
        Number of frames to move the division.  Positive shifts ahead in time,
        negative shifts behind.

    Returns
    -------
    BaseGraph
        A copy of the input graph with the division shifted.

    Raises
    ------
    ValueError
        If the node does not have exactly 2 children, or if shifting behind and
        the node does not have exactly 1 parent.

    Examples
    --------
    Move a division one frame earlier:

    ```python
    new_graph = shift_division(graph, node_id=5, frames=-1)
    ```

    Move a division two frames later:

    ```python
    new_graph = shift_division(graph, node_id=5, frames=2)
    ```
    """
    new_graph = graph.copy()

    if frames == 0:
        return new_graph

    current_node = node_id

    if frames > 0:
        for _ in range(frames):
            current_node = _shift_division_ahead_once(new_graph, current_node)
    else:
        for _ in range(-frames):
            current_node = _shift_division_behind_once(new_graph, current_node)

    return new_graph

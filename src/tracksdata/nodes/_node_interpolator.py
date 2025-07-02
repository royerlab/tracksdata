import logging
import math
from copy import deepcopy
from typing import Any, Protocol

from tqdm import tqdm

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges._generic_edges import GenericNodeFunctionEdgeAttrs
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes._base_nodes import BaseNodesOperator
from tracksdata.nodes._mask import bbox_interpolation_offset
from tracksdata.utils._logging import LOG


class NodeInterpolationFunc(Protocol):
    def __call__(
        self,
        src_attrs: dict[str, Any],
        tgt_attrs: dict[str, Any],
        new_t: int,
        delta_t: int,
    ) -> dict[str, Any]: ...


def _replace_if_exists(
    new_attrs: dict[str, Any],
    attr_key: str,
    value: Any,
) -> None:
    """
    Replace the value of an attribute if it exists in the new attributes.
    """
    if attr_key in new_attrs:
        new_attrs[attr_key] = value


def default_node_interpolation(
    src_attrs: dict[str, Any],
    tgt_attrs: dict[str, Any],
    new_t: int,
    delta_t: int,
) -> dict[str, Any]:
    """
    Default node interpolator.
    Interpolates the 'mask', 'z', 'y', 'x' attributes between the source and target nodes.

    Parameters
    ----------
    src_attrs : dict[str, Any]
        Source node attributes.
    tgt_attrs : dict[str, Any]
        Target node attributes.
    new_t : int
        Time point of the new node.
    delta_t : int
        Current delta time.

    Returns
    -------
    dict[str, Any]
        Interpolated node.
    """
    new_attrs = deepcopy(tgt_attrs)
    new_attrs.pop(DEFAULT_ATTR_KEYS.NODE_ID, None)

    new_attrs[DEFAULT_ATTR_KEYS.T] = new_t

    t_tgt = tgt_attrs[DEFAULT_ATTR_KEYS.T]
    w = (t_tgt - new_t) / delta_t

    if w < 0 or w > 1:
        raise ValueError(f"w = {w} is not between 0 and 1")

    ndim = new_attrs[DEFAULT_ATTR_KEYS.MASK].mask.ndim
    new_bbox = new_attrs[DEFAULT_ATTR_KEYS.MASK].bbox

    # updating bounding box
    offset = bbox_interpolation_offset(
        tgt_bbox=tgt_attrs[DEFAULT_ATTR_KEYS.MASK].bbox,
        src_bbox=src_attrs[DEFAULT_ATTR_KEYS.MASK].bbox,
        w=w,
    )

    new_bbox[ndim:] = new_bbox[ndim:] + offset
    new_bbox[:ndim] = new_bbox[:ndim] + offset
    new_attrs[DEFAULT_ATTR_KEYS.MASK].bbox = new_bbox

    for o, attr_key in zip(offset[::-1], ["x", "y", "z"], strict=False):
        new_attrs[attr_key] += o

    if tgt_attrs.get(DEFAULT_ATTR_KEYS.TRACK_ID) != src_attrs.get(DEFAULT_ATTR_KEYS.TRACK_ID):
        new_attrs[DEFAULT_ATTR_KEYS.TRACK_ID] = -1

    _replace_if_exists(
        new_attrs,
        DEFAULT_ATTR_KEYS.SOLUTION,
        src_attrs.get(DEFAULT_ATTR_KEYS.SOLUTION, False) and tgt_attrs.get(DEFAULT_ATTR_KEYS.SOLUTION, False),
    )

    _replace_if_exists(
        new_attrs,
        DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
        -1,
    )
    _replace_if_exists(
        new_attrs,
        DEFAULT_ATTR_KEYS.MATCH_SCORE,
        0,
    )
    _replace_if_exists(
        new_attrs,
        DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK,
        False,
    )

    return new_attrs


class NodeInterpolationEdgeAttrs(Protocol):
    """
    Function to recompute the edge attributes between newly inserted nodes
    during node interpolation.
    """

    def __call__(
        self,
        long_edge: dict[str, Any],
        src_attrs: dict[str, Any],
        tgt_attrs: dict[str, Any],
        new_attrs: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]: ...


def default_node_interpolation_edge_attrs(
    long_edge: dict[str, Any],
    src_attrs: dict[str, Any],
    tgt_attrs: dict[str, Any],
    new_attrs: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """
    Default node interpolation edge attributes.
    By default, it computes:

        - `delta_t`: the absolute difference between the source and target time points
        - `edge_weight`: the Euclidean distance between the source and target nodes
        - `solution`: it removes the solution attribute from the source and target nodes
                      and adding to the new edges.

    Parameters
    ----------
    long_edge : dict[str, Any]
        Edge defining the interpolation.
    src_attrs : dict[str, Any]
        Source node attributes.
    tgt_attrs : dict[str, Any]
        Target node attributes.
    new_attrs : dict[str, Any]
        New node attributes.

    Returns
    -------
    tuple[dict[str, Any] | None, dict[str, Any] | None]
        Source to new node edge attributes and new to target edge attributes.
        Return `None` if no edge should be added.
    """

    new_edge_attrs = []

    for x, y in [(src_attrs, new_attrs), (new_attrs, tgt_attrs)]:
        new_edge_attrs.append(
            {
                "delta_t": abs(x[DEFAULT_ATTR_KEYS.T] - y[DEFAULT_ATTR_KEYS.T]),
                DEFAULT_ATTR_KEYS.EDGE_WEIGHT: math.sqrt(
                    sum((x.get(attr_key, 0.0) - y.get(attr_key, 0.0)) ** 2 for attr_key in ["z", "y", "x"])
                ),
            }
        )

    solution = long_edge.get(DEFAULT_ATTR_KEYS.SOLUTION, False)
    if solution is not None:
        new_edge_attrs[0][DEFAULT_ATTR_KEYS.SOLUTION] = solution
        new_edge_attrs[1][DEFAULT_ATTR_KEYS.SOLUTION] = solution

    return new_edge_attrs


class NodeInterpolator(BaseNodesOperator):
    """
    Interpolate nodes between non-consecutive time points (delta_t > 1).
    """

    def __init__(
        self,
        delta_t_key: str = "delta_t",
        show_progress: bool = True,
        node_interpolation_func: NodeInterpolationFunc = default_node_interpolation,
        edge_attrs_func: NodeInterpolationEdgeAttrs = default_node_interpolation_edge_attrs,
        validate_keys: bool = False,
    ):
        super().__init__(show_progress=show_progress)
        self.delta_t_key = delta_t_key
        self.node_interpolation_func = node_interpolation_func
        self.edge_attrs_func = edge_attrs_func
        self.validate_keys = validate_keys

    def add_nodes(self, graph: BaseGraph, *, t: None = None) -> None:
        if t is not None:
            raise ValueError("'t' must be None for node interpolation")

        if self.delta_t_key not in graph.edge_attr_keys:
            LOG.warning(
                "The key '%s' is not in graph.edge_attrs (%s). Inserting edge attribute `delta_t` using '%s' attribute",
                self.delta_t_key,
                graph.edge_attr_keys,
                DEFAULT_ATTR_KEYS.T,
            )
            GenericNodeFunctionEdgeAttrs(
                func=lambda x, y: abs(x - y),
                attr_keys=DEFAULT_ATTR_KEYS.T,
                output_key=self.delta_t_key,
                show_progress=self.show_progress,
            ).add_edge_attrs(graph)

        edge_attrs = graph.edge_attrs()
        long_edges = edge_attrs.filter(edge_attrs[self.delta_t_key] > 1)

        selected_node_ids = set(long_edges[DEFAULT_ATTR_KEYS.EDGE_SOURCE]) | set(
            long_edges[DEFAULT_ATTR_KEYS.EDGE_TARGET]
        )

        nodes = graph.node_attrs(
            node_ids=list(selected_node_ids),
        )
        nodes_by_id = {node[DEFAULT_ATTR_KEYS.NODE_ID]: node for node in nodes.iter_rows(named=True)}

        for long_edge in tqdm(
            list(long_edges.iter_rows(named=True)),
            disable=not self.show_progress,
            desc="Interpolating and adding nodes",
        ):
            delta_t = long_edge[self.delta_t_key]

            src_id = long_edge[DEFAULT_ATTR_KEYS.EDGE_SOURCE]
            tgt_id = long_edge[DEFAULT_ATTR_KEYS.EDGE_TARGET]

            src_attrs = nodes_by_id[src_id]
            tgt_attrs = nodes_by_id[tgt_id]

            while delta_t > 1:
                new_node_attrs = self.node_interpolation_func(
                    src_attrs=src_attrs,
                    tgt_attrs=tgt_attrs,
                    new_t=tgt_attrs[DEFAULT_ATTR_KEYS.T] - 1,
                    delta_t=delta_t,
                )

                new_node_id = graph.add_node(new_node_attrs, validate_keys=self.validate_keys)
                nodes_by_id[new_node_id] = new_node_attrs

                src_to_new_edge_attrs, new_to_tgt_edge_attrs = self.edge_attrs_func(
                    long_edge=long_edge,
                    src_attrs=src_attrs,
                    tgt_attrs=tgt_attrs,
                    new_attrs=new_node_attrs,
                )

                if src_to_new_edge_attrs is not None:
                    graph.add_edge(
                        source_id=src_id,
                        target_id=new_node_id,
                        attrs=src_to_new_edge_attrs,
                        validate_keys=self.validate_keys,
                    )

                if new_to_tgt_edge_attrs is not None:
                    graph.add_edge(
                        source_id=new_node_id,
                        target_id=tgt_id,
                        attrs=new_to_tgt_edge_attrs,
                        validate_keys=self.validate_keys,
                    )

                if long_edge.get(DEFAULT_ATTR_KEYS.SOLUTION, False):
                    graph.update_edge_attrs(
                        attrs={DEFAULT_ATTR_KEYS.SOLUTION: False},
                        edge_ids=[long_edge[DEFAULT_ATTR_KEYS.EDGE_ID]],
                    )

                if LOG.isEnabledFor(logging.INFO):
                    LOG.info("s -> t (before): %s", long_edge[DEFAULT_ATTR_KEYS.SOLUTION])
                    LOG.info(
                        "s -> t (after): %s",
                        graph.edge_attrs(attr_keys=[DEFAULT_ATTR_KEYS.SOLUTION])
                        .filter(edge_id=long_edge[DEFAULT_ATTR_KEYS.EDGE_ID])
                        .to_dicts()[0],
                    )
                    LOG.info("s -> n: %s", src_to_new_edge_attrs)
                    LOG.info("n -> t: %s", new_to_tgt_edge_attrs)

                delta_t -= 1
                tgt_attrs = new_node_attrs
                tgt_id = new_node_id
                # replacing long_edge by new shorter but still long edge
                long_edge = src_to_new_edge_attrs.copy()

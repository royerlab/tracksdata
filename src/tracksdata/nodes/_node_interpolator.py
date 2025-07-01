import math
from copy import deepcopy
from typing import Any, Protocol

from tqdm import tqdm

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges._generic_edges import GenericNodeFunctionEdgeAttrs
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes._base_nodes import BaseNodesOperator
from tracksdata.utils._logging import LOG


class NodeInterpolationFunc(Protocol):
    def __call__(
        self,
        src_attrs: dict[str, Any],
        tgt_attrs: dict[str, Any],
        new_t: int,
        delta_t: int,
    ) -> dict[str, Any]: ...


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
    bbox = new_attrs[DEFAULT_ATTR_KEYS.MASK].bbox

    for i, attr_key in enumerate(["x", "y", "z"][:ndim]):
        if attr_key not in src_attrs or attr_key not in tgt_attrs:
            continue
        # TODO: make this part more clear
        src_val = src_attrs[attr_key]
        tgt_val = tgt_attrs[attr_key]
        new_val = w * src_val + (1 - w) * tgt_val
        new_attrs[attr_key] = new_val
        offset = round(new_val - tgt_val) - 1
        bbox[ndim - i - 1] = bbox[ndim - i - 1] + offset
        bbox[2 * ndim - i - 1] = bbox[2 * ndim - i - 1] + offset

    new_attrs[DEFAULT_ATTR_KEYS.MASK].bbox = bbox

    return new_attrs


class NodeInterpolationEdgeAttrs(Protocol):
    """
    Function to recompute the edge attributes between newly inserted nodes
    during node interpolation.
    """

    def __call__(
        self,
        src_attrs: dict[str, Any],
        tgt_attrs: dict[str, Any],
    ) -> dict[str, Any]: ...


def default_node_interpolation_edge_attrs(
    src_attrs: dict[str, Any],
    tgt_attrs: dict[str, Any],
) -> dict[str, Any]:
    """
    Default node interpolation edge attributes.
    By default, it computes:

        - `delta_t`: the absolute difference between the source and target time points
        - `edge_weight`: the Euclidean distance between the source and target nodes

    Parameters
    ----------
    src_attrs : dict[str, Any]
        Source node attributes.
    tgt_attrs : dict[str, Any]
        Target node attributes.

    Returns
    -------
    dict[str, Any]
        Node interpolation edge attributes.
    """
    return {
        "delta_t": abs(src_attrs[DEFAULT_ATTR_KEYS.T] - tgt_attrs[DEFAULT_ATTR_KEYS.T]),
        DEFAULT_ATTR_KEYS.EDGE_WEIGHT: math.sqrt(
            sum((src_attrs.get(attr_key, 0.0) - tgt_attrs.get(attr_key, 0.0)) ** 2 for attr_key in ["z", "y", "x"])
        ),
    }


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
    ):
        super().__init__(show_progress=show_progress)
        self.delta_t_key = delta_t_key
        self.node_interpolation_func = node_interpolation_func
        self.edge_attrs_func = edge_attrs_func

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

        edge_attrs = graph.edge_attrs(attr_keys=[self.delta_t_key])
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
            desc="Interpolating nodes",
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

                new_node_id = graph.add_node(new_node_attrs)
                nodes_by_id[new_node_id] = new_node_attrs

                graph.add_edge(
                    source_id=src_id,
                    target_id=new_node_id,
                    attrs=self.edge_attrs_func(
                        src_attrs=src_attrs,
                        tgt_attrs=new_node_attrs,
                    ),
                )

                graph.add_edge(
                    source_id=new_node_id,
                    target_id=tgt_id,
                    attrs=self.edge_attrs_func(
                        src_attrs=new_node_attrs,
                        tgt_attrs=tgt_attrs,
                    ),
                )

                delta_t -= 1
                tgt_attrs = new_node_attrs

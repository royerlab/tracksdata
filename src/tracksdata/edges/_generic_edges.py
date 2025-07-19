from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from tracksdata.attrs import NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges._base_edge_attrs import BaseEdgeAttrsOperator
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.utils._logging import LOG


class GenericFuncEdgeAttrs(BaseEdgeAttrsOperator):
    """
    Add weights to the edges of the graph based on the output of a function.

    When provided multiple attribute keys, the function should take a dict
    with the keys as values for each node.

    When provided a single attribute key, the function should take the value
    for each node.

    For example, if the function is ``func(source_attr, target_attr)``,
    and the attribute keys are ``["a", "b"]``, then the function should be
    ``func({"a": 1, "b": 2}, {"a": 3, "b": 4})``.

    For a single attribute key "a", the function should take a single value
    for each node, as ``func(1, 3)``.

    Parameters
    ----------
    func : Callable[[dict[str, Any] | Any, dict[str, Any] | Any], Any]
        The function to apply to the source and target attributes.
    attr_keys : Sequence[str] | str
        The keys of the attributes to pass to the function.
    output_key : str
        The key to store the output of the function.
    """

    output_key: str

    def __init__(
        self,
        func: Callable[[dict[str, Any] | Any, dict[str, Any] | Any], Any],
        attr_keys: Sequence[str] | str,
        output_key: str,
    ) -> None:
        super().__init__(output_key=output_key)
        self.attr_keys = attr_keys
        self.func = func

    def _init_edge_attrs(self, graph: BaseGraph) -> None:
        """
        Initialize the edge attributes for the graph.
        """
        if self.output_key not in graph.edge_attr_keys:
            graph.add_edge_attr_key(self.output_key, default_value=-99999.0)

    def _edge_attrs_per_time(
        self,
        t: int,
        *,
        graph: BaseGraph,
    ) -> tuple[list[int], dict[str, list[Any]]]:
        """
        Add weights to the edges of the graph based on the output of a function
        for a specific time point.

        Parameters
        ----------
        t : int
            The time point to add weights for.
        graph : BaseGraph
            The graph to add weights to.
        """
        graph_filter = graph.filter(NodeAttr(DEFAULT_ATTR_KEYS.T) == t, include_targets=True)
        edges_df = graph_filter.edge_attrs(attr_keys=[])

        if len(edges_df) == 0:
            LOG.warning(f"No edges found for time point {t} to successors")
            return [], {}

        att_keys_list = self.attr_keys if isinstance(self.attr_keys, list) else [self.attr_keys]
        nodes_df = graph_filter.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, *att_keys_list])

        source_df = edges_df.rename({DEFAULT_ATTR_KEYS.EDGE_SOURCE: DEFAULT_ATTR_KEYS.NODE_ID}).join(
            nodes_df,
            on=DEFAULT_ATTR_KEYS.NODE_ID,
            how="left",
        )

        target_df = edges_df.rename({DEFAULT_ATTR_KEYS.EDGE_TARGET: DEFAULT_ATTR_KEYS.NODE_ID}).join(
            nodes_df,
            on=DEFAULT_ATTR_KEYS.NODE_ID,
            how="left",
        )

        for df in [source_df, target_df]:
            if len(df) != len(edges_df):
                raise ValueError(
                    f"Number of edges ({len(edges_df)}) and nodes ({len(df)}) do not match. Something went wrong."
                )

        weights = np.zeros(len(edges_df), dtype=np.float32)

        if isinstance(self.attr_keys, str):
            # faster than creating a dict
            for i, (source_attr, target_attr) in enumerate(
                zip(
                    source_df[self.attr_keys],
                    target_df[self.attr_keys],
                    strict=True,
                )
            ):
                weights[i] = self.func(source_attr, target_attr)
        else:
            # a bit more expensive to create a dict but more flexible
            for i, (source_attr, target_attr) in enumerate(
                zip(
                    source_df[self.attr_keys].iter_rows(named=True),
                    target_df[self.attr_keys].iter_rows(named=True),
                    strict=True,
                )
            ):
                weights[i] = self.func(source_attr, target_attr)

        return edges_df[DEFAULT_ATTR_KEYS.EDGE_ID].to_list(), {self.output_key: weights}

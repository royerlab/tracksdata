import abc
from collections.abc import Sequence
from typing import Any

from toolz import curry

from tracksdata.graph._base_graph import BaseGraph
from tracksdata.utils._multiprocessing import multiprocessing_apply


class BaseEdgesOperator(abc.ABC):
    """
    Base class indicating methods required to insert edges into a graph.
    It will interact with a `BaseGraph` to do so.
    """

    def __init__(self, output_key: Sequence[str] | str):
        self.output_key = output_key

    @abc.abstractmethod
    def _init_edge_attrs(self, graph: BaseGraph) -> None:
        """
        Initialize the edge attributes for the graph.
        """

    def add_edges(
        self,
        graph: BaseGraph,
        *,
        t: int | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Initialize the edges from nodes of given `time` to nodes in neighboring
        times (`time` + `\delta time`)

        Parameters
        ----------
        graph : BaseGraph
            The graph to initialize the edges in.
        t: int
            The time of the nodes to initialize the edges from.
        **kwargs: Any
            Additional keyword arguments to pass to the `add_edges` method.
        """
        self._init_edge_attrs(graph)

        if t is None:
            time_points = graph.time_points()
        else:
            time_points = [t]

        for edge_attrs in multiprocessing_apply(
            curry(self._add_edges_per_time, graph=graph, **kwargs),
            time_points,
            desc="Adding edges",
        ):
            graph.bulk_add_edges(edge_attrs)

    @abc.abstractmethod
    def _add_edges_per_time(
        self,
        t: int,
        *,
        graph: BaseGraph,
        **kwargs: Any,
    ) -> None:
        """
        Add edges to a graph at a given time point.

        Parameters
        ----------
        t : int
            The time point to add edges for.
        graph : BaseGraph
            The graph to add edges to.
        **kwargs : Any
            Additional keyword arguments to pass to the `_add_edges_per_time` method.
        """

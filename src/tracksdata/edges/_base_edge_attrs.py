import abc
from collections.abc import Sequence
from typing import Any

from toolz import curry

from tracksdata.graph._base_graph import BaseGraph
from tracksdata.utils._multiprocessing import multiprocessing_apply


class BaseEdgeAttrsOperator(abc.ABC):
    """
    Base class indicating methods required to add attributes to edges in a graph.
    It will interact with a `BaseGraph` to do so.
    """

    def __init__(self, output_key: Sequence[str] | str):
        self.output_key = output_key

    @abc.abstractmethod
    def _init_edge_attrs(self, graph: BaseGraph) -> None:
        """
        Initialize the edge attributes for the graph.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add attributes to.
        """

    def add_edge_attrs(
        self,
        graph: BaseGraph,
        *,
        t: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Add attributes to the edges of the graph for a given time or all time points.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add attributes to.
        t : int | None
            The time point to add attributes for. If None, add attributes for all time points.
        **kwargs : Any
            Additional keyword arguments to pass to the `_add_edge_attrs_per_time` method.
        """
        self._init_edge_attrs(graph)

        if t is None:
            time_points = graph.time_points()
        else:
            time_points = [t]

        for edge_ids, edge_attrs in multiprocessing_apply(
            func=curry(self._edge_attrs_per_time, graph=graph, **kwargs),
            sequence=time_points,
            desc="Adding edge attributes",
        ):
            graph.update_edge_attrs(edge_ids=edge_ids, attrs=edge_attrs)

    @abc.abstractmethod
    def _edge_attrs_per_time(
        self,
        t: int,
        *,
        graph: BaseGraph,
        **kwargs: Any,
    ) -> tuple[list[int], dict[str, list[Any]]]:
        """
        Add attributes to edges of a graph at a given time point.

        Parameters
        ----------
        t : int
            The time point to add attributes for.
        graph : BaseGraph
            The graph to add attributes to.
        **kwargs : Any
            Additional keyword arguments.
        """

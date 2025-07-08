import abc
from collections.abc import Sequence
from typing import Any

from toolz import curry

from tracksdata.graph._base_graph import BaseGraph
from tracksdata.utils._multiprocessing import multiprocessing_apply


class BaseNodeAttrsOperator(abc.ABC):
    """
    Base class indicating methods required to add attributes to nodes in a graph.
    It will interact with a `BaseGraph` to do so.
    """

    def __init__(
        self,
        output_key: Sequence[str] | str,
    ) -> None:
        self.output_key = output_key

    @abc.abstractmethod
    def _init_node_attrs(self, graph: BaseGraph) -> None:
        """
        Initialize the node attributes for the graph.
        """

    def add_node_attrs(
        self,
        graph: BaseGraph,
        *,
        t: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Add attributes to nodes of a graph.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add attributes to.
        t : int | None
            The time point to add attributes for. If None, add attributes for all time points.
        **kwargs : Any
            Additional keyword arguments to pass to the `_add_node_attrs_per_time` method.
        """
        self._init_node_attrs(graph)

        if t is None:
            time_points = graph.time_points()
        else:
            time_points = [t]

        for node_ids, node_attrs in multiprocessing_apply(
            func=curry(self._node_attrs_per_time, graph=graph, **kwargs),
            sequence=time_points,
            desc="Adding node attributes",
        ):
            graph.update_node_attrs(node_ids=node_ids, attrs=node_attrs)

    @abc.abstractmethod
    def _node_attrs_per_time(
        self,
        t: int,
        *,
        graph: BaseGraph,
        **kwargs: Any,
    ) -> tuple[list[int], dict[str, list[Any]]]:
        """
        Add attributes to nodes of a graph at a given time point.

        Parameters
        ----------
        t : int
            The time point to add attributes for. If None, add attributes for all time points.
        graph : BaseGraph
            The graph to add attributes to.
        **kwargs : Any
            Additional keyword arguments to pass to the `_add_node_attrs_per_time` method.

        Returns
        -------
        tuple[list[int], dict[str, list[Any]]]
            The node ids and the attributes to add to the graph.
        """

import abc
from typing import Any

from tracksdata.graph._base_graph import BaseGraph


class BaseNodesOperator(abc.ABC):
    """
    Base class indicating methods required to insert nodes into a graph.
    It will interact with a `BaseGraph` to do so.
    """

    @abc.abstractmethod
    def add_nodes(
        self,
        graph: BaseGraph,
        *,
        t: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the nodes for a given time or all time points.

        Parameters
        ----------
        graph : BaseGraph
            The graph to initialize the nodes in.
        t : int | None
            The time point to add nodes for. If None, add nodes for all time points.
        **kwargs : Any
            Additional keyword arguments to pass to the `_add_nodes_per_time` method.
        """

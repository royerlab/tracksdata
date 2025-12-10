import abc
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from tracksdata.graph._graph_view import GraphView


class BaseFilter(abc.ABC):
    def __init__(self) -> None:
        self._cache: dict[tuple[str, Any, Any], Any] = {}

    def clear_cache(self) -> None:
        """
        Clear the cache of the filter, otherwise methods are cached and won't return the latest data.
        """
        self._cache.clear()

    def is_empty(self) -> bool:
        return len(self.node_ids()) == 0

    @abc.abstractmethod
    def node_attrs(self, attr_keys: list[str] | None = None, unpack: bool = False) -> pl.DataFrame:
        """
        Get the attributes of the nodes resulting from the filter.
        """

    @abc.abstractmethod
    def edge_attrs(self, attr_keys: list[str] | None = None, unpack: bool = False) -> pl.DataFrame:
        """
        Get the attributes of the edges resulting from the filter.
        """

    @abc.abstractmethod
    def node_ids(self) -> list[int]:
        """
        Get the ids of the nodes resulting from the filter.
        """

    @abc.abstractmethod
    def edge_ids(self) -> list[int]:
        """
        Get the ids of the edges resulting from the filter.
        """

    @abc.abstractmethod
    def subgraph(
        self,
        node_attr_keys: list[str] | None = None,
        edge_attr_keys: list[str] | None = None,
    ) -> "GraphView":
        """
        Get a subgraph of the graph resulting from the filter.
        """

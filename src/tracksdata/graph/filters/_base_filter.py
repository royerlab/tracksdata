import abc
import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import polars as pl

from tracksdata.utils._logging import LOG

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
        node_attr_keys: Sequence[str] | str | None = None,
        edge_attr_keys: Sequence[str] | str | None = None,
    ) -> "GraphView":
        """
        Get a subgraph of the graph resulting from the filter.
        """


def _make_hashable(obj: Any) -> Any:
    if isinstance(obj, tuple | list):
        return tuple(_make_hashable(o) for o in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((_make_hashable(k), _make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, set):
        return frozenset(_make_hashable(o) for o in obj)
    elif hasattr(obj, "__dict__"):
        return _make_hashable(vars(obj))  # for custom objects
    elif hasattr(obj, "__hash__") and obj.__hash__:
        return obj
    else:
        return repr(obj)  # fallback for anything else


F = TypeVar("F", bound=Callable[..., Any])


def cache_method(func: F) -> F:
    """
    Cache the result of a method.

    Parameters
    ----------
    func : Callable[..., Any]
        The method to cache.

    Returns
    -------
    Callable[..., Any]
        The wrapped method.
    """

    @functools.wraps(func)
    def _wrapper(self: BaseFilter, *args: Any, **kwargs: Any) -> Any:
        try:
            key = (
                func.__name__,
                _make_hashable(args),
                _make_hashable(kwargs),
            )
        except Exception as e:
            LOG.warning(f"_{func.__name__} failed to hash {args} and {kwargs}:\n{e}")
            return func(self, *args, **kwargs)

        if key in self._cache:
            return self._cache[key]
        result = func(self, *args, **kwargs)
        self._cache[key] = result
        return result

    return _wrapper

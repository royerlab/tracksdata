import functools
from collections.abc import Callable
from typing import Any, TypeVar

from tracksdata.utils._logging import LOG


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
    def _wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self, "_cache"):
            LOG.warning(
                f"{self.__class__.__name__} has no `_cache` attribute.\n"
                "Adding for this class instance. The code should be refactored to add `_cache` to the class itself."
            )
            self._cache = {}

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

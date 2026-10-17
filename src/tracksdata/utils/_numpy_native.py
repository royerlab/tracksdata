"""Coercion of numpy scalars into the native Python scalars that other libraries understand."""

from collections.abc import Sequence
from typing import Any

import numpy as np


def is_int_like(value: Any) -> bool:
    """
    Whether ``value`` is a single integer, either a Python ``int`` or a numpy integer.

    ``np.int64`` is not a subclass of ``int``, so a plain ``isinstance(value, int)``
    check rejects the numpy integers that come out of nearly every numpy or polars
    operation. Use this wherever a scalar id must be told apart from a sequence of ids.
    """
    return isinstance(value, int | np.integer)


def to_native(value: Any) -> Any:
    """
    Return the native Python equivalent of a numpy scalar, or ``value`` unchanged.

    Database drivers do not know about numpy's scalar types. ``sqlite3``, for example,
    falls back to the buffer protocol and binds ``np.int64(7)`` as its raw little-endian
    byte buffer (a BLOB), which never compares equal to an ``INTEGER`` column, so queries
    silently match no rows instead of raising. Numpy floats and strings happen to survive
    because they subclass their Python counterparts, which makes the corruption look
    selective.

    Parameters
    ----------
    value : Any
        The value to convert. Non-numpy values are returned as-is.

    Returns
    -------
    Any
        ``value.item()`` for numpy scalars, otherwise ``value``.
    """
    # `np.generic` is the base class of every numpy scalar, and excludes (0-dim)
    # arrays, which must be passed through untouched.
    if isinstance(value, np.generic):
        return value.item()
    return value


def to_native_list(values: Sequence[Any] | np.ndarray) -> list[Any]:
    """
    Return ``values`` as a list with every numpy scalar converted to native Python.

    Parameters
    ----------
    values : Sequence[Any] | np.ndarray
        The values to convert.

    Returns
    -------
    list[Any]
        A new list of native Python scalars.
    """
    if isinstance(values, np.ndarray):
        return values.tolist()
    return [to_native(v) for v in values]

import abc
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

ArrayIndex = ArrayLike | int | slice | tuple[ArrayLike | int | slice, ...]


class BaseReadOnlyArray(np.lib.mixins.NDArrayOperatorsMixin, abc.ABC):
    """
    Base class for read-only array-like objects.

    Arithmetic and comparison operators (e.g. `array_view == 0`,
    `array_view + np.ones(...)`) materialize the array content and
    delegate to the corresponding numpy ufunc via `__array_ufunc__`.
    """

    # NDArrayOperatorsMixin defines `__eq__`, which would otherwise reset
    # `__hash__` to None; keep the default identity hash.
    __hash__ = object.__hash__

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if kwargs.get("out") is not None:
            raise TypeError(f"`out` is not supported for read-only {type(self).__name__}.")
        inputs = tuple(np.asarray(x) if isinstance(x, BaseReadOnlyArray) else x for x in inputs)
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __len__(self) -> int:
        """Returns the length of the first dimension of the array."""
        return self.shape[0]

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the array."""
        return len(self.shape)

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the array."""

    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        """Returns the dtype of the array."""

    @abc.abstractmethod
    def __getitem__(self, index: ArrayIndex) -> ArrayLike:
        """Returns a slice of the array."""

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """Returns the total number of elements in the array."""


class BaseWritableArray(BaseReadOnlyArray):
    """
    Base class for writable array-like objects.
    """

    @abc.abstractmethod
    def __setitem__(
        self,
        index: ArrayIndex,
        value: ArrayLike,
    ) -> None:
        """Sets a slice of the array."""

    @abc.abstractmethod
    def commit(self) -> None:
        """Commits the changes to the array."""
        # TODO: @caroline @teun, should we have this?
        #       I'm concerned writing an array atomically will be problematic and slow.

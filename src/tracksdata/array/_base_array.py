import abc

import numpy as np
from numpy.typing import ArrayLike

ArrayIndex = ArrayLike | int | slice | tuple[ArrayLike | int | slice, ...]


class BaseReadOnlyArray(abc.ABC):
    """
    Base class for read-only array-like objects.
    """

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

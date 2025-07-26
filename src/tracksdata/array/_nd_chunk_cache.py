import itertools
from typing import Sequence, Tuple, Dict, Callable

import numpy as np

import itertools
from collections import OrderedDict
from typing import Callable, Dict, Sequence, Tuple

import numpy as np


class NDChunkCache:
    """
    N-dimensional full-volume cache with LRU eviction on the *time* axis.

    Parameters
    ----------
    compute_func : Callable[[int, Tuple[slice, ...], np.ndarray], None]
        User-supplied function painting one *chunk* given (t, chunk_slices) in the buffer.
    shape : Sequence[int]
        The absolute shape of a single time-point volume.
    chunk_shape : Sequence[int]
        Length of a chunk in every axis. The element count defines dimensionality.
    max_buffers : int
        Maximum number of time-points kept simultaneously.
    dtype : np.dtype, optional
        Stored dtype of buffers (default float32).
    """

    # ------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------
    def __init__(
        self,
        compute_func: Callable[[int, tuple[slice, ...], np.ndarray], None],
        shape: Sequence[int],
        chunk_shape: Sequence[int],
        max_buffers: int = 4,
        dtype=np.uint64,
    ):
        self.compute_func = compute_func
        self.shape: Tuple[int, ...] = tuple(shape)
        self.chunk_shape: Tuple[int, ...] = tuple(chunk_shape)
        self.ndim: int = len(self.shape)
        if len(self.shape) != len(self.chunk_shape):
            raise ValueError("`shape` and `chunk_shape` must have same length")

        # Grid size in chunks, e.g. (nz, ny, nx, …)
        # Use ceiling division to handle non-divisible shapes
        self.grid_shape: Tuple[int, ...] = tuple((fs + cs - 1) // cs for fs, cs in zip(self.shape, self.chunk_shape))

        self.max_buffers = max_buffers
        self.dtype = dtype

        # (LRU) mapping   t  ->  {"buffer": ndarray, "ready": boolean ndarray}
        self._store: "OrderedDict[int, Dict[str, np.ndarray]]" = OrderedDict()

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    def _ensure_buffer(self, t: int) -> Dict[str, np.ndarray]:
        """
        Return the dictionary holding buffer & ready-mask for time `t`,
        allocating (and evicting) if necessary.
        """
        if t in self._store:
            # Move to MRU position
            self._store.move_to_end(t)
            return self._store[t]

        # Evict oldest if over limit
        if len(self._store) >= self.max_buffers:
            self._store.popitem(last=False)

        # Allocate new buffer and boolean mask
        buf = np.zeros(self.shape, dtype=self.dtype)
        ready = np.zeros(self.grid_shape, dtype=bool)
        self._store[t] = {"buffer": buf, "ready": ready}
        return self._store[t]

    def _chunk_bounds(self, slices: Tuple[slice, ...]) -> Tuple[Tuple[int, int], ...]:
        """Return inclusive chunk-index bounds for every axis."""
        return tuple(
            (s.start // cs, (s.stop - 1) // cs) for s, cs in zip(slices, self.chunk_shape)
        )

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def get(self, time: int, volume_slicing: Tuple[slice | int, ...]) -> np.ndarray:
        """
        Retrieve data for time `t` and arbitrary dimensional slices.

        Returns a *view* into the cached full buffer (zero-copy).
        """
        if len(volume_slicing) != self.ndim:
            raise ValueError("Number of slices must equal dimensionality")
        volume_slicing_slices = tuple(
            slc if isinstance(slc, slice) else slice(slc, slc + 1) for slc in volume_slicing
        )

        store_entry = self._ensure_buffer(time)
        buf = store_entry["buffer"]
        ready = store_entry["ready"]

        # Which chunks are touched by this request?
        bounds = self._chunk_bounds(volume_slicing_slices)
        chunk_ranges = [range(lo, hi + 1) for lo, hi in bounds]

        # For every intersected chunk, compute if not ready
        for chunk_idx in itertools.product(*chunk_ranges):
            if ready[chunk_idx]:
                continue  # already filled

            # Absolute slice covering this chunk
            chunk_slc = tuple(
                slice(ci * cs, min((ci + 1) * cs, fs)) 
                for ci, cs, fs in zip(chunk_idx, self.chunk_shape, self.shape)
            )
            # Handle the case where chunk_slc exceeds volume_slices
            self.compute_func(time, chunk_slc, buf)
            ready[chunk_idx] = True

        # Return view on the big buffer
        return buf[volume_slicing]



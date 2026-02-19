import itertools
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


def _to_slice(slc: slice | int | Sequence[int]) -> slice:
    """Convert int or slice to slice."""
    if isinstance(slc, int):
        return slice(slc, slc + 1)
    elif isinstance(slc, slice):
        return slc
    elif isinstance(slc, Sequence):
        return slice(min(slc), max(slc) + 1)
    else:
        raise TypeError(f"Unsupported type for slicing: {type(slc)}")


@dataclass
class CacheEntry:
    buffer: np.ndarray
    ready: np.ndarray

    """
    A cache entry for a single time-point.

    Parameters
    ----------
    buffer : np.ndarray
        The buffer to storing the entire volume.
    ready : np.ndarray
        A boolean mask indicating which chunks have been computed.
    """


class NDChunkCache:
    """
    N-dimensional full-volume cache with LRU eviction on the *time* axis.

    Parameters
    ----------
    compute_func : Callable[[int, tuple[slice, ...], np.ndarray], None]
        User-supplied function painting one *chunk* given (t, chunk_slices) in the buffer.
    shape : Sequence[int]
        The absolute shape of a single time-point volume.
    chunk_shape : Sequence[int]
        Length of a chunk in every axis. The element count defines dimensionality.
    max_buffers : int
        Maximum number of time-points kept simultaneously.
    dtype : np.dtype, optional
        Stored dtype of buffers (default np.uint64).
    """

    def __init__(
        self,
        compute_func: Callable[[int, tuple[slice, ...], np.ndarray], None],
        shape: Sequence[int],
        chunk_shape: Sequence[int],
        buffer_cache_size: int = 4,
        dtype=np.uint64,
    ):
        self.compute_func = compute_func
        self.shape: tuple[int, ...] = tuple(shape)
        self.chunk_shape: tuple[int, ...] = tuple(chunk_shape)
        self.ndim: int = len(self.shape)
        if len(self.shape) != len(self.chunk_shape):
            raise ValueError("`shape` and `chunk_shape` must have same length")

        # Grid size in chunks, e.g. (nz, ny, nx, …)
        # Use ceiling division to handle non-divisible shapes
        self.grid_shape: tuple[int, ...] = tuple(
            (fs + cs - 1) // cs for fs, cs in zip(self.shape, self.chunk_shape, strict=True)
        )

        self.max_buffers = buffer_cache_size
        self.dtype = dtype

        # (LRU) mapping   t  ->  {"buffer": ndarray, "ready": boolean ndarray}
        self._store: OrderedDict[int, CacheEntry] = OrderedDict()

    def _ensure_buffer(self, t: int) -> CacheEntry:
        """
        Return the dictionary holding buffer & ready-mask for time `t`,
        allocating (and evicting) if necessary.

        Parameters
        ----------
        t : int
            The time point to retrieve data for.

        Returns
        -------
        CacheEntry
            The cache entry for the time point.
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
        self._store[t] = CacheEntry(buffer=buf, ready=ready)
        return self._store[t]

    def _chunk_bounds(self, slices: tuple[slice, ...]) -> tuple[tuple[int, int], ...]:
        """Return inclusive chunk-index bounds for every axis."""
        return tuple((s.start // cs, (s.stop - 1) // cs) for s, cs in zip(slices, self.chunk_shape, strict=True))

    def _chunk_slice(self, chunk_idx: tuple[int, ...]) -> tuple[slice, ...]:
        """Return absolute volume slices for a chunk index."""
        return tuple(
            slice(ci * cs, min((ci + 1) * cs, fs))
            for ci, cs, fs in zip(chunk_idx, self.chunk_shape, self.shape, strict=True)
        )

    def get(self, time: int, volume_slicing: tuple[slice | int | Sequence[int], ...]) -> np.ndarray:
        """
        Retrieve data for `time` and arbitrary dimensional slices.

        Parameters
        ----------
        time : int
            The time point to retrieve data for.
        volume_slicing : tuple[slice | int | Sequence[int], ...]
            Slicing specification for the volume dimensions.

        Returns
        -------
        np.ndarray
            A view into the cached full buffer (zero-copy).
        """
        if len(volume_slicing) != self.ndim:
            raise ValueError("Number of slices must equal dimensionality")

        volume_slicing_slices = tuple(_to_slice(slc) for slc in volume_slicing)

        store_entry = self._ensure_buffer(time)
        # Which chunks are touched by this request?
        bounds = self._chunk_bounds(volume_slicing_slices)
        chunk_ranges = [range(lo, hi + 1) for lo, hi in bounds]

        # For every intersected chunk, compute if not ready
        for chunk_idx in itertools.product(*chunk_ranges):
            if store_entry.ready[chunk_idx]:
                continue  # already filled

            # Absolute slice covering this chunk
            chunk_slc = self._chunk_slice(chunk_idx)
            # Handle the case where chunk_slc exceeds volume_slices
            self.compute_func(time, chunk_slc, store_entry.buffer)
            store_entry.ready[chunk_idx] = True

        # Return view on the big buffer
        return store_entry.buffer[volume_slicing]

    def invalidate(
        self,
        *,
        time: int | None = None,
        volume_slicing: tuple[slice | int | Sequence[int], ...] | None = None,
    ) -> None:
        """
        Invalidate a cached region.

        Parameters
        ----------
        time : int | None, optional
            Time point to invalidate. If None, applies to all currently cached times.
        volume_slicing : tuple[slice | int | Sequence[int], ...] | None, optional
            Volume region to invalidate. If None, invalidates the full volume for the selected times.
        """
        if time is None:
            times = list(self._store.keys())
        elif time in self._store:
            times = [time]
        else:
            return

        if volume_slicing is not None and len(volume_slicing) != self.ndim:
            raise ValueError("Number of slices must equal dimensionality")

        region_slices = None
        if volume_slicing is not None:
            region_slices = tuple(_to_slice(slc) for slc in volume_slicing)

        for t in times:
            store_entry = self._store[t]

            if region_slices is None:
                store_entry.ready.fill(False)
                store_entry.buffer.fill(0)
                continue

            clipped_slices = []
            for slc, size in zip(region_slices, self.shape, strict=True):
                start = max(0, slc.start)
                stop = min(size, slc.stop)
                if stop <= start:
                    return
                clipped_slices.append(slice(start, stop))

            bounds = self._chunk_bounds(tuple(clipped_slices))
            chunk_ranges = [range(lo, hi + 1) for lo, hi in bounds]
            for chunk_idx in itertools.product(*chunk_ranges):
                chunk_slc = self._chunk_slice(chunk_idx)
                store_entry.ready[chunk_idx] = False
                store_entry.buffer[chunk_slc] = 0

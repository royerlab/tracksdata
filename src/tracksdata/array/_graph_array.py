import numpy as np
from numpy.typing import ArrayLike

from tracksdata.array._base_array import ArrayIndex, BaseReadOnlyArray
from tracksdata.attrs import NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes._mask import Mask
from tracksdata.utils._dtypes import polars_dtype_to_numpy_dtype
from tracksdata.array._nd_chunk_cache import NDChunkCache

import numpy as np

DEFAULT_CHUNK_SIZE = 128
DEFAULT_DTYPE = np.int32

class GraphArrayView(BaseReadOnlyArray):
    """
    Class used to view the content of a graph as an array.

    The resulting graph behaves as a read-only numpy array,
    displaying arbitrary attributes inside their respective instance mask.

    The content is lazy loaded from the original data source as
    it's done with a [zarr.Array](https://zarr.readthedocs.io/en/stable/index.html)

    Parameters
    ----------
    graph : BaseGraph
        The graph to view as an array.
    shape : tuple[int, ...]
        The shape of the array.
    attr_key : str
        The attribute key to view as an array.
    offset : int | np.ndarray, optional
        The offset to apply to the array.
    """

    def __init__(
        self,
        graph: BaseGraph,
        shape: tuple[int, ...],
        attr_key: str,
        offset: int | np.ndarray = 0,
        chunk_shape: tuple[int] | None = None,
        max_buffers: int = 32,
    ):
        if attr_key not in graph.node_attr_keys:
            raise ValueError(f"Attribute key '{attr_key}' not found in graph. Expected '{graph.node_attr_keys}'")

        self.graph = graph
        self._shape = shape
        self._attr_key = attr_key
        self._offset = offset
        if chunk_shape is None:
            chunk_shape = tuple([DEFAULT_CHUNK_SIZE]*(len(shape) - 1))  # Default chunk shape
        self.max_buffers = max_buffers

        # Infer the dtype from the graph's attribute
        # TODO improve performance
        df = graph.node_attrs(attr_keys=[self._attr_key])
        if df.is_empty():
            dtype = DEFAULT_DTYPE
        else:
            dtype = polars_dtype_to_numpy_dtype(df[self._attr_key].dtype)
            # napari support for bool is limited
            if np.issubdtype(dtype, bool):
                dtype = np.uint8

        self._dtype = dtype

        self.cache = NDChunkCache(
            compute_func=self._fill_array,
            shape=shape[1:],
            chunk_shape=chunk_shape,
            max_buffers=max_buffers,
            dtype=np.dtype(self._dtype),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __getitem__(self, index: ArrayIndex) -> ArrayLike:
        if isinstance(index, tuple):
            time, volume_slicing = index[0], index[1:]
        else:  # if only 1 (time) is provided
            time = index
            volume_slicing = tuple()

        if isinstance(time, slice):  # if all time points are requested
            # XXX could be dask? should be benchmarked
            return np.stack(
                [
                    self.__getitem__((t,) + volume_slicing).copy()
                    for t in range(*time.indices(self.shape[0]))
                ]
            )
        else:
            try:
                time = time.item()  # convert from numpy.int to int
            except AttributeError:
                time = time

        # Extend the volume_slicing by filling in missing dimensions by slice(None)
        if len(volume_slicing) < self.ndim - 1:
            volume_slicing = volume_slicing + (slice(None),) * (self.ndim - 1 - len(volume_slicing))

        volume_slicing = list(volume_slicing)
        for i, slc in enumerate(volume_slicing):
            if isinstance(slc, slice):
                start = slc.start if slc.start is not None else 0
                stop = slc.stop if slc.stop is not None else self.shape[i + 1]
                step = slc.step if slc.step is not None else 1
                volume_slicing[i] = slice(start, stop, step)
    
        volume_slicing = tuple(volume_slicing)
        return self.cache.get(
            time=time,
            volume_slicing=volume_slicing,
        )

    def _fill_array(self, time: int, volume_slicing: ArrayIndex, buffer: np.ndarray) -> np.ndarray:
        # TODO handling the slices for volume_slicing
        graph_filter = self.graph.filter(NodeAttr(DEFAULT_ATTR_KEYS.T) == time)
        df = graph_filter.node_attrs(
            attr_keys=[self._attr_key, DEFAULT_ATTR_KEYS.MASK],
        )

        for mask, value in zip(df[DEFAULT_ATTR_KEYS.MASK], df[self._attr_key], strict=False):
            mask: Mask
            mask.paint_buffer(buffer, value, offset=self._offset)

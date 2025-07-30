import numpy as np
from numpy.typing import ArrayLike

from tracksdata.array._base_array import ArrayIndex, BaseReadOnlyArray
from tracksdata.attrs import NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.functional._mask import paint_mask_to_buffer
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.utils._dtypes import polars_dtype_to_numpy_dtype


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
    ):
        if attr_key not in graph.node_attr_keys:
            raise ValueError(f"Attribute key '{attr_key}' not found in graph. Expected '{graph.node_attr_keys}'")

        self.graph = graph
        self._shape = shape
        self._attr_key = attr_key
        self._offset = offset
        self._dtype = np.int32

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __getitem__(self, index: ArrayIndex) -> ArrayLike:
        # FIXME:
        # - just for testing, not final implementation
        # - make use of offset
        if isinstance(index, tuple):
            index = index[0]

        if isinstance(index, int):
            graph_filter = self.graph.filter(NodeAttr(DEFAULT_ATTR_KEYS.T) == index)

            if graph_filter.is_empty():
                return np.zeros(self.shape[1:], dtype=self.dtype)

            df = graph_filter.node_attrs(
                attr_keys=[self._attr_key, DEFAULT_ATTR_KEYS.MASK, DEFAULT_ATTR_KEYS.BBOX],
            )

            dtype = polars_dtype_to_numpy_dtype(df[self._attr_key].dtype)

            # napari support for bool is limited
            if np.issubdtype(dtype, bool):
                dtype = np.uint8

            self._dtype = dtype

            # TODO: reuse buffer
            buffer = np.zeros(self.shape[1:], dtype=self.dtype)

            for bbox, mask, value in zip(
                df[DEFAULT_ATTR_KEYS.BBOX],
                df[DEFAULT_ATTR_KEYS.MASK],
                df[self._attr_key],
                strict=True,
            ):
                paint_mask_to_buffer(buffer, bbox, mask, value, offset=self._offset)

            return buffer
        else:
            raise NotImplementedError("Not implemented")

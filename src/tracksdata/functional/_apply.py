import itertools
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TypeVar, overload

import numpy as np

from tracksdata.graph._base_graph import BaseGraph
from tracksdata.graph.filters._base_filter import BaseFilter
from tracksdata.utils._logging import LOG

T = TypeVar("T")  # per tile return type
R = TypeVar("R")  # reduce return type


MapFunc = Callable[[BaseFilter, BaseFilter], T]
ReduceFunc = Callable[[list[T]], R]


@dataclass
class TilingScheme:
    """
    Tiling scheme for the graph.
    Graph will be sliced with 'tile_shape' + 2 * 'overlap' per axis.

    Parameters
    ----------
    tile_shape : tuple[int, ...]
        The shape of the tile.
    overlap : tuple[int, ...]
        The overlap between tiles.
    attrs : list[str] | None, optional
        The attributes to include in the tile. If None, all attributes will be included.
        By default "t", "z", "y", "x" are included. If some columns are not present, they will be ignored.
    """

    tile_shape: tuple[int, ...]
    overlap: tuple[int, ...]
    attrs: list[str] | None = None

    def __post_init__(self) -> None:
        if len(self.tile_shape) != len(self.overlap):
            raise ValueError(
                "'TilingScheme.tile_shape' and 'TilingScheme.overlap' must have the same length, "
                f"got {len(self.tile_shape)} and {len(self.overlap)}"
            )

        if any(tile_shape <= 0 for tile_shape in self.tile_shape):
            raise ValueError(f"'TilingScheme.tile_shape' must be greater than 0, got {self.tile_shape}")

        if any(overlap < 0 for overlap in self.overlap):
            raise ValueError(f"'TilingScheme.overlap' must be non-negative, got {self.overlap}")

        if self.attrs is not None:
            if len(self.attrs) != len(self.tile_shape):
                raise ValueError(
                    f"'TilingScheme.attrs' must have the same length as 'TilingScheme.tile_shape', "
                    f"got {len(self.attrs)} and {len(self.tile_shape)}"
                )


@overload
def apply_tiled(
    graph: BaseGraph,
    tiling_scheme: TilingScheme,
    func: MapFunc,
    *,
    agg_func: None,
) -> Iterator[T]: ...


@overload
def apply_tiled(
    graph: BaseGraph,
    tiling_scheme: TilingScheme,
    func: MapFunc,
    *,
    agg_func: ReduceFunc,
) -> R: ...


def _get_tiles_corner(
    start: Sequence[int],
    end: Sequence[int],
    tiling_scheme: TilingScheme,
) -> list[tuple[int, ...]]:
    # eps is required when min == max, otherwise np.arange is empty
    eps = 1e-6
    tiles_corner = list(
        itertools.product(
            *[
                np.arange(s, e + eps * (s == e), t).tolist()
                for s, e, t in zip(start, end, tiling_scheme.tile_shape, strict=True)
            ]
        )
    )

    LOG.info("Created %s tiles", len(tiles_corner))
    return tiles_corner


def _yield_apply_tiled(
    graph: BaseGraph,
    tiling_scheme: TilingScheme,
    func: MapFunc,
) -> Iterator[T] | R:
    """
    See `apply_tiled` for more details.
    """
    # if agg_func is provided, we need to reduce the results
    if tiling_scheme.attrs is None:
        # default attrs
        attr_keys = ["t", "z", "y", "x"]
        attr_keys = [a for a in attr_keys if a in graph.node_attr_keys]
    else:
        attr_keys = tiling_scheme.attrs

    spatial_filter = graph.spatial_filter(attr_keys)
    nodes_df = graph.node_attrs(attr_keys=attr_keys)

    start = nodes_df.min().transpose().to_series().to_list()
    end = nodes_df.max().transpose().to_series().to_list()

    LOG.info("Tiling start %s", start)
    LOG.info("Tiling end %s", end)

    tiles_corner = _get_tiles_corner(start, end, tiling_scheme)

    no_overlap = all(o == 0 for o in tiling_scheme.overlap)

    for corner in tiles_corner:
        # corner considers the overlap, so right needs to be shifted by 2 * o
        # -1e-8 is because tracksdata filter slicing is inclusive
        slicing = tuple(slice(c, c + t - 1e-6) for c, t in zip(corner, tiling_scheme.tile_shape, strict=True))
        filtered_graph = spatial_filter[slicing]

        LOG.info("Tiling with corner %s", corner)
        LOG.info("Slicing %s", slicing)

        if no_overlap:
            filtered_graph_with_overlap = filtered_graph
        else:
            slicing_with_overlap = tuple(
                slice(s.start - o, s.stop + o) for s, o in zip(slicing, tiling_scheme.overlap, strict=True)
            )
            filtered_graph_with_overlap = spatial_filter[slicing_with_overlap]
            LOG.info("Slicing with overlap %s", slicing_with_overlap)

        yield func(filtered_graph_with_overlap, filtered_graph)


def apply_tiled(
    graph: BaseGraph,
    tiling_scheme: TilingScheme,
    func: MapFunc,
    *,
    agg_func: ReduceFunc | None,
) -> Iterator[T] | R:
    """
    Apply a function to a graph tiled by the tiling scheme.
    Graph will be sliced with 'tile_shape' + 2 * 'overlap' per axis.

    Parameters
    ----------
    graph : BaseGraph
        The graph to apply the function to.
    tiling_scheme : TilingScheme
        The tiling scheme to use.
    func : MapFunc
        The function to apply to each tile.
        It takes two arguments:
        - filtered_graph_with_overlap: the subgraph inside the tile with the overlap
        - filtered_graph: the subgraph inside the tile without the overlap
        If all overlaps are 0, filtered_graph_with_overlap == filtered_graph with minimal overhead.

    agg_func : ReduceFunc | None, optional
        The function to reduce the results of the function. If None, the results will be yielded.

    Returns
    -------
    Iterator[T] | R
        The results of the function. If agg_func is provided, the results will be reduced.
        Otherwise, the results will be yielded.
    """
    # this needs to be a separate function because python behave weirdly
    # with functions with both yield and return statements
    res_generator = _yield_apply_tiled(
        graph=graph,
        tiling_scheme=tiling_scheme,
        func=func,
    )

    # if agg_func is provided, we need to reduce the results
    if agg_func is not None:
        return agg_func(res_generator)

    return res_generator

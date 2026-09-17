"""ASV benchmarks for ``GraphArrayView``'s ``downscale`` parameter.

Rendering a graph at full resolution allocates one whole-volume buffer per
timepoint and paints every mask at full resolution. ``downscale`` samples the
masks on a strided view instead, so both the buffer and the paint cost shrink
by roughly the product of the factors.

``track_buffer_mbytes_*`` reports the cached buffer size, which is the metric
the parameter targets and is independent of machine speed. The ``time_*``
methods measure the cold-fetch wall time that follows from it.

The benchmarks skip themselves on revisions predating ``downscale`` so that
this file can live on ``main`` before the feature lands.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from benchmarks.common import IS_CI
from tracksdata.array import GraphArrayView
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.nodes._mask import Mask

if TYPE_CHECKING:
    from tracksdata.graph import RustWorkXGraph

FRAMES = 2 if IS_CI else 4
# Anisotropic on purpose: matching factors to physical voxel size is the
# intended usage, so ``(1, 4, 4)`` is the case worth tracking.
SHAPE = (FRAMES, 32, 256, 256)
MASK_SIZE = (16, 56, 56)
NODES_PER_FRAME = 20 if IS_CI else 40
COLD_FRAME = 0


def _downscale_supported() -> bool:
    """Whether the checked-out revision has the ``downscale`` parameter."""
    return "downscale" in inspect.signature(GraphArrayView.__init__).parameters


def _build_graph() -> RustWorkXGraph:
    # In-memory on purpose: with SQLGraph, per-node mask decompression happens at
    # full resolution whatever `downscale` is, which caps the ratio and obscures
    # the paint cost this benchmark exists to track.
    from tracksdata.graph import RustWorkXGraph

    graph = RustWorkXGraph()
    graph.add_node_attr_key("label", dtype=pl.Int64)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, pl.Object)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, pl.Array(pl.Int64, 6))

    rng = np.random.default_rng(0)
    label = 1
    for t in range(FRAMES):
        starts = np.stack(
            [rng.integers(0, s - m, size=NODES_PER_FRAME) for s, m in zip(SHAPE[1:], MASK_SIZE, strict=True)],
            axis=1,
        )
        for start in starts:
            bbox = np.concatenate([start, start + np.asarray(MASK_SIZE)])
            graph.add_node(
                {
                    DEFAULT_ATTR_KEYS.T: int(t),
                    "label": label,
                    DEFAULT_ATTR_KEYS.MASK: Mask(np.ones(MASK_SIZE, dtype=bool), bbox=bbox),
                    DEFAULT_ATTR_KEYS.BBOX: bbox,
                },
                validate_keys=False,
            )
            label += 1
    return graph


class DownscaleBenchmark:
    """Cold whole-frame fetches at several downscale factors."""

    # Each timed call builds a fresh (empty-cache) view, so one invocation per
    # sample is correct; batching them would measure the cache instead.
    number = 1
    timeout = 300

    params = [None, (1, 2, 2), (1, 4, 4), (2, 2, 2), 4]
    param_names = ["downscale"]

    def setup(self, downscale) -> None:
        if downscale is not None and not _downscale_supported():
            raise NotImplementedError("GraphArrayView has no `downscale` parameter on this revision")
        self.graph = _build_graph()

    def _make_view(self, downscale) -> GraphArrayView:
        kwargs = {} if downscale is None else {"downscale": downscale}
        return GraphArrayView(
            graph=self.graph,
            shape=SHAPE,
            attr_key="label",
            dtype=np.uint32,
            **kwargs,
        )

    def time_cold_whole_frame(self, downscale) -> None:
        np.asarray(self._make_view(downscale)[COLD_FRAME])

    def track_buffer_mbytes(self, downscale) -> float:
        """Cached buffer size for one timepoint, in MB."""
        view = self._make_view(downscale)
        np.asarray(view[COLD_FRAME])
        return view._cache._store[COLD_FRAME].buffer.nbytes / 1e6

    def track_labels_rendered(self, downscale) -> int:
        """Objects still visible, to catch a factor that silently drops them."""
        view = self._make_view(downscale)
        return int(len(np.unique(np.asarray(view[COLD_FRAME]))) - 1)


if __name__ == "__main__":
    import time

    print(f"shape={SHAPE}  masks/frame={NODES_PER_FRAME}  mask={MASK_SIZE}")
    print(f"downscale supported on this revision: {_downscale_supported()}\n")
    bench = DownscaleBenchmark()
    base_t = base_mb = None
    for downscale in DownscaleBenchmark.params:
        try:
            bench.setup(downscale)
        except NotImplementedError as e:
            print(f"  {str(downscale):>10s}  skipped ({e})")
            continue
        mbytes = bench.track_buffer_mbytes(downscale)
        labels = bench.track_labels_rendered(downscale)
        reps = 7
        start = time.perf_counter()
        for _ in range(reps):
            bench.time_cold_whole_frame(downscale)
        wall = (time.perf_counter() - start) * 1e3 / reps
        if base_t is None:
            base_t, base_mb = wall, mbytes
            print(f"  {str(downscale):>10s}  {wall:8.1f} ms  {mbytes:8.2f} MB   labels={labels}")
        else:
            print(
                f"  {str(downscale):>10s}  {wall:8.1f} ms  {mbytes:8.2f} MB   "
                f"labels={labels}  ({base_t / wall:.1f}x faster, {base_mb / mbytes:.0f}x smaller)"
            )

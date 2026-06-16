"""ASV benchmarks for NDChunkCache chunk coalescing in GraphArrayView.

A cold ``view[t]`` fetch touches every chunk overlapping the request. The
coalescing fix aggregates the not-ready chunks into a single bounding-box
slice and calls the compute function once, collapsing N per-chunk SQL queries
(and N mask decompressions) per cold frame into one.

The ``time_*`` methods measure wall time. The ``track_executes_*`` methods
report the number of SQL executes for the same access pattern -- this is the
metric the fix targets (1 execute per cold frame instead of one per chunk),
and it is independent of machine speed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import sqlalchemy as sa

from benchmarks.common import IS_CI
from tracksdata.array import GraphArrayView
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.nodes._mask import Mask

if TYPE_CHECKING:
    # Imported only for type hints: a module-level ``SQLGraph`` name would be
    # picked up by asv's discovery (it has a ``time_points`` method) and asv
    # would try to instantiate the bare class, which requires constructor args.
    from tracksdata.graph import SQLGraph

FRAME_SIZE = 512
CHUNK = 128
MASK_SIZE = 32
NODES_PER_FRAME = 50
# Frames only need to cover the distinct cold/cached scenarios below.
FRAMES = 8 if IS_CI else 16
SHAPE = (FRAMES, FRAME_SIZE, FRAME_SIZE)

# Frame indices used by the scenarios (a fresh view is built per cold call, so
# reusing the same index across repeats stays genuinely cold).
WARM_FRAME = 0
COLD_FRAME = 1


def _build_graph() -> SQLGraph:
    from tracksdata.graph import SQLGraph

    graph = SQLGraph(drivername="sqlite", database=":memory:", overwrite=True)
    graph.add_node_attr_key("label", dtype=pl.Int64)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, pl.Object)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, pl.Array(pl.Int64, 4))

    rng = np.random.default_rng(0)
    label = 1
    for t in range(FRAMES):
        # Spread nodes uniformly across the frame so most chunks are touched.
        positions = rng.integers(0, FRAME_SIZE - MASK_SIZE, size=(NODES_PER_FRAME, 2))
        for y, x in positions:
            mask_data = np.ones((MASK_SIZE, MASK_SIZE), dtype=bool)
            mask = Mask(mask_data, bbox=np.array([y, x, y + MASK_SIZE, x + MASK_SIZE]))
            graph.add_node(
                {
                    DEFAULT_ATTR_KEYS.T: int(t),
                    "label": label,
                    DEFAULT_ATTR_KEYS.MASK: mask,
                    DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
                },
                validate_keys=False,
            )
            label += 1
    return graph


class ChunkCoalescingBenchmark:
    """Cold/cached GraphArrayView frame fetches with a chunked NDChunkCache."""

    # Cold fetches must not be batched back-to-back: each timed call builds a
    # fresh (empty-cache) view, so a single invocation per sample is correct.
    number = 1
    timeout = 300

    def setup(self) -> None:
        self.graph = _build_graph()
        self._exec_count = 0

        @sa.event.listens_for(self.graph._engine, "before_cursor_execute")
        def _count(conn, cursor, statement, parameters, context, executemany) -> None:
            self._exec_count += 1

        # A pre-warmed view for the cached-path scenarios.
        self.warm_view = self._make_view()
        np.asarray(self.warm_view[WARM_FRAME])

    def _make_view(self) -> GraphArrayView:
        return GraphArrayView(
            graph=self.graph,
            shape=SHAPE,
            attr_key="label",
            chunk_shape=(CHUNK, CHUNK),
        )

    # --- wall time --------------------------------------------------------

    def time_cold_whole_frame(self) -> None:
        view = self._make_view()
        np.asarray(view[COLD_FRAME])

    def time_cached_whole_frame(self) -> None:
        np.asarray(self.warm_view[WARM_FRAME])

    def time_cold_single_chunk(self) -> None:
        view = self._make_view()
        np.asarray(view[COLD_FRAME, :CHUNK, :CHUNK])

    # --- SQL execute counts (the coalescing metric) -----------------------

    def track_executes_cold_whole_frame(self) -> int:
        view = self._make_view()
        self._exec_count = 0
        np.asarray(view[COLD_FRAME])
        return self._exec_count

    def track_executes_cached_whole_frame(self) -> int:
        self._exec_count = 0
        np.asarray(self.warm_view[WARM_FRAME])
        return self._exec_count

    def track_executes_cold_single_chunk(self) -> int:
        view = self._make_view()
        self._exec_count = 0
        np.asarray(view[COLD_FRAME, :CHUNK, :CHUNK])
        return self._exec_count


if __name__ == "__main__":
    # Standalone validation: prints wall time and SQL execute counts.
    import time

    bench = ChunkCoalescingBenchmark()
    bench.setup()
    chunks_per_frame = (FRAME_SIZE // CHUNK) ** 2
    print(f"chunks per frame = {chunks_per_frame} (expected 1 execute per cold frame after fix)\n")

    for scenario in ("cold_whole_frame", "cached_whole_frame", "cold_single_chunk"):
        executes = getattr(bench, f"track_executes_{scenario}")()
        start = time.perf_counter()
        getattr(bench, f"time_{scenario}")()
        wall_ms = (time.perf_counter() - start) * 1e3
        print(f"  {scenario:<20s}  wall_ms={wall_ms:9.2f}  executes={executes}")

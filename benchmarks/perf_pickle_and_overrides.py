"""Micro-benchmark for the pickle-decode + schema-override fast paths.

Stubs out optional native deps so the script runs in a minimal env.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types

# ---------------------------------------------------------------------------
# Stub optional native deps (ilpy, spatial_graph.PointRTree)
# ---------------------------------------------------------------------------


def _install_stubs() -> None:
    if "ilpy" not in sys.modules:
        ilpy = types.ModuleType("ilpy")

        class _Stub:
            def __init__(self, *a, **k):
                pass

        for name in (
            "Constraints",
            "Objective",
            "Preference",
            "Solution",
            "Solver",
            "SolverStatus",
            "Variable",
            "VariableType",
        ):
            setattr(ilpy, name, _Stub)
        sys.modules["ilpy"] = ilpy

    try:
        import spatial_graph  # noqa: F401
    except ImportError:
        sg = types.ModuleType("spatial_graph")

        class PointRTree:
            def __init__(self, *a, **k):
                self._items: dict[int, tuple] = {}

            def insert_point_items(self, ids, positions):
                for i, p in zip(list(ids), list(positions), strict=True):
                    self._items[int(i)] = tuple(p)

            def insert_bb_items(self, ids, lows, highs):
                for i, lo, hi in zip(list(ids), list(lows), list(highs), strict=True):
                    self._items[int(i)] = (tuple(lo), tuple(hi))

            def delete_items(self, ids):
                for i in ids:
                    self._items.pop(int(i), None)

            def search(self, lows, highs):
                return list(self._items.keys())

        sg.PointRTree = PointRTree
        sys.modules["spatial_graph"] = sg


_install_stubs()

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

# ---------------------------------------------------------------------------
# Instrumentation: count Series constructions of Object/Binary dtype.
# We patch pl.Series.__init__ so a single counter ticks each time a new
# Series is built with Object or Binary dtype (the "row-wise rebuild" path).
# ---------------------------------------------------------------------------

N_OBJECT_BINARY_SERIES = 0
_ORIG_SERIES_INIT = pl.Series.__init__


def _counting_series_init(self, *args, **kwargs):
    _ORIG_SERIES_INIT(self, *args, **kwargs)
    global N_OBJECT_BINARY_SERIES
    try:
        dt = self.dtype
    except Exception:
        return
    if dt in (pl.Object, pl.Binary):
        N_OBJECT_BINARY_SERIES += 1


pl.Series.__init__ = _counting_series_init  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Build a SQLGraph populated with masks.
# ---------------------------------------------------------------------------


def build_graph(db_path: str, n_frames: int, nodes_per_frame: int, mask_size: int):
    from tracksdata.constants import DEFAULT_ATTR_KEYS
    from tracksdata.graph import SQLGraph
    from tracksdata.nodes import Mask

    if os.path.exists(db_path):
        os.remove(db_path)

    graph = SQLGraph(
        drivername="sqlite",
        database=db_path,
        overwrite=True,
    )

    rng = np.random.default_rng(0)
    rand_mask = rng.random((mask_size, mask_size)) > 0.5

    # add the mask + a scalar attr key
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, dtype=pl.Object, default_value=None)
    graph.add_node_attr_key("score", dtype=pl.Float64, default_value=0.0)

    for t in range(n_frames):
        batch = []
        for i in range(nodes_per_frame):
            mask = Mask(mask=rand_mask.copy(), bbox=np.array([0, 0, mask_size, mask_size]))
            batch.append({"t": t, DEFAULT_ATTR_KEYS.MASK: mask, "score": float(i)})
        graph.bulk_add_nodes(batch)

    return graph


# ---------------------------------------------------------------------------
# Bench scenarios
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--nodes-per-frame", type=int, default=100)
    parser.add_argument("--mask-size", type=int, default=32)
    parser.add_argument("--ranges", type=int, default=100)
    parser.add_argument("--range-size", type=int, default=50)
    parser.add_argument("--db", type=str, default="/tmp/perf_pickle.db")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Re-introduce the pre-patch triple-pass + uncached override for an apples-to-apples comparison.",
    )
    args = parser.parse_args(argv)

    import cloudpickle
    import polars.selectors as cs

    from tracksdata.constants import DEFAULT_ATTR_KEYS
    from tracksdata.graph import SQLGraph
    from tracksdata.utils import _dataframe as _df_mod

    if args.baseline:
        # Restore the pre-patch triple-pass implementation: (1) map_elements ->
        # Object Series per binary col, (2) Series(to_list()) rebuild per Object
        # col, (3) a second typed rebuild from schema_overrides — what the old
        # SQLGraph._cast_array_columns did.
        def _baseline_unpickle(df, schema_overrides=None):
            schema_overrides = schema_overrides or {}
            df = df.map_columns(
                cs.binary(),
                lambda x: x.map_elements(cloudpickle.loads, return_dtype=pl.Object),
            )
            for col, dtype in zip(df.columns, df.dtypes, strict=True):
                if isinstance(dtype, pl.Object):
                    try:
                        df = df.with_columns(pl.Series(df[col].to_list()).alias(col))
                    except Exception:
                        pass
            casts = []
            for key, target in schema_overrides.items():
                if key not in df.columns:
                    continue
                try:
                    casts.append(pl.Series(key, df[key].to_list(), dtype=target))
                except Exception:
                    continue
            if casts:
                df = df.with_columns(casts)
            return df

        _df_mod.unpickle_bytes_columns = _baseline_unpickle
        import tracksdata.graph._sql_graph as _sgm

        _sgm.unpickle_bytes_columns = _baseline_unpickle

        # Force schema-override recompute on every call.
        _orig_so = SQLGraph._polars_schema_override

        def _uncached_so(self, table_class):
            self._node_polars_override_cache = None
            self._edge_polars_override_cache = None
            result = _orig_so(self, table_class)
            self._node_polars_override_cache = None
            self._edge_polars_override_cache = None
            return result

        SQLGraph._polars_schema_override = _uncached_so  # type: ignore[assignment]

    graph = build_graph(args.db, args.frames, args.nodes_per_frame, args.mask_size)

    # --- instrument _polars_schema_override hits/misses --------------------
    n_recomp = 0
    n_hits = 0

    _orig = SQLGraph._polars_schema_override

    def _counted(self, table_class):
        nonlocal n_recomp, n_hits
        is_node = table_class.__tablename__ == self.Node.__tablename__
        cached = self._node_polars_override_cache if is_node else self._edge_polars_override_cache
        if cached is None:
            n_recomp += 1
        else:
            n_hits += 1
        result = _orig(self, table_class)
        # Re-read cache after _orig in case the underlying impl was patched to be uncached.
        return result

    SQLGraph._polars_schema_override = _counted  # type: ignore[assignment]

    # warm up by reading once (this populates the cache by side effect)
    global N_OBJECT_BINARY_SERIES
    N_OBJECT_BINARY_SERIES = 0
    n_recomp = 0
    n_hits = 0

    all_node_ids = graph.node_ids()
    range_size = args.range_size
    starts = list(range(0, min(args.ranges * range_size, len(all_node_ids)), range_size))

    # Scenario A: many filtered node_attrs calls.
    t0 = time.perf_counter()
    for s in starts[: args.ranges]:
        ids = all_node_ids[s : s + range_size]
        df = graph.filter(node_ids=ids).node_attrs(attr_keys=["score", DEFAULT_ATTR_KEYS.MASK])
        assert df.height == len(ids)
    wall_filtered_ms = (time.perf_counter() - t0) * 1000.0

    # Scenario B: one whole-table call.
    t1 = time.perf_counter()
    df_all = graph.node_attrs(attr_keys=["score", DEFAULT_ATTR_KEYS.MASK])
    wall_whole_ms = (time.perf_counter() - t1) * 1000.0
    assert df_all.height == args.frames * args.nodes_per_frame

    print()
    print("=== perf_pickle_and_overrides ===")
    print(f"frames={args.frames}  nodes_per_frame={args.nodes_per_frame}  mask={args.mask_size}x{args.mask_size}")
    print(f"ranges={args.ranges}  range_size={args.range_size}")
    print()
    print(f"scenario A: {args.ranges} filtered node_attrs calls")
    print(f"  wall_ms                          : {wall_filtered_ms:9.1f}")
    print(f"  n_pl_series_object_binary        : {N_OBJECT_BINARY_SERIES}")
    print(f"  n_schema_override_recomputes     : {n_recomp}")
    print(f"  n_schema_override_hits           : {n_hits}")
    print()
    print("scenario B: 1 whole-table node_attrs call")
    print(f"  wall_ms                          : {wall_whole_ms:9.1f}")
    print()
    print("# baseline numbers from `--baseline` (pre-patch triple-pass + uncached override):")
    print("#   scenario A: schema_override_recomputes ~= ranges+1, pl.Series Object/Binary builds ~= 2N+1")
    print("#                                                       (excludes the FFI map_elements row pass that")
    print("#                                                       does not go through pl.Series.__init__)")
    print("# after patch:")
    print("#   scenario A: schema_override_recomputes == 1 (warm-up only) and Series builds ~= N+1")


if __name__ == "__main__":
    main()

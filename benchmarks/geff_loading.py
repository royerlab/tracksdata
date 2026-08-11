"""ASV benchmarks for loading and querying GEFF graph backends.

The suite separates construction from steady-state read queries because
``ZarrSQLGraph`` intentionally defers array reads until a query is collected,
whereas the other backends materialize the GEFF during ``from_geff``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from geff.core_io import write_arrays
from geff_spec import Axis, GeffMetadata, PropMetadata

import tracksdata as td
from benchmarks.common import IS_CI
from tracksdata.attrs import EdgeAttr, NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS

if IS_CI:
    NODE_SIZES = (10_000,)
else:
    NODE_SIZES = (1_000, 100_000)

BACKEND_NAMES = (
    "RustWorkXGraph",
    "IndexedRXGraph",
    "SQLGraphMemory",
    "ZarrSQLGraph",
)
N_TIME_POINTS = 100
N_SUCCESSOR_SEEDS = 256


def _write_geff(path: Path, n_nodes: int) -> None:
    """Write a deterministic scalar-property GEFF benchmark fixture."""
    node_ids = np.arange(n_nodes, dtype=np.uint64)
    node_props = {
        DEFAULT_ATTR_KEYS.T: {
            "values": np.arange(n_nodes, dtype=np.int32) % N_TIME_POINTS,
            "missing": None,
        },
        "score": {
            "values": np.linspace(0.0, 1.0, n_nodes, dtype=np.float32),
            "missing": None,
        },
        "label": {
            "values": np.arange(n_nodes, dtype=np.int32),
            "missing": None,
        },
    }

    edge_source = np.arange(max(0, n_nodes - 1), dtype=np.uint64)
    edge_target = edge_source + 1
    edge_ids = np.column_stack((edge_source, edge_target))
    edge_props = {
        "weight": {
            "values": np.linspace(0.0, 1.0, len(edge_ids), dtype=np.float32),
            "missing": None,
        }
    }
    metadata = GeffMetadata(
        directed=True,
        axes=[Axis(name=DEFAULT_ATTR_KEYS.T, type="time")],
        node_props_metadata={
            DEFAULT_ATTR_KEYS.T: PropMetadata(identifier=DEFAULT_ATTR_KEYS.T, dtype="int32"),
            "score": PropMetadata(identifier="score", dtype="float32"),
            "label": PropMetadata(identifier="label", dtype="int32"),
        },
        edge_props_metadata={"weight": PropMetadata(identifier="weight", dtype="float32")},
        extra={"tracksdata": {"benchmark_nodes": n_nodes}},
    )
    write_arrays(
        path,
        node_ids=node_ids,
        node_props=node_props,
        edge_ids=edge_ids,
        edge_props=edge_props,
        metadata=metadata,
        overwrite=True,
        zarr_format=3,
    )


def _load_graph(backend_name: str, path: str) -> td.graph.BaseGraph:
    """Load one backend from the common GEFF fixture."""
    if backend_name == "SQLGraphMemory":
        graph, _ = td.graph.SQLGraph.from_geff(
            path,
            drivername="sqlite",
            database=":memory:",
            engine_kwargs={"connect_args": {"check_same_thread": False}},
        )
        return graph

    backend = getattr(td.graph, backend_name)
    graph, _ = backend.from_geff(path)
    return graph


def _dispose_graph(graph: td.graph.BaseGraph) -> None:
    """Release SQLAlchemy resources without assuming every backend has an engine."""
    if type(graph) is td.graph.SQLGraph:
        graph._engine.dispose()


class _GeffFixture:
    """Shared ASV cache containing identical GEFF inputs for every backend."""

    param_names = ("backend", "n_nodes")
    params = (BACKEND_NAMES, NODE_SIZES)
    timeout = 300

    def setup_cache(self) -> dict[int, str]:
        root = Path(tempfile.mkdtemp(prefix="tracksdata_geff_benchmark_"))
        paths: dict[int, str] = {}
        for n_nodes in NODE_SIZES:
            path = root / f"graph_{n_nodes}.geff"
            _write_geff(path, n_nodes)
            paths[n_nodes] = str(path)
        return paths


class GeffLoadBenchmark(_GeffFixture):
    """Fresh-object ``from_geff`` construction cost for each backend."""

    number = 1
    warmup_time = 0

    def setup(self, paths: dict[int, str], backend_name: str, n_nodes: int) -> None:
        self.path = paths[n_nodes]
        self.graph: td.graph.BaseGraph | None = None

    def teardown(self, paths: dict[int, str], backend_name: str, n_nodes: int) -> None:
        if self.graph is not None:
            _dispose_graph(self.graph)

    def time_from_geff(self, paths: dict[int, str], backend_name: str, n_nodes: int) -> None:
        self.graph = _load_graph(backend_name, self.path)


class GeffQueryBenchmark(_GeffFixture):
    """Read-query costs after backend construction has completed."""

    def setup(self, paths: dict[int, str], backend_name: str, n_nodes: int) -> None:
        self.graph = _load_graph(backend_name, paths[n_nodes])
        self.filter_time = N_TIME_POINTS // 2
        seed_count = min(N_SUCCESSOR_SEEDS, max(1, n_nodes - 1))
        self.successor_seeds = np.linspace(0, max(0, n_nodes - 2), seed_count, dtype=np.int64).tolist()

    def teardown(self, paths: dict[int, str], backend_name: str, n_nodes: int) -> None:
        _dispose_graph(self.graph)

    def time_node_attrs(self, paths: dict[int, str], backend_name: str, n_nodes: int) -> None:
        self.graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.T, "score", "label"])

    def time_edge_attrs(self, paths: dict[int, str], backend_name: str, n_nodes: int) -> None:
        self.graph.edge_attrs(attr_keys=["weight"])

    def time_filter_projected_node_attrs(
        self,
        paths: dict[int, str],
        backend_name: str,
        n_nodes: int,
    ) -> None:
        self.graph.filter(NodeAttr(DEFAULT_ATTR_KEYS.T) == self.filter_time).node_attrs(
            attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, "score"]
        )

    def time_filter_edge_ids(self, paths: dict[int, str], backend_name: str, n_nodes: int) -> None:
        self.graph.filter(EdgeAttr("weight") >= 0.5).edge_ids()

    def time_successors_batch(self, paths: dict[int, str], backend_name: str, n_nodes: int) -> None:
        self.graph.successors(self.successor_seeds)

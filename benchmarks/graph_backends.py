from __future__ import annotations

from collections.abc import Callable

from tracksdata.attrs import EdgeAttr, NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges import DistanceEdges
from tracksdata.graph import IndexedRXGraph, RustWorkXGraph
from tracksdata.graph import SQLGraph as _BaseSQLGraph
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes import RandomNodes
from tracksdata.options import set_options
from tracksdata.solvers import NearestNeighborsSolver

N_TIME_POINTS = 50
NODE_SIZES = (1_000, 10_000, 100_000)
WORKER_COUNTS = (1, 4)


class SQLGraphWithMemory(_BaseSQLGraph):
    def __init__(self) -> None:
        super().__init__(drivername="sqlite", database=":memory:", overwrite=True)


class SQLGraphDisk(_BaseSQLGraph):
    def __init__(self) -> None:
        import datetime

        path = f"/tmp/_benchmarks_tracksdata_db_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        super().__init__(drivername="sqlite", database=path, overwrite=True)


BACKENDS: dict[str, type[BaseGraph]] = {
    "RustWorkXGraph": RustWorkXGraph,
    "IndexedRXGraph": IndexedRXGraph,
    "SQLGraphWithMemory": SQLGraphWithMemory,
    "SQLGraphDisk": SQLGraphDisk,
}


def _build_pipeline(
    n_time_points: int, n_nodes_per_tp: int
) -> list[tuple[str, Callable[[BaseGraph], None | BaseGraph]]]:
    lower = max(1, int(n_nodes_per_tp * 0.95))
    upper = max(lower, int(n_nodes_per_tp * 1.05))
    return [
        (
            "random_nodes",
            RandomNodes(
                n_time_points=n_time_points,
                n_nodes_per_tp=(lower, upper),
                n_dim=3,
            ).add_nodes,
        ),
        ("distance_edges", DistanceEdges(distance_threshold=10, n_neighbors=5).add_edges),
        (
            "nearest_neighbors_solver",
            NearestNeighborsSolver(
                edge_weight=-EdgeAttr(DEFAULT_ATTR_KEYS.EDGE_DIST),
                max_children=2,
                return_solution=False,
            ).solve,
        ),
        (
            "subgraph",
            lambda graph: graph.filter(
                NodeAttr(DEFAULT_ATTR_KEYS.SOLUTION) == True,
                EdgeAttr(DEFAULT_ATTR_KEYS.SOLUTION) == True,
            ).subgraph(),
        ),
        ("assign_tracks", lambda graph: graph.assign_tracklet_ids()),
    ]


class GraphBackendsBenchmark:
    """
    ASV benchmark suite that times each step of the graph-building pipeline per backend.
    """

    param_names = ("backend", "n_nodes", "n_workers")
    params = (tuple(BACKENDS), NODE_SIZES, WORKER_COUNTS)

    def setup(self, backend_name: str, n_nodes: int, n_workers: int) -> None:
        if backend_name == "SQLGraphWithMemory" and n_workers > 1:
            msg = "SQLGraphWithMemory does not support multiprocessing."
            raise NotImplementedError(msg)

        self.backend_name = backend_name
        self.n_nodes = n_nodes
        self.n_workers = n_workers

        set_options(show_progress=False, n_workers=n_workers)
        self.backend_cls = BACKENDS[backend_name]
        n_nodes_per_tp = max(1, n_nodes // N_TIME_POINTS)
        self.pipeline = _build_pipeline(N_TIME_POINTS, n_nodes_per_tp)

    def _fresh_graph(self) -> BaseGraph:
        return self.backend_cls()

    def _prepare_graph_for(self, step_name: str) -> tuple[BaseGraph, Callable[[BaseGraph], None | BaseGraph]]:
        graph = self._fresh_graph()
        for name, func in self.pipeline:
            if name == step_name:
                return graph, func
            result = func(graph)
            if result is not None:
                graph = result
        msg = f"Unknown pipeline step '{step_name}'."
        raise ValueError(msg)

    def time_graph_init(self, backend_name: str, n_nodes: int, n_workers: int) -> None:
        self._fresh_graph()

    def time_random_nodes(self, backend_name: str, n_nodes: int, n_workers: int) -> None:
        graph, func = self._prepare_graph_for("random_nodes")
        func(graph)

    def time_distance_edges(self, backend_name: str, n_nodes: int, n_workers: int) -> None:
        graph, func = self._prepare_graph_for("distance_edges")
        func(graph)

    def time_nearest_neighbors_solver(self, backend_name: str, n_nodes: int, n_workers: int) -> None:
        graph, func = self._prepare_graph_for("nearest_neighbors_solver")
        func(graph)

    def time_subgraph(self, backend_name: str, n_nodes: int, n_workers: int) -> None:
        graph, func = self._prepare_graph_for("subgraph")
        func(graph)

    def time_assign_tracks(self, backend_name: str, n_nodes: int, n_workers: int) -> None:
        graph, func = self._prepare_graph_for("assign_tracks")
        func(graph)

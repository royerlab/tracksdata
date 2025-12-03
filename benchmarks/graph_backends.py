from __future__ import annotations

import warnings

from asv_runner.benchmarks.mark import SkipNotImplemented

import tracksdata as td  # Graph classes are not imported globally to avoid running "time_point" function at benchmark
from tracksdata.attrs import EdgeAttr, NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges import DistanceEdges
from tracksdata.nodes import RandomNodes
from tracksdata.options import set_options
from tracksdata.solvers import NearestNeighborsSolver
from tracksdata.utils._logging import LOG

warnings.filterwarnings("ignore")
LOG.setLevel("ERROR")

N_TIME_POINTS = 50
NODE_SIZES = (1_000, 100_000)
WORKER_COUNTS = (1, 4)

# With subclassing, the asv calls "time_point" function as a benchmark.
# So we define a function to return objects.


class SQLGraphWithMemory(td.graph.SQLGraph):
    def __init__(self):
        super().__init__(drivername="sqlite", database=":memory:", overwrite=True)

    def time_points(self):
        raise SkipNotImplemented("This is not a benchmark.")


class SQLGraphDisk(td.graph.SQLGraph):
    def __init__(self):
        import datetime

        path = f"/tmp/_benchmarks_tracksdata_db_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}+{id(self)}.db"
        super().__init__(drivername="sqlite", database=path, overwrite=True)

    def time_points(self):
        raise SkipNotImplemented("This is not a benchmark.")


BACKENDS = {
    "RustWorkXGraph": td.graph.RustWorkXGraph,
    "IndexedRXGraph": td.graph.IndexedRXGraph,
    "SQLGraphWithMemory": SQLGraphWithMemory,
    "SQLGraphDisk": SQLGraphDisk,
}


def _get_subgraph(graph):
    return graph.filter(
        NodeAttr(DEFAULT_ATTR_KEYS.SOLUTION) == True,
        EdgeAttr(DEFAULT_ATTR_KEYS.SOLUTION) == True,
    ).subgraph()


def _assign_tracks(graph):
    graph.assign_tracklet_ids()
    return None


def _build_pipeline(n_time_points: int, n_nodes_per_tp: int) -> dict:
    lower = max(1, int(n_nodes_per_tp * 0.95))
    upper = max(lower, int(n_nodes_per_tp * 1.05))
    return {
        "random_nodes": RandomNodes(
            n_time_points=n_time_points,
            n_nodes_per_tp=(lower, upper),
            n_dim=3,
        ).add_nodes,
        "distance_edges": DistanceEdges(distance_threshold=10, n_neighbors=5).add_edges,
        "nearest_neighbors_solver": NearestNeighborsSolver(
            edge_weight=-EdgeAttr(DEFAULT_ATTR_KEYS.EDGE_DIST),
            max_children=2,
            return_solution=False,
        ).solve,
        "subgraph": _get_subgraph,
        "assign_tracks": _assign_tracks,
    }


def _fresh_graph(backend_name) -> td.graph.BaseGraph:
    return BACKENDS[backend_name]()


class GraphBackendsBenchmark:
    """
    ASV benchmark suite that times each step of the graph-building pipeline per backend.
    """

    param_names = ("backend", "n_nodes", "n_workers")
    params = (tuple(BACKENDS), NODE_SIZES, WORKER_COUNTS)

    def setup_cache(self):
        pipelines = {}
        graphs = {}
        for n_nodes in NODE_SIZES:
            n_nodes_per_tp = max(1, n_nodes // N_TIME_POINTS)
            pipeline = _build_pipeline(N_TIME_POINTS, n_nodes_per_tp)
            pipelines[n_nodes] = pipeline
            for target_name in pipeline.keys():
                graph = _fresh_graph("RustWorkXGraph")
                for name, func in pipeline.items():
                    if name == target_name:
                        graphs[(n_nodes, name)] = graph
                        break
                    res = func(graph)
                    if res is not None:
                        graph = res
        return {"pipelines": pipelines, "graphs": graphs}

    def setup(self, cache: dict, backend_name: str, n_nodes: int, n_workers: int) -> None:
        if n_workers > 1 and "SQLGraph" in backend_name:
            raise SkipNotImplemented("SQLGraph does not support multiprocessing with multiple workers.")
        pipelines = cache["pipelines"]
        graphs = cache["graphs"]
        self.input_graphs = {}
        set_options(n_workers=n_workers, show_progress=False)
        for name in pipelines[n_nodes]:
            self.input_graphs[name] = BACKENDS[backend_name].from_other(graphs[(n_nodes, name)])

    def time_graph_init(self, cache: dict, backend_name: str, n_nodes: int, n_workers: int) -> None:
        _fresh_graph(backend_name)

    def time_random_nodes(self, cache: dict, backend_name: str, n_nodes: int, n_workers: int) -> None:
        pipelines = cache["pipelines"]
        input_graph = self.input_graphs["random_nodes"]
        pipelines[n_nodes]["random_nodes"](input_graph)

    def time_distance_edges(self, cache: dict, backend_name: str, n_nodes: int, n_workers: int) -> None:
        pipelines = cache["pipelines"]
        input_graph = self.input_graphs["distance_edges"]
        pipelines[n_nodes]["distance_edges"](input_graph)

    def time_nearest_neighbors_solver(self, cache: dict, backend_name: str, n_nodes: int, n_workers: int) -> None:
        pipelines = cache["pipelines"]
        input_graph = self.input_graphs["nearest_neighbors_solver"]
        pipelines[n_nodes]["nearest_neighbors_solver"](input_graph)

    def time_subgraph(self, cache: dict, backend_name: str, n_nodes: int, n_workers: int) -> None:
        pipelines = cache["pipelines"]
        input_graph = self.input_graphs["subgraph"]
        pipelines[n_nodes]["subgraph"](input_graph)

    def time_assign_tracks(self, cache: dict, backend_name: str, n_nodes: int, n_workers: int) -> None:
        pipelines = cache["pipelines"]
        input_graph = self.input_graphs["assign_tracks"]
        pipelines[n_nodes]["assign_tracks"](input_graph)

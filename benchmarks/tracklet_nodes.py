from itertools import pairwise

import tracksdata as td

if __name__ == "__main__":
    from common import BACKENDS, IS_CI  # For local testing
else:
    from benchmarks.common import BACKENDS, IS_CI

if not IS_CI:
    ALL_LINAGE_SIZES = (
        1,
        100,
    )
    NODE_SIZES = (
        10,
        100,
        1_000,
    )
else:
    ALL_LINAGE_SIZES = (100,)
    NODE_SIZES = (1_000,)


class TrackletNodesBenchmark:
    param_names = ("backend", "n_nodes", "n_lineages")
    params = (tuple(BACKENDS), NODE_SIZES, ALL_LINAGE_SIZES)

    def setup(self, backend_name: str, n_nodes: int, n_lineages: int) -> None:
        graph = BACKENDS[backend_name]()
        for i in range(n_lineages):
            node_ids = graph.bulk_add_nodes([{td.DEFAULT_ATTR_KEYS.T: i} for i in range(n_nodes)])
            graph.bulk_add_edges(
                [
                    {td.DEFAULT_ATTR_KEYS.EDGE_SOURCE: n1, td.DEFAULT_ATTR_KEYS.EDGE_TARGET: n2}
                    for n1, n2 in pairwise(node_ids)
                ]
            )
            if i == 0:
                self.node_ids = node_ids

        self.graph = graph

    def time_tracklet_nodes(self, backend_name: str, n_nodes: int, n_lineages: int) -> None:
        return self.graph.tracklet_nodes([self.node_ids[len(self.node_ids) // 2]])


if __name__ == "__main__":
    import cProfile

    tnb = TrackletNodesBenchmark()
    tnb.setup("SQLGraphDisk", 1000, 100)
    with cProfile.Profile() as pr:
        tnb.time_tracklet_nodes("SQLGraphDisk", 1000, 100)
    pr.dump_stats("result.pstat")

from itertools import pairwise

import tracksdata as td
from tracksdata.attrs import NodeAttr

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


class TrackletSubgraphBenchmark:
    """`assign_tracklet_ids` and materializing a single tracklet as a subgraph.

    Each lineage is an unbranched chain spanning every frame, so it maps to
    exactly one tracklet. Selecting one track in a viewer filters on
    `tracklet_id` -- unlike `t`, that column is neither indexed nor encoded in
    the node ids, so the filter is a full scan of the node table and grows with
    the whole graph rather than with the track.

    `assign_tracklet_ids` is the counterpart worst case: the SQL backend
    implements it by materializing the *entire* graph as a `GraphView`, so it
    is bounded by subgraph construction rather than by the tracklet logic.
    """

    param_names = ("backend", "n_frames", "n_lineages")
    params = (tuple(BACKENDS), (200,) if IS_CI else (200, 1_000), (100,))

    timeout = 600

    def setup(self, backend_name: str, n_frames: int, n_lineages: int) -> None:
        self.graph = BACKENDS[backend_name]()
        prev_ids: list[int] = []
        for t in range(n_frames):
            ids = self.graph.bulk_add_nodes([{td.DEFAULT_ATTR_KEYS.T: t} for _ in range(n_lineages)])
            if prev_ids:
                self.graph.bulk_add_edges(
                    [
                        {td.DEFAULT_ATTR_KEYS.EDGE_SOURCE: s, td.DEFAULT_ATTR_KEYS.EDGE_TARGET: d}
                        for s, d in zip(prev_ids, ids, strict=True)
                    ]
                )
            prev_ids = ids

        self.graph.assign_tracklet_ids()
        tracklet_key = td.DEFAULT_ATTR_KEYS.TRACKLET_ID
        tracklet_ids = self.graph.node_attrs(attr_keys=[tracklet_key])[tracklet_key].unique().to_list()
        self.target_tracklet = tracklet_ids[len(tracklet_ids) // 2]

    def time_assign_tracklet_ids(self, backend_name: str, n_frames: int, n_lineages: int) -> None:
        self.graph.assign_tracklet_ids()

    def time_filter_tracklet_node_ids(self, backend_name: str, n_frames: int, n_lineages: int) -> None:
        self.graph.filter(NodeAttr(td.DEFAULT_ATTR_KEYS.TRACKLET_ID) == self.target_tracklet).node_ids()

    def time_subgraph_one_tracklet(self, backend_name: str, n_frames: int, n_lineages: int) -> None:
        self.graph.filter(NodeAttr(td.DEFAULT_ATTR_KEYS.TRACKLET_ID) == self.target_tracklet).subgraph()

    def time_subgraph_one_tracklet_by_ids(self, backend_name: str, n_frames: int, n_lineages: int) -> None:
        # Two-step form: resolve the ids first, then materialize by id. Isolates
        # subgraph construction from the unindexed tracklet_id scan.
        node_ids = self.graph.filter(NodeAttr(td.DEFAULT_ATTR_KEYS.TRACKLET_ID) == self.target_tracklet).node_ids()
        self.graph.filter(node_ids=node_ids).subgraph()


if __name__ == "__main__":
    import cProfile

    tnb = TrackletNodesBenchmark()
    tnb.setup("SQLGraphDisk", 1000, 100)
    with cProfile.Profile() as pr:
        tnb.time_tracklet_nodes("SQLGraphDisk", 1000, 100)
    pr.dump_stats("result.pstat")

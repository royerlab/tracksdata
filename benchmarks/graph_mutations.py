"""Benchmarks for graph mutations and queries beyond the pipeline benchmarks.

These cover hot paths (`remove_node`, `update_node_attrs`, `filter`) that
the workflow benchmarks don't exercise, including the GraphView variants
that downstream interactive-editing use cases hit.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import polars as pl

import tracksdata as td
from benchmarks.common import BACKENDS, IS_CI
from tracksdata.attrs import NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph.filters import BBoxSpatialFilter

# Total node count. Tuned so the current (pre-fix) `remove_node` finishes
# within the per-benchmark timeout — see PR discussion for sizing rationale.
if not IS_CI:
    NODE_SIZES = (1_000, 5_000)
else:
    NODE_SIZES = (1_000,)

# Operations per benchmark invocation.
N_OPS = 50

# Number of independent lineages (chains across time).
N_LINEAGES = 50


def _noop(*_args) -> None:
    """No-op slot to enable the `node_updated` signal-payload path in benchmarks."""


def _build_graph(backend_name: str, n_nodes: int) -> td.graph.BaseGraph:
    graph = BACKENDS[backend_name]()
    graph.add_node_attr_key("score", dtype=pl.Float64)
    nodes_per_lineage = max(2, n_nodes // N_LINEAGES)
    for _ in range(N_LINEAGES):
        node_ids = graph.bulk_add_nodes([{DEFAULT_ATTR_KEYS.T: t, "score": 0.0} for t in range(nodes_per_lineage)])
        graph.bulk_add_edges(
            [{DEFAULT_ATTR_KEYS.EDGE_SOURCE: a, DEFAULT_ATTR_KEYS.EDGE_TARGET: b} for a, b in pairwise(node_ids)]
        )
    return graph


class GraphMutationsBenchmark:
    """Mutations and standalone queries not covered by the pipeline benchmarks."""

    param_names = ("backend", "n_nodes")
    params = (tuple(BACKENDS), NODE_SIZES)

    # The mutation benchmarks are destructive: a timed call removes nodes from
    # the graph built in `setup`, so it can only run once per fresh graph. asv
    # re-runs `setup` before each *sample* (via the timeit setup), but only if
    # we keep it out of the warmup loop, which repeats the function without
    # re-setup. So pin `number = 1` (one call per sample) and `warmup_time = 0`
    # (skip warmup entirely) — otherwise the second warmup call hits an
    # already-removed node and raises `Node N does not exist in the graph.`
    number = 1
    warmup_time = 0
    timeout = 300

    def setup(self, backend_name: str, n_nodes: int) -> None:
        self.graph = _build_graph(backend_name, n_nodes)
        self.view = self.graph.filter().subgraph()
        all_ids = self.graph.node_ids()
        self.removal_targets = all_ids[:N_OPS]
        self.update_targets = all_ids[: N_OPS * 4]

        # Separate view with a no-op listener attached. Without a listener,
        # update_node_attrs skips the signal-payload computation entirely, so
        # the P2-2 optimization (deriving new_attrs from old + applied) isn't
        # exercised. This view is the BBoxSpatialFilter / GraphArrayView use case.
        self.listened_view = self.graph.filter().subgraph()
        self.listened_view.node_updated.connect(_noop)
        # Smaller batch, representative of interactive editing where the saved
        # query overhead is a larger fraction of the total work.
        self.listener_update_targets = all_ids[:N_OPS]

    # --- remove_node ------------------------------------------------------

    def time_remove_node_root(self, backend_name: str, n_nodes: int) -> None:
        for nid in self.removal_targets:
            self.graph.remove_node(nid)

    def time_remove_node_view(self, backend_name: str, n_nodes: int) -> None:
        for nid in self.removal_targets:
            self.view.remove_node(nid)

    # --- update_node_attrs (bulk) ----------------------------------------

    def time_update_node_attrs_root(self, backend_name: str, n_nodes: int) -> None:
        self.graph.update_node_attrs(node_ids=self.update_targets, attrs={"score": 1.0})

    def time_update_node_attrs_view(self, backend_name: str, n_nodes: int) -> None:
        self.view.update_node_attrs(node_ids=self.update_targets, attrs={"score": 1.0})

    def time_update_node_attrs_view_with_listener(self, backend_name: str, n_nodes: int) -> None:
        self.listened_view.update_node_attrs(node_ids=self.listener_update_targets, attrs={"score": 1.0})

    # --- filter (standalone, materialized to ids) ------------------------

    def time_filter_node_ids(self, backend_name: str, n_nodes: int) -> None:
        self.graph.filter(NodeAttr(DEFAULT_ATTR_KEYS.T) >= 1).node_ids()


def _build_bbox_graph(backend_name: str, n_nodes: int) -> td.graph.BaseGraph:
    """Graph whose nodes carry a bbox, so a real BBoxSpatialFilter can index them."""
    graph = BACKENDS[backend_name]()
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, dtype=pl.Array(pl.Int64, 4))
    graph.add_node_attr_key("score", dtype=pl.Float64)
    nodes_per_lineage = max(2, n_nodes // N_LINEAGES)
    for lineage in range(N_LINEAGES):
        graph.bulk_add_nodes(
            [
                {
                    DEFAULT_ATTR_KEYS.T: t,
                    DEFAULT_ATTR_KEYS.BBOX: np.asarray([lineage, t, lineage + 2, t + 2]),
                    "score": 0.0,
                }
                for t in range(nodes_per_lineage)
            ]
        )
    return graph


class SpatialFilterUpdateBenchmark:
    """`update_node_attrs` with a live BBoxSpatialFilter attached to `node_updated`.

    This is the `assign_tracklet_ids` regression case: a non-spatial bulk write
    (e.g. ``tracklet_id``) emits ``node_updated``, and before the fix the filter
    re-indexes every node in the rtree even though no bbox/frame changed. The
    `_noop`-listener benchmark above only times the producer-side payload build;
    this one times the consumer-side rtree work that actually regressed.
    """

    param_names = ("backend", "n_nodes")
    params = (tuple(BACKENDS), NODE_SIZES)

    number = 1
    warmup_time = 0
    timeout = 300

    def setup(self, backend_name: str, n_nodes: int) -> None:
        self.graph = _build_bbox_graph(backend_name, n_nodes)
        # Attach a real spatial filter so node_updated drives rtree mutations.
        self.spatial_filter = BBoxSpatialFilter(
            self.graph, frame_attr_key=DEFAULT_ATTR_KEYS.T, bbox_attr_key=DEFAULT_ATTR_KEYS.BBOX
        )
        self.target_ids = self.graph.node_ids()

    def time_update_non_spatial_attr_with_filter(self, backend_name: str, n_nodes: int) -> None:
        # Non-spatial write: bbox/frame untouched -> should be O(1) for the filter
        # after the fix, O(N) rtree churn before it.
        self.graph.update_node_attrs(node_ids=self.target_ids, attrs={"score": 1.0})

    def time_update_bbox_attr_with_filter(self, backend_name: str, n_nodes: int) -> None:
        # Spatial write: bbox genuinely changes -> filter must re-index. Guards
        # against the fix over-eagerly skipping legitimate updates.
        new_bboxes = [np.asarray([i, i, i + 3, i + 3]) for i in range(len(self.target_ids))]
        self.graph.update_node_attrs(node_ids=self.target_ids, attrs={DEFAULT_ATTR_KEYS.BBOX: new_bboxes})

"""Benchmarks for graph mutations and queries beyond the pipeline benchmarks.

These cover hot paths (`remove_node`, `update_node_attrs`, `filter`) that
the workflow benchmarks don't exercise, including the GraphView variants
that downstream interactive-editing use cases hit.
"""

from __future__ import annotations

from itertools import pairwise

import polars as pl

import tracksdata as td
from benchmarks.common import BACKENDS, IS_CI
from tracksdata.attrs import NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS

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

    # ASV's default `number=10` would invoke each method 10x back-to-back,
    # which breaks stateful mutations — force a single invocation per rep.
    number = 1
    timeout = 300

    def setup(self, backend_name: str, n_nodes: int) -> None:
        self.graph = _build_graph(backend_name, n_nodes)
        self.view = self.graph.filter().subgraph()
        all_ids = self.graph.node_ids()
        self.removal_targets = all_ids[:N_OPS]
        self.update_targets = all_ids[: N_OPS * 4]

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

    # --- filter (standalone, materialized to ids) ------------------------

    def time_filter_node_ids(self, backend_name: str, n_nodes: int) -> None:
        self.graph.filter(NodeAttr(DEFAULT_ATTR_KEYS.T) >= 1).node_ids()

from collections.abc import Sequence
from typing import TYPE_CHECKING

import polars as pl

from tracksdata.attrs import AttrComparison
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._rustworkx_graph import (
    IndexedRXGraph,
    RXFilter,
    _create_filter_func,
)
from tracksdata.graph.filters._base_filter import cache_method

if TYPE_CHECKING:
    from tracksdata.graph._graph_view import GraphView


class IndexRXFilter(RXFilter):
    _graph: "GraphView | IndexedRXGraph"

    def __init__(
        self,
        *attr_comps: AttrComparison,
        graph: "GraphView | IndexedRXGraph",
        node_ids: Sequence[int] | None = None,
        include_targets: bool = False,
        include_sources: bool = False,
    ) -> None:
        super().__init__(
            *attr_comps,
            graph=graph,
            node_ids=node_ids,
            include_targets=include_targets,
            include_sources=include_sources,
        )

    @cache_method
    def edge_attrs(self, attr_keys: list[str] | None = None, unpack: bool = False) -> pl.DataFrame:
        df = super().edge_attrs(attr_keys, unpack)
        return self._graph._map_df_to_external(
            df, [DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]
        )

    @cache_method
    def node_ids(self) -> list[int]:
        indices = super().node_ids()
        return self._graph._map_to_external(indices)

    @cache_method
    def subgraph(
        self,
        node_attr_keys: Sequence[str] | str | None = None,
        edge_attr_keys: Sequence[str] | str | None = None,
    ) -> "GraphView":
        from tracksdata.graph._graph_view import GraphView

        node_ids = self.node_ids()

        rx_graph, node_map = self._graph._rx_subgraph_with_nodemap(node_ids)
        if self._edge_attr_comps:
            _filter_func = _create_filter_func(self._edge_attr_comps)
            for src, tgt, attr in rx_graph.weighted_edge_list():
                if not _filter_func(attr):
                    rx_graph.remove_edge(src, tgt)

        root = self._graph
        if hasattr(self._graph, "_root"):
            root = self._graph._root

        # Ensure the time key is in the node attributes
        if node_attr_keys is not None:
            node_attr_keys = [DEFAULT_ATTR_KEYS.T, *node_attr_keys]
            node_attr_keys = list(dict.fromkeys(node_attr_keys))

        graph_view = GraphView(
            rx_graph,
            node_map_to_root=dict(node_map.items()),
            root=root,
            node_attr_keys=node_attr_keys,
            edge_attr_keys=edge_attr_keys,
        )

        return graph_view

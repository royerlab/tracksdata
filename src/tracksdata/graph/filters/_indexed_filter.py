from collections.abc import Sequence
from typing import TYPE_CHECKING

import polars as pl

from tracksdata.attrs import Filter
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._rustworkx_graph import (
    IndexedRXGraph,
    RXFilter,
    _create_filter_func,
)
from tracksdata.utils._cache import cache_method

if TYPE_CHECKING:
    from tracksdata.graph._graph_view import GraphView, ViewMode


def _clamp_to_parent(
    requested: Sequence[str] | str | None,
    parent_keys: list[str] | None,
) -> Sequence[str] | str | None:
    """
    Limit the keys a child view claims to hold to the ones its parent held.

    ``parent_keys is None`` means the parent held everything its own root has,
    so there is nothing to clamp. Otherwise requested keys the parent did not
    hold are dropped rather than claimed: the child's rx payloads are copied
    from the parent, so a key the parent lacked has no values here either.
    Claiming it makes reads return schema defaults instead of raising or
    falling back to the root.
    """
    if parent_keys is None:
        return requested
    if requested is None:
        return parent_keys
    if isinstance(requested, str):
        requested = [requested]
    held = set(parent_keys)
    return [k for k in requested if k in held]


class IndexRXFilter(RXFilter):
    _graph: "GraphView | IndexedRXGraph"

    def __init__(
        self,
        *attr_comps: Filter,
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
        *,
        mode: "ViewMode | None" = None,
        root_fallback: bool = False,
    ) -> "GraphView":
        from tracksdata.graph._graph_view import GraphView, ViewMode

        node_ids = self.node_ids()

        rx_graph, node_map = self._graph._rx_subgraph_with_nodemap(node_ids)
        if self._edge_attr_comps:
            _filter_func = _create_filter_func(self._edge_attr_comps, self._graph._edge_attr_schemas())
            for src, tgt, attr in rx_graph.weighted_edge_list():
                if not _filter_func(attr):
                    rx_graph.remove_edge(src, tgt)

        root = self._graph
        if hasattr(self._graph, "_root"):
            # A view of a partial view is flattened onto the shared root, but it
            # holds only what its parent held -- the rx payloads above were
            # copied from the parent, not re-read from the root. Both the key
            # lists and the fallback flag have to be inherited, or the child
            # advertises the root's columns while holding no values for them.
            root = self._graph._root
            node_attr_keys = _clamp_to_parent(node_attr_keys, self._graph._node_attr_keys)
            edge_attr_keys = _clamp_to_parent(edge_attr_keys, self._graph._edge_attr_keys)
            root_fallback = root_fallback or self._graph._root_fallback

        graph_view = GraphView(
            rx_graph,
            node_map_to_root=dict(node_map.items()),
            root=root,
            mode=mode if mode is not None else ViewMode.WRITE_THROUGH,
            node_attr_keys=node_attr_keys,
            edge_attr_keys=edge_attr_keys,
            root_fallback=root_fallback,
        )

        return graph_view

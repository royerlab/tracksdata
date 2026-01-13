import abc
import functools
import operator
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import geff
import numpy as np
import polars as pl
import rustworkx as rx
from geff.core_io import construct_var_len_props, write_arrays
from geff_spec import Axis, GeffMetadata, PropMetadata
from numpy.typing import ArrayLike
from psygnal import Signal
from zarr.storage import StoreLike

from tracksdata.attrs import AttrComparison, NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.utils._cache import cache_method
from tracksdata.utils._dtypes import (
    column_to_numpy,
    infer_default_value,
    polars_dtype_to_numpy_dtype,
)
from tracksdata.utils._logging import LOG
from tracksdata.utils._multiprocessing import multiprocessing_apply

if TYPE_CHECKING:
    from traccuracy import TrackingGraph

    from tracksdata.graph.filters._base_filter import BaseFilter
    from tracksdata.graph.filters._spatial_filter import (
        BBoxSpatialFilter,
        SpatialFilter,
    )
    from tracksdata.metrics._matching import Matching
else:
    TrackingGraph = Any


T = TypeVar("T", bound="BaseGraph")


class BaseGraph(abc.ABC):
    """
    Base class for a graph backend.
    """

    node_added = Signal(int)
    node_removed = Signal(int)

    def __init__(self) -> None:
        self._cache = {}

    def supports_custom_indices(self) -> bool:
        """
        Whether the graph backend supports custom indices.
        """
        return False

    @staticmethod
    def _validate_attributes(
        attrs: dict[str, Any],
        reference_keys: list[str],
        mode: str,
    ) -> None:
        """
        Validate the attributes of a node.

        Parameters
        ----------
        attrs : dict[str, Any]
            The attributes to validate.
        reference_keys : list[str]
            The keys to validate against.
        mode : str
            The mode to validate against, for example "node" or "edge".
        """
        for key in attrs.keys():
            if key not in reference_keys:
                raise ValueError(
                    f"{mode} attribute key '{key}' not found in existing keys: "
                    f"'{reference_keys}'\nInitialize with "
                    f"`graph.add_{mode}_attr_key(key, default_value)`"
                )

        for ref_key in reference_keys:
            if ref_key not in attrs.keys() and ref_key != DEFAULT_ATTR_KEYS.NODE_ID:
                raise ValueError(
                    f"Attribute '{ref_key}' not found in attrs: '{attrs.keys()}'\nRequested keys: '{reference_keys}'"
                )

    @abc.abstractmethod
    def add_node(
        self,
        attrs: dict[str, Any],
        validate_keys: bool = True,
        index: int | None = None,
    ) -> int:
        """
        Add a node to the graph at time t.

        Parameters
        ----------
        attrs : Any
            The attributes of the node to be added, must have a "t" key.
            The keys of the attributes will be used as the attributes of the node.
            For example:
            ```python
            graph.add_node(dict(t=0, label="A", intensity=100))
            ```
        validate_keys : bool
            Whether to check if the attributes keys are valid.
            If False, the attributes keys will not be checked,
            useful to speed up the operation when doing bulk insertions.
        index : int | None
            Optional node index/ID to use. If None, the backend will assign
            an appropriate ID. Only supported by certain backends (e.g., SQLGraph).

        Returns
        -------
        int
            The ID of the added node.
        """

    def bulk_add_nodes(
        self,
        nodes: list[dict[str, Any]],
        indices: list[int] | None = None,
    ) -> list[int]:
        """
        Faster method to add multiple nodes to the graph with less overhead and fewer checks.

        Parameters
        ----------
        nodes : list[dict[str, Any]]
            The data of the nodes to be added.
            The keys of the data will be used as the attributes of the nodes.
            Must have "t" key.
        indices : list[int] | None
            Optional list of node indices/IDs to use. If None, the backend will assign
            appropriate IDs. Only supported by certain backends (e.g., SQLGraph).
            Must be the same length as nodes if provided.

        Returns
        -------
        list[int]
            The IDs of the added nodes.
        """
        if len(nodes) == 0:
            return []

        self._validate_indices_length(nodes, indices)

        # this method benefits the SQLGraph backend
        if indices is None:
            return [self.add_node(node, validate_keys=False) for node in nodes]
        else:
            return [
                self.add_node(node, validate_keys=False, index=idx) for node, idx in zip(nodes, indices, strict=True)
            ]

    def _validate_indices_length(self, nodes: list[dict[str, Any]], indices: list[int] | None) -> None:
        if indices is not None and len(indices) != len(nodes):
            raise ValueError(f"Length of indices ({len(indices)}) must match length of nodes ({len(nodes)})")

    @abc.abstractmethod
    def remove_node(self, node_id: int) -> None:
        """
        Remove a node from the graph.

        This method removes the specified node and all edges connected to it
        (both incoming and outgoing edges).

        Parameters
        ----------
        node_id : int
            The ID of the node to remove.

        Raises
        ------
        ValueError
            If the node_id does not exist in the graph.
        """

    @abc.abstractmethod
    def add_edge(
        self,
        source_id: int,
        target_id: int,
        attrs: dict[str, Any],
        validate_keys: bool = True,
    ) -> int:
        """
        Add an edge to the graph.

        Parameters
        ----------
        source_id : int
            The ID of the source node.
        target_id : int
            The ID of the target node.
        attrs : dict[str, Any]
            Additional attributes for the edge.
        validate_keys : bool
            Whether to check if the attributes keys are valid.
            If False, the attributes keys will not be checked,
            useful to speed up the operation when doing bulk insertions.

        Returns
        -------
        int
            The ID of the added edge.
        """

    @abc.abstractmethod
    def remove_edge(
        self,
        source_id: int | None = None,
        target_id: int | None = None,
        *,
        edge_id: int | None = None,
    ) -> None:
        """
        Remove an edge from the graph.

        Either provide `edge_id` to remove by edge identifier, or
        provide both `source_id` and `target_id` to remove by endpoints.

        Parameters
        ----------
        source_id : int | None
            The ID of the source node (when removing by endpoints).
        target_id : int | None
            The ID of the target node (when removing by endpoints).
        edge_id : int | None
            The ID of the edge to delete (when removing by ID).

        Raises
        ------
        ValueError
            If the specified edge does not exist or insufficient identifiers are provided.
        """

    @overload
    def bulk_add_edges(
        self,
        edges: list[dict[str, Any]],
        return_ids: Literal[False],
    ) -> None: ...

    @overload
    def bulk_add_edges(
        self,
        edges: list[dict[str, Any]],
        return_ids: Literal[True],
    ) -> list[int]: ...

    def bulk_add_edges(
        self,
        edges: list[dict[str, Any]],
        return_ids: bool = False,
    ) -> list[int] | None:
        """
        Faster method to add multiple edges to the graph with less overhead and fewer checks.

        Parameters
        ----------
        edges : list[dict[str, Any]]
            The data of the edges to be added.
            The keys of the data will be used as the attributes of the edges.
            Must have "source_id" and "target_id" keys.
        return_ids : bool
            Whether to return the IDs of the added edges.
            If False, the edges are added and the method returns None.

        Examples
        --------
        ```python
        edges = [
            {"source_id": 1, "target_id": 2, "weight": 0.8},
            {"source_id": 2, "target_id": 3, "weight": 0.9"},
        ]
        graph.bulk_add_edges(edges)
        ```

        Returns
        -------
        list[int] | None
            The IDs of the added edges.
        """
        # this method benefits the SQLGraph backend
        if return_ids:
            edge_ids = []
            for edge in edges:
                edge_ids.append(
                    self.add_edge(
                        edge.pop(DEFAULT_ATTR_KEYS.EDGE_SOURCE),
                        edge.pop(DEFAULT_ATTR_KEYS.EDGE_TARGET),
                        edge,
                        validate_keys=False,
                    )
                )
            return edge_ids

        # avoiding many ifs and appends
        for edge in edges:
            self.add_edge(
                edge.pop(DEFAULT_ATTR_KEYS.EDGE_SOURCE),
                edge.pop(DEFAULT_ATTR_KEYS.EDGE_TARGET),
                edge,
                validate_keys=False,
            )

    def add_overlap(
        self,
        source_id: int,
        target_id: int,
    ) -> int:
        """
        Add a new overlap to the graph.
        Overlapping nodes are mutually exclusive.

        Parameters
        ----------
        source_id : int
            The ID of the source node.
        target_id : int
            The ID of the target node.

        Returns
        -------
        int
            The ID of the added overlap.
        """
        raise NotImplementedError(f"{self.__class__.__name__} backend does not support overlaps.")

    def bulk_add_overlaps(
        self,
        overlaps: list[list[int, 2]],
    ) -> None:
        """
        Add multiple overlaps to the graph.
        Overlapping nodes are mutually exclusive.

        Parameters
        ----------
        overlaps : list[list[int, 2]]
            The IDs of the nodes to add the overlaps for.

        See Also
        --------
        [add_overlap][tracksdata.graph.BaseGraph.add_overlap]:
            Add a single overlap to the graph.
        """
        for source_id, target_id in overlaps:
            self.add_overlap(source_id, target_id)

    def overlaps(
        self,
        node_ids: list[int] | None = None,
    ) -> list[list[int, 2]]:
        """
        Get the overlaps between the nodes in `node_ids`.
        If `node_ids` is None, all nodes are used.

        Parameters
        ----------
        node_ids : list[int] | None
            The IDs of the nodes to get the overlaps for.
            If None, all nodes are used.

        Returns
        -------
        list[list[int, 2]]
            The overlaps between the nodes in `node_ids`.
        """
        return []

    def has_overlaps(self) -> bool:
        """
        Check if the graph has any overlaps.

        Returns
        -------
        bool
            True if the graph has any overlaps, False otherwise.
        """
        return False

    @overload
    def successors(
        self,
        node_ids: int,
        attr_keys: Sequence[str] | str | None = ...,
        *,
        return_attrs: Literal[True],
    ) -> pl.DataFrame: ...

    @overload
    def successors(
        self,
        node_ids: list[int] | None,
        attr_keys: Sequence[str] | str | None = ...,
        *,
        return_attrs: Literal[True],
    ) -> dict[int, pl.DataFrame]: ...

    @overload
    def successors(
        self,
        node_ids: int,
        attr_keys: Sequence[str] | str | None = ...,
        *,
        return_attrs: Literal[False] = False,
    ) -> list[int]: ...

    @overload
    def successors(
        self,
        node_ids: list[int] | None,
        attr_keys: Sequence[str] | str | None = ...,
        *,
        return_attrs: Literal[False] = False,
    ) -> dict[int, list[int]]: ...

    @abc.abstractmethod
    def successors(
        self,
        node_ids: list[int] | int | None,
        attr_keys: Sequence[str] | str | None = None,
        *,
        return_attrs: bool = False,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]:
        """
        Get the sucessors of a list of nodes.

        Parameters
        ----------
        node_ids : list[int] | int | None
            The IDs of the nodes to get the sucessors for.
            If None, all nodes are used.
        attr_keys : Sequence[str] | str | None
            The attribute keys to retrieve when ``return_attrs`` is True.
            If None, all attributes are included.
        return_attrs : bool, default False
            Whether to return node attributes in a `polars.DataFrame`. When False only
            the successor node IDs are returned.

        Returns
        -------
        dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]
            When ``return_attrs`` is True, returns a DataFrame for a single node or a dictionary
            mapping each node ID to a DataFrame of neighbor attributes. Otherwise returns a list
            of neighbor node IDs for a single node or a dictionary mapping each node ID to its
            neighbor ID list.
        """

    @overload
    def predecessors(
        self,
        node_ids: int,
        attr_keys: Sequence[str] | str | None = ...,
        *,
        return_attrs: Literal[True],
    ) -> pl.DataFrame: ...

    @overload
    def predecessors(
        self,
        node_ids: list[int] | None,
        attr_keys: Sequence[str] | str | None = ...,
        *,
        return_attrs: Literal[True],
    ) -> dict[int, pl.DataFrame]: ...

    @overload
    def predecessors(
        self,
        node_ids: int,
        attr_keys: Sequence[str] | str | None = ...,
        *,
        return_attrs: Literal[False] = False,
    ) -> list[int]: ...

    @overload
    def predecessors(
        self,
        node_ids: list[int] | None,
        attr_keys: Sequence[str] | str | None = ...,
        *,
        return_attrs: Literal[False] = False,
    ) -> dict[int, list[int]]: ...

    @abc.abstractmethod
    def predecessors(
        self,
        node_ids: list[int] | int | None,
        attr_keys: Sequence[str] | str | None = None,
        *,
        return_attrs: bool = False,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]:
        """
        Get the predecessors of a list of nodes.

        Parameters
        ----------
        node_ids : list[int] | int | None
            The IDs of the nodes to get the predecessors for. If None, all nodes are used.
        attr_keys : Sequence[str] | str | None
            The attribute keys to retrieve when ``return_attrs`` is True.
            If None, all attributes are included.
        return_attrs : bool, default False
            Whether to return node attributes in a `polars.DataFrame`. When False only
            the predecessor node IDs are returned.

        Returns
        -------
        dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]
            When ``return_attrs`` is True, returns a DataFrame for a single node or a dictionary
            mapping each node ID to a DataFrame of neighbor attributes. Otherwise returns a list
            of neighbor node IDs for a single node or a dictionary mapping each node ID to its
            neighbor ID list.
        """

    def _validate_subgraph_args(
        self,
        node_ids: Sequence[int] | None = None,
        node_attr_comps: list[AttrComparison] | None = None,
        edge_attr_comps: list[AttrComparison] | None = None,
    ) -> None:
        if node_ids is None and not node_attr_comps and not edge_attr_comps:
            raise ValueError("Either node IDs or one of the attributes' comparisons must be provided")

    @abc.abstractmethod
    def node_ids(self) -> list[int]:
        """
        Get the IDs of all nodes in the graph.
        """

    @abc.abstractmethod
    def edge_ids(self) -> list[int]:
        """
        Get the IDs of all edges in the graph.
        """

    @abc.abstractmethod
    def filter(
        self,
        *attr_filters: AttrComparison,
        node_ids: Sequence[int] | None = None,
        include_targets: bool = False,
        include_sources: bool = False,
    ) -> "BaseFilter":
        """
        Creates a filter object that can be used to create a subgraph or query ids and attributes.

        Parameters
        ----------
        *attr_filters : AttrComparison
            The attributes to filter the nodes by.
        node_ids : Sequence[int] | None
            The IDs of the nodes to include in the filter.
            If None, all nodes are used.
        include_targets : bool
            Whether to include edges out-going from the given node_ids even
            if the target node is not in the given node_ids.
        include_sources : bool
            Whether to include edges incoming to the given node_ids even
            if the source node is not in the given node_ids.

        Returns
        -------
        BaseFilter
            A filter object that can be used to create a subgraph or query attributes.
        """

    @abc.abstractmethod
    def time_points(self) -> list[int]:
        """
        Get the unique time points in the graph.
        """

    @abc.abstractmethod
    def node_attrs(
        self,
        *,
        attr_keys: Sequence[str] | str | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        """
        Get the attributes of the nodes as a pandas DataFrame.

        Parameters
        ----------
        attr_keys : Sequence[str] | str | None
            The attribute keys to get.
            If None, all attributesare used.
        unpack : bool
            Whether to unpack array attributes into multiple scalar attributes.

        Returns
        -------
        pl.DataFrame
            A polars DataFrame with the attributes of the nodes.
        """

    @abc.abstractmethod
    def edge_attrs(
        self,
        *,
        attr_keys: Sequence[str] | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        """
        Get the attributes of the edges as a polars DataFrame.

        Parameters
        ----------
        attr_keys : Sequence[str] | None
            The attribute keys to get.
            If None, all attributesare used.
        unpack : bool
            Whether to unpack array attributesinto multiple scalar attributes.
        """

    @abc.abstractmethod
    def node_attr_keys(self) -> list[str]:
        """
        Get the keys of the attributes of the nodes.
        """

    @abc.abstractmethod
    def edge_attr_keys(self) -> list[str]:
        """
        Get the keys of the attributes of the edges.
        """

    @abc.abstractmethod
    def add_node_attr_key(self, key: str, default_value: Any) -> None:
        """
        Add a new attribute key to the graph.
        All existing nodes will have the default value for the new attribute key.
        """

    @abc.abstractmethod
    def remove_node_attr_key(self, key: str) -> None:
        """
        Remove an existing node attribute key from the graph.

        Parameters
        ----------
        key : str
            The attribute key to remove.
        """

    @abc.abstractmethod
    def add_edge_attr_key(self, key: str, default_value: Any) -> None:
        """
        Add a new attribute key to the graph.
        All existing edges will have the default value for the new attribute key.
        """

    @abc.abstractmethod
    def remove_edge_attr_key(self, key: str) -> None:
        """
        Remove an existing edge attribute key from the graph.

        Parameters
        ----------
        key : str
            The attribute key to remove.
        """

    @abc.abstractmethod
    def num_edges(self) -> int:
        """
        The number of edges in the graph.
        """

    @abc.abstractmethod
    def num_nodes(self) -> int:
        """
        The number of nodes in the graph.
        """

    @abc.abstractmethod
    def update_node_attrs(
        self,
        *,
        attrs: dict[str, Any],
        node_ids: Sequence[int] | None = None,
    ) -> None:
        """
        Update the attributes of the nodes.

        Parameters
        ----------
        attrs : dict[str, Any]
            The attributes to update.
        node_ids : Sequence[int] | None
            The IDs of the nodes to update or None to update all nodes.
        """

    @abc.abstractmethod
    def update_edge_attrs(
        self,
        *,
        attrs: dict[str, Any],
        edge_ids: Sequence[int] | None = None,
    ) -> None:
        """
        Update the attributes of the edges.

        Parameters
        ----------
        attrs : dict[str, Any]
            Attributes to be updated.
        edge_ids : Sequence[int] | None
            The IDs of the edges to update or None to update all edges.
        """

    @classmethod
    def from_ctc(cls: type[T], data_dir: str | Path, **kwargs) -> T:
        """
        Create a graph from a CTC data directory.

        Parameters
        ----------
        data_dir : str | Path
            The path to the CTC data directory.
        **kwargs : Any
            Additional arguments to pass to the graph constructor.

        Examples
        --------
        ```python
        graph = BaseGraph.from_ctc("Fluo-N2DL-HeLa/01_GT/TRA")
        ```

        See Also
        --------
        [from_ctc][tracksdata.io._ctc.from_ctc]:
            Load a CTC ground truth file into a graph.

        [RegionPropsNodes][tracksdata.nodes.RegionPropsNodes]:
            Operator to create nodes from label images.

        Returns
        -------
        BaseGraph
            A graph with the nodes and edges from the CTC data directory.

        See Also
        --------
        [to_ctc][tracksdata.graph.BaseGraph.to_ctc]:
            Save a graph to a CTC ground truth directory.
        """
        from tracksdata.io._ctc import from_ctc

        graph = cls(**kwargs)
        from_ctc(data_dir, graph)
        return graph

    def to_ctc(
        self,
        output_dir: str | Path,
        *,
        shape: tuple[int, ...] | None = None,
        tracklet_id_key: str = DEFAULT_ATTR_KEYS.TRACKLET_ID,
        overwrite: bool = False,
    ) -> None:
        """
        Save the graph to a CTC ground truth directory.

        Parameters
        ----------
        output_dir : str | Path
            The directory to save the graph to.
        shape : tuple[int, ...]
            The shape of the label images (T, (Z), Y, X).
            If None, the shape is inferred from the graph metadata `shape` key.
        tracklet_id_key : str
            The attribute key to use for the track IDs.
        overwrite : bool
            Whether to overwrite the output directory if it exists.

        Examples
        --------
        ```python
        # ...
        solution_graph = solver.solve(graph)
        solution_graph.assign_tracklet_ids()
        solution_graph.to_ctc(shape=(10, 100, 100), output_dir="01_RES")
        ```

        See Also
        --------
        [to_ctc][tracksdata.io.to_ctc]:
            Save a graph to a CTC ground truth directory.
        """
        from tracksdata.io._ctc import to_ctc

        to_ctc(
            graph=self,
            shape=shape,
            output_dir=output_dir,
            tracklet_id_key=tracklet_id_key,
            overwrite=overwrite,
        )

    @classmethod
    def from_array(
        cls: type[T],
        positions: ArrayLike,
        tracklet_ids: ArrayLike | None = None,
        tracklet_id_graph: dict[int, int] | None = None,
        **kwargs,
    ) -> T:
        """
        Create a graph from a numpy array.

        Parameters
        ----------
        positions : np.ndarray
            (N, 4 or 3) dimensional array of positions.
            Defined by (T, (Z), Y, X) coordinates.
        tracklet_ids : np.ndarray | None
            Track ids of the nodes if available.
        tracklet_id_graph : dict[int, int] | None
            Mapping of division as child track id (key) to parent track id (value) relationships.
        **kwargs : Any
            Additional arguments to pass to the graph constructor.

        Returns
        -------
        BaseGraph
            A graph with the nodes and edges from the numpy array.
        """
        from tracksdata.io._numpy_array import from_array

        graph = cls(**kwargs)
        from_array(
            positions=np.asarray(positions),
            graph=graph,
            tracklet_ids=tracklet_ids,
            tracklet_id_graph=tracklet_id_graph,
        )
        return graph

    @abc.abstractmethod
    @overload
    def in_degree(self, node_ids: int) -> int: ...

    @abc.abstractmethod
    @overload
    def in_degree(self, node_ids: list[int] | None = None) -> list[int]: ...

    @abc.abstractmethod
    @overload
    def out_degree(self, node_ids: int) -> int: ...

    @abc.abstractmethod
    @overload
    def out_degree(self, node_ids: list[int] | None = None) -> list[int]: ...

    @abc.abstractmethod
    def in_degree(self, node_ids: list[int] | int | None = None) -> list[int] | int:
        """
        Get the in-degree of a list of nodes.
        """

    @abc.abstractmethod
    def out_degree(self, node_ids: list[int] | int | None = None) -> list[int] | int:
        """
        Get the out-degree of a list of nodes.
        """

    def match(
        self,
        other: "BaseGraph",
        matching: "Matching | None" = None,
        matched_node_id_key: str = DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
        match_score_key: str = DEFAULT_ATTR_KEYS.MATCH_SCORE,
        matched_edge_mask_key: str = DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK,
    ) -> None:
        """
        Match the nodes of the graph to the nodes of another graph.

        Parameters
        ----------
        other : BaseGraph
            The other graph to match to.
        matching : Matching | None
            The matching strategy to use. If None, defaults to
            MaskMatching with optimal=True and min_reference_intersection=0.5.
            See [MaskMatching][tracksdata.metrics.MaskMatching] and
            [DistanceMatching][tracksdata.metrics.DistanceMatching] for available strategies.
        matched_node_id_key : str
            The key of the output value of the corresponding node ID in the other graph.
        match_score_key : str
            The key of the output value of the match score between matched nodes.
        matched_edge_mask_key : str
            The key of the output as a boolean value indicating if a corresponding edge exists in the other graph.

        Examples
        --------
        Match using default mask-based matching:

        ```python
        graph1.match(graph2)
        ```

        Match using distance-based matching:

        ```python
        from tracksdata.metrics import DistanceMatching

        matching = DistanceMatching(optimal=True, max_distance=10.0, centroid_keys=("y", "x"))
        graph1.match(graph2, matching=matching)
        ```

        Match with custom mask matching threshold:

        ```python
        from tracksdata.metrics import MaskMatching

        matching = MaskMatching(optimal=True, min_reference_intersection=0.7)
        graph1.match(graph2, matching=matching)
        ```
        """
        from tracksdata.metrics._ctc_metrics import _matching_data
        from tracksdata.metrics._matching import MaskMatching

        if matching is None:
            matching = MaskMatching(optimal=True, min_reference_intersection=0.5)

        matching_data = _matching_data(
            self,
            other,
            input_graph_key=DEFAULT_ATTR_KEYS.NODE_ID,
            reference_graph_key=DEFAULT_ATTR_KEYS.NODE_ID,
            matching=matching,
        )

        if matched_node_id_key not in self.node_attr_keys():
            self.add_node_attr_key(matched_node_id_key, -1)

        if match_score_key not in self.node_attr_keys():
            self.add_node_attr_key(match_score_key, 0.0)

        if matched_edge_mask_key not in self.edge_attr_keys():
            self.add_edge_attr_key(matched_edge_mask_key, False)

        node_ids = functools.reduce(operator.iadd, matching_data["mapped_comp"])
        other_ids = functools.reduce(operator.iadd, matching_data["mapped_ref"])
        scores = functools.reduce(operator.iadd, matching_data["scores"])

        if len(node_ids) == 0:
            LOG.warning("No matching nodes found.")
            return

        self.update_node_attrs(
            node_ids=node_ids,
            attrs={matched_node_id_key: other_ids, match_score_key: scores},
        )

        other_to_node_ids = dict(zip(other_ids, node_ids, strict=True))

        self_edges_df = self.edge_attrs(attr_keys=[])
        other_edges_df = other.edge_attrs(attr_keys=[])

        other_edges_df = other_edges_df.with_columns(
            other_edges_df[col].map_elements(other_to_node_ids.get, return_dtype=pl.Int64).alias(col)
            for col in [DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]
        )

        edge_ids = self_edges_df.join(
            other_edges_df,
            on=[DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET],
            how="inner",
        )[DEFAULT_ATTR_KEYS.EDGE_ID]

        if len(edge_ids) == 0:
            LOG.warning("No matching edges found.")
            return

        self.update_edge_attrs(
            edge_ids=edge_ids,
            attrs={matched_edge_mask_key: True},
        )

    @classmethod
    def from_other(cls: type[T], other: "BaseGraph", **kwargs) -> T:
        """
        Create a graph from another graph.

        Parameters
        ----------
        other : BaseGraph
            The other graph to create a new graph from.
        **kwargs : Any
            Additional arguments to pass to the graph constructor.

        Returns
        -------
        BaseGraph
            A graph with the nodes and edges from the other graph.
        """
        # add node attributes
        node_attrs = other.node_attrs()
        other_node_ids = node_attrs[DEFAULT_ATTR_KEYS.NODE_ID]
        node_attrs = node_attrs.drop(DEFAULT_ATTR_KEYS.NODE_ID)

        graph = cls(**kwargs)

        for col in node_attrs.columns:
            if col != DEFAULT_ATTR_KEYS.T:
                first_value = node_attrs[col].first()
                graph.add_node_attr_key(col, infer_default_value(first_value))

        if graph.supports_custom_indices():
            new_node_ids = graph.bulk_add_nodes(
                list(node_attrs.rows(named=True)),
                indices=other_node_ids.to_list(),
            )
        else:
            new_node_ids = graph.bulk_add_nodes(list(node_attrs.rows(named=True)))
            if other.supports_custom_indices():
                LOG.warning(
                    f"Other graph ({type(other).__name__}) supports custom indices, but this graph "
                    f"({type(graph).__name__}) does not, indexing automatically."
                )

        # mapping from old node ids to new node ids
        node_map = dict(zip(other_node_ids, new_node_ids, strict=True))

        # add edge attributes
        edge_attrs = other.edge_attrs()
        edge_attrs = edge_attrs.drop(DEFAULT_ATTR_KEYS.EDGE_ID)

        for col in edge_attrs.columns:
            if col not in [DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]:
                graph.add_edge_attr_key(col, edge_attrs[col].first())

        edge_attrs = edge_attrs.with_columns(
            edge_attrs[col].map_elements(node_map.get, return_dtype=pl.Int64).alias(col)
            for col in [DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]
        )
        graph.bulk_add_edges(list(edge_attrs.rows(named=True)))

        if other.has_overlaps():
            overlaps = other.overlaps()
            overlaps = np.vectorize(node_map.get)(np.asarray(overlaps, dtype=int))
            graph.bulk_add_overlaps(overlaps.tolist())

        return graph

    def compute_overlaps(self, iou_threshold: float = 0.0) -> None:
        """
        Find overlapping nodes within each frame and add them their overlap relation into the graph.

        Parameters
        ----------
        iou_threshold : float
            Nodes with an IoU greater than this threshold are considered overlapping.
            If 0, all nodes are considered overlapping.

        Examples
        --------
        ```python
        graph.set_overlaps(iou_threshold=0.5)
        ```
        """
        if iou_threshold < 0.0 or iou_threshold > 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0")

        def _estimate_overlaps(t: int) -> list[list[int, 2]]:
            node_attrs = self.filter(NodeAttr(DEFAULT_ATTR_KEYS.T) == t).node_attrs(
                attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.MASK],
            )
            node_ids = node_attrs[DEFAULT_ATTR_KEYS.NODE_ID].to_list()
            masks = node_attrs[DEFAULT_ATTR_KEYS.MASK].to_list()
            overlaps = []
            for i in range(len(masks)):
                mask_i = masks[i]
                for j in range(i + 1, len(masks)):
                    if mask_i.iou(masks[j]) > iou_threshold:
                        overlaps.append([node_ids[i], node_ids[j]])
            return overlaps

        for overlaps in multiprocessing_apply(
            func=_estimate_overlaps,
            sequence=self.time_points(),
            desc="Setting overlaps",
        ):
            self.bulk_add_overlaps(overlaps)

    def summary(
        self,
        attrs_stats: bool = False,
        print_summary: bool = True,
    ) -> str:
        """
        Print a summary of the graph.

        Parameters
        ----------
        attrs_stats : bool
            If true it will print statistics about the attributes of the nodes and edges.
        print_summary : bool
            If true it will print the summary of the graph.

        Returns
        -------
        str
            A string with the summary of the graph.
        """
        summary = ""

        summary += "Graph summary:\n"
        summary += f"Number of nodes: {self.num_nodes()}\n"
        summary += f"Number of edges: {self.num_edges()}\n"
        summary += f"Number of overlaps: {len(self.overlaps())}\n"

        time_points = self.time_points()
        summary += f"Number of distinct time points: {len(time_points)}\n"
        if len(time_points) > 0:
            summary += f"Start time: {min(time_points)}\n"
            summary += f"End time: {max(time_points)}\n"
        else:
            summary += "No time points found.\n"

        if attrs_stats:
            nodes_attrs = self.node_attrs()
            edges_attrs = self.edge_attrs()
            summary += "\nNodes attributes:\n"
            summary += str(nodes_attrs.describe())
            summary += "\nEdges attributes:\n"
            summary += str(edges_attrs.describe())

        if print_summary:
            print(summary)

        return summary

    def clear_cache(self) -> None:
        """
        Clear the cache of the graph.

        NOTE: in the future we might want to allow clearing the cache by function name.
        """
        self._cache.clear()

    @cache_method
    def spatial_filter(self, attr_keys: list[str] | None = None) -> "SpatialFilter":
        """
        Create a spatial filter for efficient spatial queries of graph nodes.

        This method creates a spatial index of graph nodes based on their spatial coordinates,
        enabling efficient querying of nodes within spatial regions of interest (ROI).

        IMPORTANT: Spatial filters are cached by default, but can be cleared with `graph.clear_cache()`.

        Parameters
        ----------
        attr_keys : list[str] | None, optional
            List of attribute keys to use as spatial coordinates. If None, defaults to
            [DEFAULT_ATTR_KEYS.T, DEFAULT_ATTR_KEYS.Z, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X]
            filtered to only include keys present in the graph.
            Common combinations include:
            - 2D: [DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X]
            - 3D: [DEFAULT_ATTR_KEYS.Z, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X] or
                  [DEFAULT_ATTR_KEYS.T, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X]
            - 4D: [DEFAULT_ATTR_KEYS.T, DEFAULT_ATTR_KEYS.Z, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X]

        Returns
        -------
        SpatialFilter
            A spatial filter object that can be used to query nodes within spatial regions
            using slice notation.

        Examples
        --------
        Create a 2D spatial filter for image coordinates:

        ```python
        spatial_filter = graph.spatial_filter(attr_keys=["y", "x"])
        # Query nodes in region y=[10, 50), x=[20, 60)
        subgraph = spatial_filter[10:50, 20:60].subgraph()
        ```

        Create a 4D spatiotemporal filter:

        ```python
        spatial_filter = graph.spatial_filter()  # Uses default ["t", "z", "y", "x"]
        # Query nodes in time=[0, 10), z=[0, 5), y=[10, 50), x=[20, 60)
        nodes_in_roi = spatial_filter[0:10, 0:5, 10:50, 20:60].node_attrs()
        ```

        See Also
        --------
        [SpatialFilter][tracksdata.graph.filters.SpatialFilter]:
            The point spatial filter query class.

        """
        from tracksdata.graph.filters._spatial_filter import SpatialFilter

        return SpatialFilter(self, attr_keys=attr_keys)

    @cache_method
    def bbox_spatial_filter(
        self,
        frame_attr_key: str | None = DEFAULT_ATTR_KEYS.T,
        bbox_attr_key: str = DEFAULT_ATTR_KEYS.BBOX,
    ) -> "BBoxSpatialFilter":
        """
        Create a spatial filter for efficient spatial queries of graph nodes using bounding boxes.

        This method creates a spatial index of graph nodes based on their bounding box coordinates,
        enabling efficient querying of nodes intersecting with spatial regions of interest (ROI).

        IMPORTANT: Bounding box spatial filters are cached by default, but can be cleared with `graph.clear_cache()`.

        Parameters
        ----------
        frame_attr_key : str | None
            The attribute key for the frame (time) coordinate.
            Default is `DEFAULT_ATTR_KEYS.T`.
            If None it will only use the bounding box coordinates.
        bbox_attr_key : str
            The attribute key for the bounding box coordinates.
            Defaults to `DEFAULT_ATTR_KEYS.BBOX`.
            The bounding box coordinates should be in the format:
            [min_x, min_y, min_z, ..., max_x, max_y, max_z, ...]
            where each dimension has a min and max value.

        Returns
        -------
        BBoxSpatialFilter
            A spatial filter object that can be used to query nodes within spatial regions
            using slice notation.

        Examples
        --------
        Create a 2D spatial filter for bounding boxes:

        ```python
        spatial_filter = graph.bbox_spatial_filter(frame_attr_key=None, box_attr_key="bbox")
        # Query nodes intersecting with region y=[10, 50), x=[20, 60)
        subgraph = spatial_filter[10:50, 20:60].subgraph()
        ```

        Create a 4D spatiotemporal filter:

        ```python
        spatial_filter = graph.bbox_spatial_filter()
        # Uses default ["t", "bbox"]
        # Query nodes intersecting with time=[0, 10), z=[0, 5), y=[10, 50), x=[20, 60)
        nodes_in_roi = spatial_filter[0:10, 0:5, 10:50, 20:60].node_attrs()
        ```

        See Also
        --------
        [BBoxSpatialFilter][tracksdata.graph.filters.BBoxSpatialFilter]:
            The bounding box spatial filter query class.
        """
        from tracksdata.graph.filters._spatial_filter import BBoxSpatialFilter

        return BBoxSpatialFilter(self, frame_attr_key=frame_attr_key, bbox_attr_key=bbox_attr_key)

    @overload
    def assign_tracklet_ids(
        self,
        output_key: str = DEFAULT_ATTR_KEYS.TRACKLET_ID,
        reset: bool = True,
        tracklet_id_offset: int | None = None,
        node_ids: list[int] | None = None,
        return_id_update: Literal[False] = False,
    ) -> rx.PyDiGraph: ...
    @overload
    def assign_tracklet_ids(
        self,
        output_key: str = DEFAULT_ATTR_KEYS.TRACKLET_ID,
        reset: bool = True,
        tracklet_id_offset: int | None = None,
        node_ids: list[int] | None = None,
        return_id_update: Literal[True] = True,
    ) -> tuple[rx.PyDiGraph, pl.DataFrame]: ...

    @abc.abstractmethod
    def assign_tracklet_ids(
        self,
        output_key: str = DEFAULT_ATTR_KEYS.TRACKLET_ID,
        reset: bool = True,
        tracklet_id_offset: int | None = None,
        node_ids: list[int] | None = None,
        return_id_update: bool = False,
    ) -> rx.PyDiGraph | tuple[rx.PyDiGraph, pl.DataFrame]:
        """
        Compute and assign track ids to nodes.
        Parameters
        ----------
        output_key : str
            The key of the output track id attribute.
        reset : bool
            Whether to reset the track ids of the graph. If True, the track ids will be reset to -1.
        tracklet_id_offset : int | None
            The starting track id, useful when assigning track ids to a subgraph.
            If None, the track ids will start from 1 or from the maximum existing track id + 1
            if the output_key already exists and reset is False.
        node_ids : list[int] | None
            The node ids to assign track ids to. If None, all nodes are used.
        return_id_update : bool
            Whether to return a DataFrame with the updated node ids and their previous and assigned track ids.

        Returns
        -------
        rx.PyDiGraph
            A compressed graph (parent -> child) with track ids lineage relationships.
            If node_ids is provided, it will only include linages including those nodes.
        pl.DataFrame
            A DataFrame with the updated node ids and their previous and assigned track ids.
            This has columns "node_id", f"{output_key}", and f"{output_key}_new".
            Only returned if return_id_update is True.
        """
        raise NotImplementedError(f"{self.__class__.__name__} backend does not support track id assignment.")

    def tracklet_nodes(self, seeds: list[int] | None) -> list[int]:
        """
        Compute the non-branching tracklets around the provided seed node_ids.

        Walks forward to successors only through nodes with exactly one successor,
        and backward to predecessors that also have out_degree == 1, until closure.

        Parameters
        ----------
        seeds : list[int]
            Seed node IDs where to start the closure.

        Returns
        -------
        list[int]
            Sorted unique node IDs forming the closure.
        """
        # NOTE: if this function becomes a bottleneck in the future it might be worth having
        # a specialized version per backend
        if seeds is None or len(seeds) == 0:
            return []

        track_node_ids: set[int] = set()
        active_ids: set[int] = set(seeds)

        while len(active_ids) > 0:
            track_node_ids.update(active_ids)

            # Successors: only nodes with exactly one successor
            succ_map = self.successors(node_ids=list(active_ids))
            successors = [int(nodes[0]) for nodes in succ_map.values() if len(nodes) == 1]

            # Predecessors: only nodes with exactly one predecessor and predecessor out_degree == 1
            pred_map = self.predecessors(node_ids=list(active_ids))
            predecessors = [int(nodes[0]) for nodes in pred_map.values() if len(nodes) == 1]

            if len(predecessors) > 0:
                out_degrees = self.out_degree(predecessors)
                if isinstance(out_degrees, int):
                    out_degrees = [out_degrees]
                predecessors = [node for node, degree in zip(predecessors, out_degrees, strict=True) if degree == 1]

            active_ids = (set(successors) | set(predecessors)) - track_node_ids

        return sorted(track_node_ids)

    def tracklet_graph(
        self,
        tracklet_id_key: str = DEFAULT_ATTR_KEYS.TRACKLET_ID,
        ignore_tracklet_id: int | None = None,
    ) -> rx.PyDiGraph:
        """
        Create a compressed tracklet graph where each node is a tracklet
        and each edge is a transition between tracklets.

        IMPORTANT:
        rx.PyDiGraph does not allow arbitrary indices, so we use the tracklet ids as node values.
        And edge values are the tuple of source and target tracklet ids.

        Parameters
        ----------
        tracklet_id_key : str
            The key of the track id attribute.
        ignore_tracklet_id : int | None
            The track id to ignore. If None, all track ids are used.

        Returns
        -------
        rx.PyDiGraph
            A compressed tracklet graph.

        See Also
        --------
        [rx_digraph_to_napari_dict][tracksdata.functional.rx_digraph_to_napari_dict]:
            Convert a tracklet graph to a napari-ready dictionary.
        """
        from tracksdata.functional._edges import join_node_attrs_to_edges

        if tracklet_id_key not in self.node_attr_keys():
            raise ValueError(f"Track id key '{tracklet_id_key}' not found in graph. Expected '{self.node_attr_keys()}'")

        nodes_df = self.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, tracklet_id_key])
        edges_df = self.edge_attrs(attr_keys=[])

        if ignore_tracklet_id is not None:
            nodes_df = nodes_df.filter(pl.col(tracklet_id_key) != ignore_tracklet_id)

        track_ids = nodes_df[tracklet_id_key].unique().to_list()
        tracklet_graph = rx.PyDiGraph()

        rx_ids = np.asarray(tracklet_graph.add_nodes_from(track_ids), dtype=int)
        track_id_to_rx_id = dict(zip(track_ids, rx_ids, strict=True))

        src_col = f"source_{tracklet_id_key}"
        tgt_col = f"target_{tracklet_id_key}"

        edges_df = (
            join_node_attrs_to_edges(
                nodes_df,
                edges_df,
                how="right",
            )
            .filter(pl.col(src_col) != pl.col(tgt_col))
            .with_columns(
                pl.col(src_col)
                .map_elements(track_id_to_rx_id.__getitem__, return_dtype=pl.Int64)
                .alias("source_rx_id"),
                pl.col(tgt_col)
                .map_elements(track_id_to_rx_id.__getitem__, return_dtype=pl.Int64)
                .alias("target_rx_id"),
            )
        )

        tracklet_graph.add_edges_from(
            zip(
                edges_df["source_rx_id"].to_list(),
                edges_df["target_rx_id"].to_list(),
                zip(edges_df[src_col].to_list(), edges_df[tgt_col].to_list(), strict=False),
                strict=True,
            )
        )

        return tracklet_graph

    @classmethod
    def from_geff(
        cls: type[T],
        geff_store: StoreLike,
        geff_read_kwargs: dict[str, Any] | None = None,
        node_attr_key_map: dict[str, str] | None = None,
        edge_attr_key_map: dict[str, str] | None = None,
        **kwargs,
    ) -> tuple[T, GeffMetadata]:
        """
        Create a graph from a geff data directory.

        Parameters
        ----------
        geff_store : StoreLike
            The store or path to the geff data directory to read the graph from.
        geff_read_kwargs : dict[str, Any] | None
            Additional keyword arguments to pass to the `geff.read` function.
        node_attr_key_map : dict[str, str] | None
            A mapping to rename node attribute keys.
            If a key is not in the mapping, it is not renamed.
        edge_attr_key_map : dict[str, str] | None
            A mapping to rename edge attribute keys.
            If a key is not in the mapping, it is not renamed.
        **kwargs
            Additional keyword arguments to pass to the graph constructor.

        Returns
        -------
        T
            The loaded graph.
        geff_metadata : GeffMetadata
            The geff metadata of the graph.
        """
        from tracksdata.graph import IndexedRXGraph

        if geff_read_kwargs is None:
            geff_read_kwargs = {}

        # this performs a roundtrip with the rustworkx graph
        rx_graph, geff_metadata = geff.read(geff_store, backend="rustworkx", **geff_read_kwargs)

        if not isinstance(rx_graph, rx.PyDiGraph):
            LOG.warning("The graph is not a directed graph, converting to directed graph.")
            rx_graph = rx_graph.to_directed()

        node_id_map = rx_graph.attrs["to_rx_id_map"]
        rx_graph.attrs = {"geff": rx_graph.attrs, **rx_graph.attrs["extra"].pop("tracksdata", {})}

        if node_attr_key_map is not None:
            LOG.info("Mapping node attributes: %s", node_attr_key_map)
            for src_k, dst_k in node_attr_key_map.items():
                geff_metadata.node_props_metadata[dst_k] = geff_metadata.node_props_metadata.pop(src_k)
                for node_attr in rx_graph.nodes():
                    node_attr[dst_k] = node_attr.pop(src_k)

        if edge_attr_key_map is not None:
            LOG.info("Mapping edge attributes: %s", edge_attr_key_map)
            for src_k, dst_k in edge_attr_key_map.items():
                geff_metadata.edge_props_metadata[dst_k] = geff_metadata.edge_props_metadata.pop(src_k)
                for edge_attr in rx_graph.edges():
                    edge_attr[dst_k] = edge_attr.pop(src_k)

        indexed_graph = IndexedRXGraph(
            rx_graph=rx_graph,
            node_id_map=node_id_map,
            **kwargs,
        )

        node_attr_key = indexed_graph.node_attr_keys()
        if DEFAULT_ATTR_KEYS.MASK in node_attr_key and DEFAULT_ATTR_KEYS.BBOX in node_attr_key:
            from tracksdata.nodes._mask import Mask

            # unsafe operation, changing graph content inplace
            for node_attr in indexed_graph.rx_graph.nodes():
                node_attr[DEFAULT_ATTR_KEYS.MASK] = Mask(
                    node_attr[DEFAULT_ATTR_KEYS.MASK].astype(bool),
                    bbox=node_attr[DEFAULT_ATTR_KEYS.BBOX],
                )

        if cls == IndexedRXGraph:
            return indexed_graph, geff_metadata

        return cls.from_other(indexed_graph, **kwargs), geff_metadata

    def to_geff(
        self,
        geff_store: StoreLike,
        geff_metadata: geff.GeffMetadata | None = None,
        zarr_format: Literal[2, 3] = 3,
    ) -> None:
        """
        Write the graph to a geff data directory.

        Parameters
        ----------
        geff_store : StoreLike
            The store or path to the geff data directory to write the graph to.
        geff_metadata : GeffMetadata | None
            The geff metadata to write to the graph.
            It automatically generates the metadata with:
            - axes: time (t) and spatial axes ((z), y, x)
            - tracklet node property: tracklet_id
        zarr_format : Literal[2, 3]
            The zarr format to write the graph to.
            Defaults to 3.
        """

        node_attrs = self.node_attrs()
        node_ids = node_attrs[DEFAULT_ATTR_KEYS.NODE_ID].to_numpy()
        node_attrs = node_attrs.drop(DEFAULT_ATTR_KEYS.NODE_ID)

        edge_attrs = self.edge_attrs().drop(DEFAULT_ATTR_KEYS.EDGE_ID)
        edge_ids = edge_attrs.select(DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET).to_numpy()
        edge_attrs = edge_attrs.drop(DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET)

        if geff_metadata is None:
            axes = [Axis(name=DEFAULT_ATTR_KEYS.T, type="time")]
            axes.extend(
                [
                    Axis(name=c, type="space")
                    for c in (DEFAULT_ATTR_KEYS.Z, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X)
                    if c in node_attrs.columns
                ]
            )

            if DEFAULT_ATTR_KEYS.TRACKLET_ID in node_attrs.columns:
                track_node_props = {
                    "tracklet": DEFAULT_ATTR_KEYS.TRACKLET_ID,
                }
            else:
                track_node_props = None

            node_props_metadata = {
                k: PropMetadata(
                    identifier=k,
                    dtype=polars_dtype_to_numpy_dtype(v.dtype) if k != DEFAULT_ATTR_KEYS.MASK else np.uint64,
                    varlength=k == DEFAULT_ATTR_KEYS.MASK,
                )
                for k, v in node_attrs.to_dict().items()
            }
            edge_props_metadata = {
                k: PropMetadata(identifier=k, dtype=polars_dtype_to_numpy_dtype(v.dtype))
                for k, v in edge_attrs.to_dict().items()
            }

            td_metadata = self.metadata().copy()
            td_metadata.pop("geff", None)  # avoid geff being written multiple times

            geff_metadata = geff.GeffMetadata(
                directed=True,
                axes=axes,
                node_props_metadata=node_props_metadata,
                edge_props_metadata=edge_props_metadata,
                track_node_props=track_node_props,
                extra={
                    "tracksdata": td_metadata,
                },
            )

        node_dict = {
            k: {"values": column_to_numpy(v), "missing": None}
            for k, v in node_attrs.to_dict().items()
            if k != DEFAULT_ATTR_KEYS.MASK
        }

        if DEFAULT_ATTR_KEYS.MASK in node_attrs.columns:
            node_dict[DEFAULT_ATTR_KEYS.MASK] = construct_var_len_props(
                [mask.mask.astype(np.uint64) for mask in node_attrs[DEFAULT_ATTR_KEYS.MASK]]
            )

        edge_dict = {k: {"values": column_to_numpy(v), "missing": None} for k, v in edge_attrs.to_dict().items()}

        write_arrays(
            geff_store,
            node_ids=node_ids.astype(np.uint64),
            node_props=node_dict,
            edge_ids=edge_ids.astype(np.uint64),
            edge_props=edge_dict,
            metadata=geff_metadata,
            zarr_format=zarr_format,
        )

    def to_traccuracy_graph(self, array_view_kwargs: dict[str, Any] | None = None) -> "TrackingGraph":
        """
        Convert the graph to a `traccuracy.TrackingGraph`.

        Parameters
        ----------
        array_view_kwargs : dict[str, Any] | None
            Additional keyword arguments to pass to the `GraphArrayView` constructor used to create the segmentation.

        Returns
        -------
        TrackingGraph
            A traccuracy graph.
        """
        from tracksdata.metrics._traccuracy import to_traccuracy_graph

        return to_traccuracy_graph(self, array_view_kwargs=array_view_kwargs)

    @abc.abstractmethod
    def has_node(self, node_id: int) -> bool:
        """
        Check if the graph has a node with the given id.
        """

    @abc.abstractmethod
    def has_edge(self, source_id: int, target_id: int) -> bool:
        """
        Check if the graph has an edge between two nodes.
        """

    @abc.abstractmethod
    def edge_id(self, source_id: int, target_id: int) -> int:
        """
        Return the edge id between two nodes.
        """

    def copy(self, **kwargs) -> "BaseGraph":
        """
        Create a copy of this graph.

        Returns
        -------
        BaseGraph
            A new graph instance with the same nodes, edges, and attributes as this graph.
        **kwargs : Any
            Additional arguments to pass to the graph constructor.

        Examples
        --------
        ```python
        copied_graph = graph.copy()
        ```

        See Also
        --------
        [from_other][tracksdata.graph.BaseGraph.from_other]:
            Create a graph from another graph.
        """
        return self.__class__.from_other(self, **kwargs)

    def __getitem__(self, node_id: int) -> "NodeInterface":
        """
        Helper method to interact with a single node.

        Parameters
        ----------
        node_id : int
            The id of the node to interact with.

        Returns
        -------
        NodeInterface
            A node interface for the given node id.
        """

        if not isinstance(node_id, int):
            raise ValueError(f"graph index must be a integer, found '{node_id}' of type {type(node_id)}")
        return NodeInterface(self, node_id)


class NodeInterface:
    """
    Helper class to interact with a single node.

    Parameters
    ----------
    graph : BaseGraph
        The graph to interact with.
    node_id : int
        The id of the node to interact with.

    See Also
    --------
    [BaseGraph][tracksdata.graph.BaseGraph] The base graph class.
    """

    def __init__(self, graph: BaseGraph, node_id: int):
        self._graph = graph
        self._node_id = node_id

    def __getitem__(self, key: str) -> Any:
        return self._graph.filter(node_ids=[self._node_id]).node_attrs(attr_keys=[key])[key].item()

    def __setitem__(self, key: str, value: Any) -> None:
        return self._graph.update_node_attrs(attrs={key: value}, node_ids=[self._node_id])

    def __str__(self) -> str:
        node_attr = self._graph.filter(node_ids=[self._node_id]).node_attrs()
        return str(node_attr)

    def __repr__(self) -> str:
        return str(self)

    def to_dict(self) -> dict[str, Any]:
        data = (
            self._graph.filter(node_ids=[self._node_id])
            .node_attrs()
            .drop(DEFAULT_ATTR_KEYS.NODE_ID)
            .rows(named=True)[0]
        )
        return data

    @abc.abstractmethod
    def metadata(self) -> dict[str, Any]:
        """
        Return the metadata of the graph.

        Returns
        -------
        dict[str, Any]
            The metadata of the graph as a dictionary.

        Examples
        --------
        ```python
        metadata = graph.metadata()
        print(metadata["shape"])
        ```
        """

    @abc.abstractmethod
    def update_metadata(self, **kwargs) -> None:
        """
        Set or update metadata for the graph.

        Parameters
        ----------
        **kwargs : Any
            The metadata items to set by key. Values will be stored as JSON.

        Examples
        --------
        ```python
        graph.update_metadata(shape=[1, 25, 25], path="path/to/image.ome.zarr")
        graph.update_metadata(description="Tracking data from experiment 1")
        ```
        """

    @abc.abstractmethod
    def remove_metadata(self, key: str) -> None:
        """
        Remove a metadata key from the graph.

        Parameters
        ----------
        key : str
            The key of the metadata to remove.

        Examples
        --------
        ```python
        graph.remove_metadata("shape")
        ```
        """

    @abc.abstractmethod
    def edge_list(self) -> list[list[int, int]]:
        """
        Get the edge list of the graph.
        """

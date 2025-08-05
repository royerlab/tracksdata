import abc
import functools
import operator
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np
import polars as pl
import rustworkx as rx
from geff import GeffMetadata
from geff.metadata_schema import Axis
from geff.rustworkx.io import read_rx
from geff.write_arrays import write_arrays
from numpy.typing import ArrayLike
from zarr.storage import StoreLike

from tracksdata.attrs import AttrComparison, NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.utils._dtypes import column_to_bytes, column_to_numpy
from tracksdata.utils._logging import LOG
from tracksdata.utils._multiprocessing import multiprocessing_apply

if TYPE_CHECKING:
    from tracksdata.graph.filters._base_filter import BaseFilter
    from tracksdata.graph.filters._spatial_filter import (
        BBoxSpatialFilter,
        SpatialFilter,
    )


T = TypeVar("T", bound="BaseGraph")


class BaseGraph(abc.ABC):
    """
    Base class for a graph backend.
    """

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
            if ref_key not in attrs.keys():
                raise ValueError(
                    f"Attribute '{ref_key}' not found in attrs: '{attrs.keys()}'\nRequested keys: '{reference_keys}'"
                )

    @abc.abstractmethod
    def add_node(
        self,
        attrs: dict[str, Any],
        validate_keys: bool = True,
        *args: Any,
        **kwargs: Any,
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
        *args: Any
            Unused arguments, included for compatibility with the child classes.
        **kwargs: Any
            Unused keyword arguments, included for compatibility with the child classes.

        Returns
        -------
        int
            The ID of the added node.
        """

    def bulk_add_nodes(
        self,
        nodes: list[dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> list[int]:
        """
        Faster method to add multiple nodes to the graph with less overhead and fewer checks.

        Parameters
        ----------
        nodes : list[dict[str, Any]]
            The data of the nodes to be added.
            The keys of the data will be used as the attributes of the nodes.
            Must have "t" key.
        *args: Any
            Unused arguments, included for compatibility with the child classes.
        **kwargs: Any
            Unused keyword arguments, included for compatibility with the child classes.

        Returns
        -------
        list[int]
            The IDs of the added nodes.
        """
        if len(nodes) == 0:
            return []

        # this method benefits the SQLGraph backend
        return [self.add_node(node, validate_keys=False) for node in nodes]

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

    @abc.abstractmethod
    def successors(
        self,
        node_ids: list[int] | int,
        attr_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        """
        Get the sucessors of a list of nodes.

        Parameters
        ----------
        node_ids : list[int] | int
            The IDs of the nodes to get the sucessors for.
        attr_keys : Sequence[str] | str | None
            The attribute keys to get.
            If None, all attributesare used.

        Returns
        -------
        dict[int, pl.DataFrame] | pl.DataFrame
            The sucessors of the nodes indexed by node ID if a list of nodes is provided.
        """

    @abc.abstractmethod
    def predecessors(
        self,
        node_ids: list[int] | int,
        attr_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        """
        Get the predecessors of a list of nodes.

        Parameters
        ----------
        node_ids : list[int] | int
            The IDs of the nodes to get the predecessors for.
        attr_keys : Sequence[str] | str | None
            The attribute keys to get.
            If None, all attributesare used.

        Returns
        -------
        dict[int, pl.DataFrame] | pl.DataFrame
            The predecessors of the nodes indexed by node ID if a list of nodes is provided.
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

    @property
    @abc.abstractmethod
    def node_attr_keys(self) -> list[str]:
        """
        Get the keys of the attributes of the nodes.
        """

    @property
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
    def add_edge_attr_key(self, key: str, default_value: Any) -> None:
        """
        Add a new attribute key to the graph.
        All existing edges will have the default value for the new attribute key.
        """

    @property
    @abc.abstractmethod
    def num_edges(self) -> int:
        """
        The number of edges in the graph.
        """

    @property
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
        shape: tuple[int, ...],
        output_dir: str | Path,
        track_id_key: str = DEFAULT_ATTR_KEYS.TRACK_ID,
        overwrite: bool = False,
    ) -> None:
        """
        Save the graph to a CTC ground truth directory.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape of the label images (T, (Z), Y, X)
        output_dir : str | Path
            The directory to save the graph to.
        track_id_key : str
            The attribute key to use for the track IDs.
        overwrite : bool
            Whether to overwrite the output directory if it exists.

        Examples
        --------
        ```python
        # ...
        solution_graph = solver.solve(graph)
        solution_graph.assign_track_ids()
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
            track_id_key=track_id_key,
            overwrite=overwrite,
        )

    @classmethod
    def from_array(
        cls: type[T],
        positions: ArrayLike,
        track_ids: ArrayLike | None = None,
        track_id_graph: dict[int, int] | None = None,
        **kwargs,
    ) -> T:
        """
        Create a graph from a numpy array.

        Parameters
        ----------
        positions : np.ndarray
            (N, 4 or 3) dimensional array of positions.
            Defined by (T, (Z), Y, X) coordinates.
        track_ids : np.ndarray | None
            Track ids of the nodes if available.
        track_id_graph : dict[int, int] | None
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
            track_ids=track_ids,
            track_id_graph=track_id_graph,
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
        matched_node_id_key : str
            The key of the output value of the corresponding node ID in the other graph.
        match_score_key : str
            The key of the output value of the match score between matched nodes
        matched_edge_mask_key : str
            The key of the output as a boolean value indicating if a corresponding edge exists in the other graph.
        """
        from tracksdata.metrics._ctc_metrics import _matching_data

        matching_data = _matching_data(
            self,
            other,
            input_graph_key=DEFAULT_ATTR_KEYS.NODE_ID,
            reference_graph_key=DEFAULT_ATTR_KEYS.NODE_ID,
            optimal_matching=True,
        )

        if matched_node_id_key not in self.node_attr_keys:
            self.add_node_attr_key(matched_node_id_key, -1)

        if match_score_key not in self.node_attr_keys:
            self.add_node_attr_key(match_score_key, 0.0)

        if matched_edge_mask_key not in self.edge_attr_keys:
            self.add_edge_attr_key(matched_edge_mask_key, False)

        node_ids = functools.reduce(operator.iadd, matching_data["mapped_comp"])
        other_ids = functools.reduce(operator.iadd, matching_data["mapped_ref"])
        ious = functools.reduce(operator.iadd, matching_data["ious"])

        if len(node_ids) == 0:
            LOG.warning("No matching nodes found.")
            return

        self.update_node_attrs(
            node_ids=node_ids,
            attrs={matched_node_id_key: other_ids, match_score_key: ious},
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
                graph.add_node_attr_key(col, node_attrs[col].first())

        new_node_ids = graph.bulk_add_nodes(list(node_attrs.rows(named=True)))
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
        summary += f"Number of nodes: {self.num_nodes}\n"
        summary += f"Number of edges: {self.num_edges}\n"
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

    def spatial_filter(self, attr_keys: list[str] | None = None) -> "SpatialFilter":
        """
        Create a spatial filter for efficient spatial queries of graph nodes.

        This method creates a spatial index of graph nodes based on their spatial coordinates,
        enabling efficient querying of nodes within spatial regions of interest (ROI).

        Parameters
        ----------
        attr_keys : list[str] | None, optional
            List of attribute keys to use as spatial coordinates. If None, defaults to
            ["t", "z", "y", "x"] filtered to only include keys present in the graph.
            Common combinations include:
            - 2D: ["y", "x"]
            - 3D: ["z", "y", "x"] or ["t", "y", "x"]
            - 4D: ["t", "z", "y", "x"]

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

    def bbox_spatial_filter(
        self,
        frame_attr_key: str | None = DEFAULT_ATTR_KEYS.T,
        bbox_attr_key: str = DEFAULT_ATTR_KEYS.BBOX,
    ) -> "BBoxSpatialFilter":
        """
        Create a spatial filter for efficient spatial queries of graph nodes using bounding boxes.

        This method creates a spatial index of graph nodes based on their bounding box coordinates,
        enabling efficient querying of nodes intersecting with spatial regions of interest (ROI).

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

    def tracklet_graph(
        self,
        track_id_key: str = DEFAULT_ATTR_KEYS.TRACK_ID,
        ignore_track_id: int | None = None,
    ) -> rx.PyDiGraph:
        """
        Create a compressed tracklet graph where each node is a tracklet
        and each edge is a transition between tracklets.

        IMPORTANT:
        rx.PyDiGraph does not allow arbitrary indices, so we use the tracklet ids as node values.
        And edge values are the tuple of source and target tracklet ids.

        Parameters
        ----------
        track_id_key : str
            The key of the track id attribute.
        ignore_track_id : int | None
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

        if track_id_key not in self.node_attr_keys:
            raise ValueError(f"Track id key '{track_id_key}' not found in graph. Expected '{self.node_attr_keys}'")

        nodes_df = self.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, track_id_key])
        edges_df = self.edge_attrs(attr_keys=[])

        if ignore_track_id is not None:
            nodes_df = nodes_df.filter(pl.col(track_id_key) != ignore_track_id)

        nodes_df = nodes_df.unique(subset=[track_id_key])

        graph = rx.PyDiGraph()
        nodes_df = nodes_df.with_columns(
            pl.Series(
                np.asarray(graph.add_nodes_from(nodes_df[track_id_key].to_list()), dtype=int),
            ).alias("rx_id"),
        )

        edges_df = (
            edges_df.join(
                nodes_df.rename({track_id_key: "source_track_id", "rx_id": "source_rx_id"}),
                left_on=DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                right_on=DEFAULT_ATTR_KEYS.NODE_ID,
                how="right",
            )
            .join(
                nodes_df.rename({track_id_key: "target_track_id", "rx_id": "target_rx_id"}),
                left_on=DEFAULT_ATTR_KEYS.EDGE_TARGET,
                right_on=DEFAULT_ATTR_KEYS.NODE_ID,
                how="right",
            )
            .filter(~pl.col(DEFAULT_ATTR_KEYS.EDGE_ID).is_null())
        )

        graph.add_edges_from(
            zip(
                edges_df["source_rx_id"].to_list(),
                edges_df["target_rx_id"].to_list(),
                zip(edges_df["source_track_id"].to_list(), edges_df["target_track_id"].to_list(), strict=False),
                strict=True,
            )
        )

        return graph

    @classmethod
    def from_geff(
        cls: type[T],
        geff_store: StoreLike,
        **kwargs,
    ) -> T:
        """
        Create a graph from a geff data directory.

        Parameters
        ----------
        geff_store : StoreLike
            The store or path to the geff data directory to read the graph from.
        **kwargs
            Additional keyword arguments to pass to the graph constructor.

        Returns
        -------
        T
            The loaded graph.
        """
        from tracksdata.graph import IndexedRXGraph

        # this performs a roundtrip with the rustworkx graph
        rx_graph, _ = read_rx(geff_store)

        if not isinstance(rx_graph, rx.PyDiGraph):
            LOG.warning("The graph is not a directed graph, converting to directed graph.")
            rx_graph = rx_graph.to_directed()

        indexed_graph = IndexedRXGraph(
            rx_graph=rx_graph,
            node_id_map=rx_graph.attrs["to_rx_id_map"],
            **kwargs,
        )

        if cls == IndexedRXGraph:
            return indexed_graph

        return cls.from_other(indexed_graph, **kwargs)

    def to_geff(
        self,
        geff_store: StoreLike,
        geff_metadata: GeffMetadata | None = None,
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
            - tracklet node property: track_id
        zarr_format : Literal[2, 3]
            The zarr format to write the graph to.
            Defaults to 3.
        """

        node_attrs = self.node_attrs()

        if DEFAULT_ATTR_KEYS.MASK in node_attrs.columns:
            node_attrs = column_to_bytes(node_attrs, DEFAULT_ATTR_KEYS.MASK)

        if geff_metadata is None:
            axes = [Axis(name=DEFAULT_ATTR_KEYS.T, type="time")]
            axes.extend([Axis(name=c, type="space") for c in ("z", "y", "x") if c in node_attrs.columns])

            if DEFAULT_ATTR_KEYS.TRACK_ID in node_attrs.columns:
                track_node_props = {
                    "tracklet": DEFAULT_ATTR_KEYS.TRACK_ID,
                }
            else:
                track_node_props = None

            geff_metadata = GeffMetadata(
                directed=True,
                axes=axes,
                track_node_props=track_node_props,
            )

        edge_attrs = self.edge_attrs().drop(DEFAULT_ATTR_KEYS.EDGE_ID)

        node_dict = {
            k: (column_to_numpy(v), None) for k, v in node_attrs.drop(DEFAULT_ATTR_KEYS.NODE_ID).to_dict().items()
        }

        edge_ids = edge_attrs.select(DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET).to_numpy()

        edge_dict = {
            k: (column_to_numpy(v), None)
            for k, v in edge_attrs.drop(
                DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                DEFAULT_ATTR_KEYS.EDGE_TARGET,
            )
            .to_dict()
            .items()
        }

        write_arrays(
            geff_store,
            metadata=geff_metadata,
            node_ids=node_attrs[DEFAULT_ATTR_KEYS.NODE_ID].to_numpy(),
            node_props=node_dict,
            edge_ids=edge_ids,
            edge_props=edge_dict,
            zarr_format=zarr_format,
        )

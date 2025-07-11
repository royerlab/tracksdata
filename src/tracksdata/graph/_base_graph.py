import abc
import functools
import operator
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from tracksdata.attrs import AttrComparison, NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.utils._logging import LOG
from tracksdata.utils._multiprocessing import multiprocessing_apply

if TYPE_CHECKING:
    from tracksdata.graph._base_filter import BaseFilter
    from tracksdata.graph._graph_view import GraphView


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

        Returns
        -------
        int
            The ID of the added node.
        """

    def bulk_add_nodes(
        self,
        nodes: list[dict[str, Any]],
    ) -> list[int]:
        """
        Faster method to add multiple nodes to the graph with less overhead and fewer checks.

        Parameters
        ----------
        nodes : list[dict[str, Any]]
            The data of the nodes to be added.
            The keys of the data will be used as the attributes of the nodes.
            Must have "t" key.

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

    @abc.abstractmethod
    def filter_nodes_by_attrs(
        self,
        *attrs: AttrComparison,
    ) -> list[int]:
        """
        Filter nodes by attributes.

        Parameters
        ----------
        attrs : AttrComparison
            Attributes to filter by, for example:
            ```python
            graph.filter_nodes_by_attrs(Attr("t") == 0, Attr("label") == "A")
            ```

        Returns
        -------
        list[int]
            The IDs of the filtered nodes.
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
        include_targets: bool = False,
        include_sources: bool = False,
    ) -> "BaseFilter":
        """
        Filter the graph by the given attributes.

        Parameters
        ----------
        *attr_filters : AttrComparison
            The attributes to filter the nodes by.
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
    def subgraph(
        self,
        *attr_filters: AttrComparison,
        node_ids: Sequence[int] | None = None,
        node_attr_keys: Sequence[str] | str | None = None,
        edge_attr_keys: Sequence[str] | str | None = None,
    ) -> "GraphView":
        """
        Create a subgraph from the graph from the given node IDs
        or attributes' filters.

        Node IDs or a single attribute filter can be used to create a subgraph.

        Parameters
        ----------
        node_ids : Sequence[int]
            The IDs of the nodes to include in the subgraph.
        *attr_filters : AttrComparison
            The attributes to filter the nodes by.
        node_attr_keys : Sequence[str] | str | None
            The attribute keys to get.
            If None, all attributesare used.
        edge_attr_keys : Sequence[str] | str | None
            The attribute keys to get.
            If None, all attributesare used.

        Returns
        -------
        GraphView
            A view of the graph with the specified nodes.
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
        node_ids: Sequence[int] | None = None,
        attr_keys: Sequence[str] | str | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        """
        Get the attributes of the nodes as a pandas DataFrame.

        Parameters
        ----------
        node_ids : list[int] | None
            The IDs of the nodes to get the attributes for.
            If None, all nodes are used.
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
        node_ids: list[int] | None = None,
        attr_keys: Sequence[str] | None = None,
        include_targets: bool = False,
        unpack: bool = False,
    ) -> pl.DataFrame:
        """
        Get the attributes of the edges as a polars DataFrame.

        Parameters
        ----------
        node_ids : list[int] | None
            The IDs of the subgraph to get the edge attributesfor.
            If None, all edges of the graph are used.
        attr_keys : Sequence[str] | None
            The attribute keys to get.
            If None, all attributesare used.
        include_targets : bool
            Whether to include edges out-going from the given node_ids even
            if the target node is not in the given node_ids.
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
        [load_ctc][tracksdata.io._ctc.load_ctc]:
            Load a CTC ground truth file into a graph.

        [RegionPropsNodes][tracksdata.nodes.RegionPropsNodes]:
            Operator to create nodes from label images.

        Returns
        -------
        BaseGraph
            A graph with the nodes and edges from the CTC data directory.
        """
        from tracksdata.io._ctc import load_ctc

        graph = cls(**kwargs)
        load_ctc(data_dir, graph)
        return graph

    @classmethod
    def from_array(
        cls: type[T],
        positions: ArrayLike,
        track_ids: ArrayLike | None = None,
        track_id_graph: dict[int, int] | None = None,
        radius: ArrayLike = 1,
        image_shape: tuple[int, ...] | None = None,
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
        radius : ArrayLike
            Integer or N-dimensional array of radii.
        image_shape : tuple[int, ...] | None
            Shape of the image if available masks are cropped to fit the image.
        **kwargs : Any
            Additional arguments to pass to the graph constructor.

        Returns
        -------
        BaseGraph
            A graph with the nodes and edges from the numpy array.
        """
        from tracksdata.io._numpy_array import load_array

        graph = cls(**kwargs)
        load_array(
            positions=positions,
            graph=graph,
            track_ids=track_ids,
            track_id_graph=track_id_graph,
            radius=radius,
            image_shape=image_shape,
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
            node_ids = self.filter_nodes_by_attrs(NodeAttr(DEFAULT_ATTR_KEYS.T) == t)
            masks = self.node_attrs(node_ids=node_ids, attr_keys=[DEFAULT_ATTR_KEYS.MASK])[DEFAULT_ATTR_KEYS.MASK]
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

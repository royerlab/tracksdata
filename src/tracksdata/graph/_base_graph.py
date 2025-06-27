import abc
import functools
import operator
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, overload

import polars as pl

from tracksdata.attrs import AttrComparison
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.utils._logging import LOG

if TYPE_CHECKING:
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
    ) -> None:
        """
        Faster method to add multiple nodes to the graph with less overhead and fewer checks.

        Parameters
        ----------
        nodes : list[dict[str, Any]]
            The data of the nodes to be added.
            The keys of the data will be used as the attributes of the nodes.
            Must have "t" key.
        """
        # this method benefits the SQLGraph backend
        for node in nodes:
            self.add_node(node, validate_keys=False)

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

    def bulk_add_edges(
        self,
        edges: list[dict[str, Any]],
    ) -> None:
        """
        Faster method to add multiple edges to the graph with less overhead and fewer checks.

        Parameters
        ----------
        edges : list[dict[str, Any]]
            The data of the edges to be added.
            The keys of the data will be used as the attributes of the edges.
            Must have "source_id" and "target_id" keys.
            For example:
            ```python
            graph.bulk_add_edges([dict(source_id=0, target_id=1, weight=1.0)])
            ```
        """
        # this method benefits the SQLGraph backend
        for edge in edges:
            self.add_edge(
                edge.pop("source_id"),
                edge.pop("target_id"),
                edge,
                validate_keys=False,
            )

    @abc.abstractmethod
    def sucessors(
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
            The IDs of the nodes to get the attributesfor.
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
        tracksdata.io._ctc.load_ctc : Load a CTC ground truth file into a graph.
        RegionPropsNodes : Operator to create nodes from label images.

        Returns
        -------
        BaseGraph
            A graph with the nodes and edges from the CTC data directory.
        """
        from tracksdata.io._ctc import load_ctc

        graph = cls(**kwargs)
        load_ctc(data_dir, graph)
        return graph

    @overload
    def in_degree(self, node_ids: int) -> int: ...

    @overload
    def in_degree(self, node_ids: list[int] | None = None) -> list[int]: ...

    @overload
    def out_degree(self, node_ids: int) -> int: ...

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

        other_to_node_ids = dict(zip(other_ids, node_ids, strict=False))

        self_edges_df = self.edge_attrs(attr_keys=[])
        other_edges_df = other.edge_attrs(attr_keys=[])

        other_edges_df = other_edges_df.with_columns(
            *{
                col: other_edges_df[col].map_elements(other_to_node_ids.get, return_dtype=pl.Int64).alias(col)
                for col in [DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]
            }
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

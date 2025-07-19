from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import rustworkx as rx
import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, Session, aliased, load_only
from sqlalchemy.sql.type_api import TypeEngine

from tracksdata.attrs import AttrComparison, split_attr_comps
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_filter import BaseFilter, cache_method
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.utils._dataframe import unpack_array_attrs, unpickle_bytes_columns
from tracksdata.utils._logging import LOG

if TYPE_CHECKING:
    from tracksdata.graph._graph_view import GraphView


def _is_builtin(obj: Any) -> bool:
    """Check if an object is a built-in type."""
    return getattr(obj.__class__, "__module__", None) == "builtins"


def _data_numpy_to_native(data: dict[str, Any]) -> None:
    """
    Convert numpy scalars to native Python scalars in place.

    Parameters
    ----------
    data : dict[str, Any]
        The data to convert. Modified in place.
    """
    for k, v in data.items():
        if np.isscalar(v) and hasattr(v, "item"):
            data[k] = v.item()


def _filter_query(
    query: sa.Select,
    table: type[DeclarativeBase],
    attr_filters: list[AttrComparison],
) -> sa.Select:
    """
    Filter a query by a list of attribute filters.

    Parameters
    ----------
    query : sa.Select
        The query to filter.
    table : type[DeclarativeBase]
        The table to filter.
    attr_filters : list[AttrComparison]
        The attribute filters to apply.

    Returns
    -------
    sa.Select
        The filtered query.
    """
    LOG.info("Filter query:\n%s", attr_filters)
    query = query.filter(
        *[attr_filter.op(getattr(table, str(attr_filter.column)), attr_filter.other) for attr_filter in attr_filters]
    )
    return query


class SQLFilter(BaseFilter):
    def __init__(
        self,
        graph: "SQLGraph",
        *attr_filters: AttrComparison,
        include_targets: bool = False,
        include_sources: bool = False,
    ):
        super().__init__()
        self._graph = graph
        self._node_attr_comps, self._edge_attr_comps = split_attr_comps(attr_filters)
        self._include_targets = include_targets
        self._include_sources = include_sources

        # creating initial query
        self._node_query: sa.Select = sa.select(self._graph.Node)
        self._edge_query: sa.Select = sa.select(self._graph.Edge)

        if self._node_attr_comps:
            # filtering nodes by attributes
            self._node_query = _filter_query(self._node_query, self._graph.Node, self._node_attr_comps)

            # if both node and edge attributes are filtered
            # we need to select subset of edges that belong to the filtered nodes
            SourceNode = aliased(self._graph.Node)
            TargetNode = aliased(self._graph.Node)

            # if the user specified include_targets or include_sources
            # we do not filter the edges to include only the selected nodes
            include_none = not self._include_targets and not self._include_sources

            if self._include_targets or include_none:
                self._edge_query = self._edge_query.join(
                    SourceNode,
                    self._graph.Edge.source_id == SourceNode.node_id,
                )
                self._edge_query = _filter_query(self._edge_query, SourceNode, self._node_attr_comps)

            if self._include_sources or include_none:
                self._edge_query = self._edge_query.join(
                    TargetNode,
                    self._graph.Edge.target_id == TargetNode.node_id,
                )
                self._edge_query = _filter_query(self._edge_query, TargetNode, self._node_attr_comps)

        if self._edge_attr_comps:
            self._edge_query = _filter_query(self._edge_query, self._graph.Edge, self._edge_attr_comps)

            # we haven't filtered the nodes by attributes
            # so we only return the nodes that are in the edges
            if not self._node_attr_comps:
                self._node_query = self._node_query.filter(
                    sa.or_(
                        self._graph.Node.node_id.in_(sa.select(self._edge_query.subquery().c.source_id)),
                        self._graph.Node.node_id.in_(sa.select(self._edge_query.subquery().c.target_id)),
                    )
                )

        if self._include_targets or self._include_sources:
            nodes_query = [self._node_query]

            edge_subq = self._edge_query.subquery()

            subq_ids = []
            if self._include_targets:
                subq_ids.append(edge_subq.c.target_id)
            if self._include_sources:
                subq_ids.append(edge_subq.c.source_id)

            for subq_id in subq_ids:
                nodes_query.append(
                    sa.select(self._graph.Node).join(
                        edge_subq,
                        self._graph.Node.node_id == subq_id,
                    )
                )

            self._node_query = sa.union(*nodes_query)

    def __enter__(self) -> "SQLFilter":
        return self

    @cache_method
    def node_ids(self) -> list[int]:
        """
        Get the ids of the nodes resulting from the filter.
        """
        with Session(self._graph._engine) as session:
            return session.execute(self._node_query.with_only_columns(self._graph.Node.node_id)).scalars().all()

    @cache_method
    def edge_ids(self) -> list[int]:
        """
        Get the ids of the edges resulting from the filter.
        """
        with Session(self._graph._engine) as session:
            return session.execute(self._edge_query.with_only_columns(self._graph.Edge.edge_id)).scalars().all()

    @cache_method
    def node_attrs(
        self,
        *,
        attr_keys: list[str] | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        query = self._query_from_attr_keys(
            query=self._node_query,
            table=self._graph.Node,
            attr_keys=attr_keys,
        )

        with Session(self._graph._engine) as session:
            nodes_attrs = pl.read_database(
                self._graph._raw_query(query),
                connection=session.connection(),
            )

        if attr_keys is not None:
            nodes_attrs = nodes_attrs.select(attr_keys)

        nodes_attrs = self._graph._cast_boolean_columns(self._graph.Node, nodes_attrs)
        nodes_attrs = unpickle_bytes_columns(nodes_attrs)

        if unpack:
            nodes_attrs = unpack_array_attrs(nodes_attrs)

        return nodes_attrs

    @staticmethod
    def _query_from_attr_keys(
        query: sa.Select,
        table: type[DeclarativeBase],
        attr_keys: list[str] | None = None,
        extra_columns: list[str] | None = None,
    ) -> sa.Select:
        if attr_keys is not None:
            attr_keys = list(dict.fromkeys(attr_keys))

            if extra_columns is not None:
                attr_keys.extend(extra_columns)

            LOG.info("Query attr_keys: %s", attr_keys)

            if isinstance(query, sa.CompoundSelect):
                union_query = query.alias("u")
                query = sa.select(
                    *[getattr(union_query.c, key) for key in attr_keys],
                )
            else:
                query = query.with_only_columns(
                    *[getattr(table, key) for key in attr_keys],
                )

        LOG.info("Query after attr_keys selection:\n%s", query)

        return query

    @cache_method
    def edge_attrs(self, attr_keys: list[str] | None = None, unpack: bool = False) -> pl.DataFrame:
        query = self._query_from_attr_keys(
            query=self._edge_query,
            table=self._graph.Edge,
            attr_keys=attr_keys,
            extra_columns=[
                DEFAULT_ATTR_KEYS.EDGE_ID,
                DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                DEFAULT_ATTR_KEYS.EDGE_TARGET,
            ],
        )

        with Session(self._graph._engine) as session:
            edges_df = pl.read_database(
                self._graph._raw_query(query),
                connection=session.connection(),
            )

        edges_df = self._graph._cast_boolean_columns(self._graph.Edge, edges_df)
        edges_df = unpickle_bytes_columns(edges_df)

        if unpack:
            edges_df = unpack_array_attrs(edges_df)

        return edges_df

    @cache_method
    def subgraph(
        self,
        node_attr_keys: Sequence[str] | str | None = None,
        edge_attr_keys: Sequence[str] | str | None = None,
    ) -> "GraphView":
        from tracksdata.graph._graph_view import GraphView

        node_query = self._query_from_attr_keys(
            query=self._node_query,
            table=self._graph.Node,
            attr_keys=node_attr_keys,
        )

        edge_query = self._query_from_attr_keys(
            query=self._edge_query,
            table=self._graph.Edge,
            attr_keys=edge_attr_keys,
            extra_columns=[
                DEFAULT_ATTR_KEYS.EDGE_ID,
                DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                DEFAULT_ATTR_KEYS.EDGE_TARGET,
            ],
        )

        with Session(self._graph._engine) as session:
            node_query = session.execute(node_query)
            edge_query = session.execute(edge_query)

            node_map_to_root = {}
            node_map_from_root = {}
            rx_graph = rx.PyDiGraph()

            for row in node_query.scalars().all():
                data = {k: v for k, v in row.__dict__.items() if not k.startswith("_")}
                root_node_id = data.pop(DEFAULT_ATTR_KEYS.NODE_ID)
                node_id = rx_graph.add_node(data)
                node_map_to_root[node_id] = root_node_id
                node_map_from_root[root_node_id] = node_id

            for row in edge_query.scalars().all():
                data = {k: v for k, v in row.__dict__.items() if not k.startswith("_")}
                source_id = node_map_from_root[data.pop(DEFAULT_ATTR_KEYS.EDGE_SOURCE)]
                target_id = node_map_from_root[data.pop(DEFAULT_ATTR_KEYS.EDGE_TARGET)]
                rx_graph.add_edge(source_id, target_id, data)

        graph = GraphView(
            rx_graph=rx_graph,
            node_map_to_root=node_map_to_root,
            root=self._graph,
        )

        return graph


class SQLGraph(BaseGraph):
    """
    SQL-based graph implementation using SQLAlchemy ORM.

    Provides persistent storage and efficient querying of large graphs with
    support for dynamic schema modification and various database backends.
    Node IDs are automatically generated based on time to ensure uniqueness
    across time points.

    Parameters
    ----------
    drivername : str
        The database driver name (e.g., 'sqlite', 'postgresql', 'mysql').
    database : str
        The database name or path. For SQLite, this is the file path.
    username : str, optional
        Database username. Not required for SQLite.
    password : str, optional
        Database password. Not required for SQLite.
    host : str, optional
        Database host. Not required for SQLite.
    port : int, optional
        Database port. Not required for SQLite.
    overwrite : bool, default False
        If True, drop and recreate all tables. Use with caution as this
        will delete all existing data.

    Attributes
    ----------
    node_id_time_multiplier : int
        Multiplier used to generate node IDs based on time (default: 1,000,000,000).
    Base : type[DeclarativeBase]
        SQLAlchemy declarative base class for this graph instance.
    Node : type[DeclarativeBase]
        SQLAlchemy model class for nodes.
    Edge : type[DeclarativeBase]
        SQLAlchemy model class for edges.

    See Also
    --------
    [RustWorkXGraph][tracksdata.graph.RustWorkXGraph]:
        In memory Rustworkx-based graph implementation.

    Examples
    --------
    Create an in-memory SQLite graph:

    ```python
    graph = SQLGraph("sqlite", ":memory:")
    ```

    Create a persistent SQLite graph:

    ```python
    graph = SQLGraph("sqlite", "my_graph.db")
    ```

    Create a PostgreSQL graph:

    ```python
    graph = SQLGraph("postgresql", "tracking_db", username="user", password="pass", host="localhost", port=5432)
    ```

    Add nodes and edges:

    ```python
    node_id = graph.add_node({"t": 0, "x": 10.5, "y": 20.3})
    edge_id = graph.add_edge(node_id, target_id, {"weight": 0.8})
    ```
    """

    node_id_time_multiplier: int = 1_000_000_000
    Base: type[DeclarativeBase]
    Node: type[DeclarativeBase]
    Edge: type[DeclarativeBase]

    def __init__(
        self,
        drivername: str,
        database: str,
        username: str | None = None,
        password: str | None = None,
        host: str | None = None,
        port: int | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        overwrite: bool = False,
    ):
        self._url = sa.engine.URL.create(
            drivername,
            username=username,
            password=password,
            host=host,
            port=port,
            database=database,
        )
        self._engine_kwargs = engine_kwargs if engine_kwargs is not None else {}
        self._engine = sa.create_engine(self._url, **self._engine_kwargs)

        # Create unique classes for this instance
        self._define_schema(overwrite=overwrite)
        self._boolean_columns: dict[str, list[str]] = {self.Node.__tablename__: [], self.Edge.__tablename__: []}

        if overwrite:
            self.Base.metadata.drop_all(self._engine)

        self.Base.metadata.create_all(self._engine)

        self._max_id_per_time = {}
        self._update_max_id_per_time()

    def _define_schema(self, overwrite: bool) -> None:
        """
        Define the database schema classes for this SQLGraph instance.

        Creates unique SQLAlchemy model classes for this graph instance to
        avoid conflicts between multiple SQLGraph instances.
        """
        metadata = sa.MetaData()
        metadata.reflect(bind=self._engine)

        class Base(DeclarativeBase):
            pass

        if len(metadata.tables) > 0 and not overwrite:
            for table_name, table in metadata.tables.items():
                cls = type(
                    table_name,
                    (Base,),
                    {
                        "__table__": table,
                        "__tablename__": table_name,
                    },
                )
                setattr(self, table_name, cls)
            self.Base = Base
            return

        class Node(Base):
            __tablename__ = "Node"

            # Use node_id as sole primary key for simpler updates
            node_id = sa.Column(sa.BigInteger, primary_key=True, unique=True)

            # Add t as a regular column
            # NOTE might want to use as index for fast time-based queries
            t = sa.Column(sa.Integer, nullable=False)

        node_tb_name = Node.__tablename__

        class Edge(Base):
            __tablename__ = "Edge"
            edge_id = sa.Column(sa.Integer, primary_key=True, unique=True, autoincrement=True)
            source_id = sa.Column(sa.BigInteger, sa.ForeignKey(f"{node_tb_name}.node_id"))
            target_id = sa.Column(sa.BigInteger, sa.ForeignKey(f"{node_tb_name}.node_id"))

        class Overlap(Base):
            __tablename__ = "Overlap"
            overlap_id = sa.Column(sa.Integer, primary_key=True, unique=True, autoincrement=True)
            source_id = sa.Column(sa.BigInteger, sa.ForeignKey(f"{node_tb_name}.node_id"))
            target_id = sa.Column(sa.BigInteger, sa.ForeignKey(f"{node_tb_name}.node_id"))

        # Assign to instance variables
        self.Base = Base
        self.Node = Node
        self.Edge = Edge
        self.Overlap = Overlap

    def _cast_boolean_columns(self, table_class: type[DeclarativeBase], df: pl.DataFrame) -> pl.DataFrame:
        """
        This is required because polars bypasses the boolean type and converts it to integer.
        """
        df = df.with_columns(
            pl.col(col).cast(pl.Boolean)
            for col in self._boolean_columns[table_class.__tablename__]
            if col in df.columns
        )
        return df

    def _update_max_id_per_time(self) -> None:
        """
        Update the maximum node ID for each time point.

        Scans the database to find the current maximum node ID for each time
        point and updates the internal cache to ensure newly created nodes
        have unique IDs.
        """
        with Session(self._engine) as session:
            for t in session.query(self.Node.t).distinct():
                self._max_id_per_time[t] = int(session.query(sa.func.max(self.Node.node_id)).scalar())

    def filter(
        self,
        *attr_filters: AttrComparison,
        include_targets: bool = False,
        include_sources: bool = False,
    ) -> "SQLFilter":
        return SQLFilter(
            self,
            *attr_filters,
            include_targets=include_targets,
            include_sources=include_sources,
        )

    def add_node(
        self,
        attrs: dict[str, Any],
        validate_keys: bool = True,
    ) -> int:
        """
        Add a node to the graph at time t.

        Node IDs are automatically generated based on the time point and
        the node_id_time_multiplier to ensure uniqueness across time points.

        Parameters
        ----------
        attrs : dict[str, Any]
            The attributes of the node to be added. Must contain a "t" key
            specifying the time point. Additional keys will be stored as
            node attributes.
        validate_keys : bool, default True
            Whether to check if the attribute keys are valid against the
            current schema. If False, validation is skipped for performance.

        Returns
        -------
        int
            The ID of the newly added node.

        Raises
        ------
        ValueError
            If validate_keys is True and the attributes contain invalid keys,
            or if the "t" key is missing.

        Examples
        --------
        ```python
        node_id = graph.add_node({"t": 0, "x": 10.5, "y": 20.3})
        node_id = graph.add_node({"t": 1, "x": 15.2, "y": 25.8, "intensity": 150.0})
        ```
        """
        if validate_keys:
            self._validate_attributes(attrs, self.node_attr_keys, "node")

            if "t" not in attrs:
                raise ValueError(f"Node attributes must have a 't' key. Got {attrs.keys()}")

        time = attrs["t"]
        default_node_id = (time * self.node_id_time_multiplier) - 1
        node_id = self._max_id_per_time.get(time, default_node_id) + 1

        node = self.Node(
            node_id=node_id,
            **attrs,
        )

        with Session(self._engine) as session:
            session.add(node)
            session.commit()

        self._max_id_per_time[time] = node_id

        return node_id

    def bulk_add_nodes(
        self,
        nodes: list[dict[str, Any]],
    ) -> None:
        """
        Add multiple nodes to the graph efficiently.

        Provides better performance than calling add_node multiple times by
        using SQLAlchemy's bulk insert functionality and reducing database
        transactions. Automatically assigns node IDs and handles time-based
        ID generation.

        Parameters
        ----------
        nodes : list[dict[str, Any]]
            List of node attribute dictionaries. Each dictionary must contain
            a "t" key and any additional attribute keys.

        Examples
        --------
        ```python
        nodes = [
            {"t": 0, "x": 10, "y": 20, "label": "A"},
            {"t": 0, "x": 15, "y": 25, "label": "B"},
        ]
        graph.bulk_add_nodes(nodes)
        ```

        Returns
        -------
        list[int]
            The IDs of the added nodes.
        """
        if len(nodes) == 0:
            return []

        node_ids = []
        for node in nodes:
            time = node["t"]
            default_node_id = (time * self.node_id_time_multiplier) - 1
            node_id = self._max_id_per_time.get(time, default_node_id) + 1
            node[DEFAULT_ATTR_KEYS.NODE_ID] = node_id
            node_ids.append(node_id)
            self._max_id_per_time[time] = node_id

        with Session(self._engine) as session:
            session.execute(sa.insert(self.Node), nodes)
            session.commit()

        return node_ids

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
            Additional attributes for the edge (e.g., weight, distance).
        validate_keys : bool, default True
            Whether to check if the attribute keys are valid against the
            current schema. If False, validation is skipped for performance.

        Returns
        -------
        int
            The ID of the newly added edge.

        Raises
        ------
        ValueError
            If validate_keys is True and the attributes contain invalid keys.

        Examples
        --------
        ```python
        edge_id = graph.add_edge(node1_id, node2_id, {"weight": 0.8})
        edge_id = graph.add_edge(node1_id, node2_id, {"weight": 0.9, "distance": 5.2, "confidence": 0.95})
        ```
        """
        if validate_keys:
            self._validate_attributes(attrs, self.edge_attr_keys, "edge")

        if hasattr(source_id, "item"):
            source_id = source_id.item()

        if hasattr(target_id, "item"):
            target_id = target_id.item()

        edge = self.Edge(
            source_id=source_id,
            target_id=target_id,
            **attrs,
        )

        with Session(self._engine) as session:
            session.add(edge)
            session.commit()
            edge_id = edge.edge_id

        return edge_id

    def bulk_add_edges(
        self,
        edges: list[dict[str, Any]],
        return_ids: bool = False,
    ) -> list[int] | None:
        """
        Add multiple edges to the graph efficiently.

        Provides better performance than calling add_edge multiple times by
        using SQLAlchemy's bulk insert functionality.

        Parameters
        ----------
        edges : list[dict[str, Any]]
            List of edge attribute dictionaries. Each dictionary must contain
            "source_id" and "target_id" keys, plus any additional feature keys.
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

        Return
        ------
        list[int] | None
            The IDs of the added edges.
        """
        if len(edges) == 0:
            if return_ids:
                return []
            return None

        for edge in edges:
            _data_numpy_to_native(edge)

        with Session(self._engine) as session:
            if return_ids:
                result = session.execute(sa.insert(self.Edge).returning(self.Edge.edge_id), edges)
                session.commit()
                return list(result.scalars().all())
            else:
                session.execute(sa.insert(self.Edge), edges)
                session.commit()

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
        overlap = self.Overlap(
            source_id=source_id,
            target_id=target_id,
        )
        with Session(self._engine) as session:
            session.add(overlap)
            session.commit()

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
        [add_overlap][tracksdata.graph.SQLGraph.add_overlap]:
            Add a single overlap to the graph.
        """
        if hasattr(overlaps, "tolist"):
            overlaps = overlaps.tolist()

        overlaps = [{"source_id": source_id, "target_id": target_id} for source_id, target_id in overlaps]
        with Session(self._engine) as session:
            session.execute(sa.insert(self.Overlap), overlaps)
            session.commit()

    def overlaps(
        self,
        node_ids: list[int] | None = None,
    ) -> list[list[int, 2]]:
        """
        Get the overlaps between the nodes in `node_ids`.
        """
        if hasattr(node_ids, "tolist"):
            node_ids = node_ids.tolist()

        with Session(self._engine) as session:
            query = session.query(self.Overlap.source_id, self.Overlap.target_id)

            if node_ids is not None:
                query = query.filter(
                    self.Overlap.source_id.in_(node_ids),
                    self.Overlap.target_id.in_(node_ids),
                )

            return [[source_id, target_id] for source_id, target_id in query.all()]

    def has_overlaps(self) -> bool:
        """
        Check if the graph has any overlaps.
        """
        with Session(self._engine) as session:
            return session.query(self.Overlap).count() > 0

    def _get_neighbors(
        self,
        node_key: str,
        neighbor_key: str,
        node_ids: list[int] | int,
        attr_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        """
        Get the predecessors or successors of nodes via database joins.

        Helper method used by sucessors() and predecessors() to efficiently
        query neighbor relationships through SQL joins.

        Parameters
        ----------
        node_key : str
            The edge attribute key for the query node (e.g., "source_id").
        neighbor_key : str
            The edge attribute key for the neighbor node (e.g., "target_id").
        node_ids : list[int] | int
            The IDs of the nodes to get neighbors for.
        attr_keys : Sequence[str] | str | None, optional
            The attribute keys to retrieve for neighbor nodes. If None,
            all attributes are retrieved.

        Returns
        -------
        dict[int, pl.DataFrame] | pl.DataFrame
            If multiple node_ids are provided, returns a dictionary mapping
            each node_id to a DataFrame of its neighbors. If a single node_id
            is provided, returns the DataFrame directly.
        """
        single_node = False
        if isinstance(node_ids, int):
            node_ids = [node_ids]
            single_node = True

        if isinstance(attr_keys, str):
            attr_keys = [attr_keys]

        with Session(self._engine) as session:
            if attr_keys is None:
                # all columns
                node_columns = [self.Node]
            else:
                node_columns = [getattr(self.Node, key) for key in attr_keys]

            query = session.query(getattr(self.Edge, node_key), *node_columns)
            query = query.join(self.Edge, getattr(self.Edge, neighbor_key) == self.Node.node_id)
            query = query.filter(getattr(self.Edge, node_key).in_(node_ids))

            node_df = pl.read_database(
                query.statement,
                connection=session.connection(),
            )
            node_df = self._cast_boolean_columns(self.Node, node_df)
            node_df = unpickle_bytes_columns(node_df)

        if single_node:
            return node_df

        neighbors_dict = {node_id: group for (node_id,), group in node_df.group_by(node_key)}
        for node_id in node_ids:
            if node_id not in neighbors_dict:
                neighbors_dict[node_id] = pl.DataFrame(schema=node_df.schema)

        return neighbors_dict

    def successors(
        self,
        node_ids: list[int] | int,
        attr_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        """
        Get the successor nodes of given nodes.

        Successors are nodes that are targets of edges originating from the
        given nodes (outgoing edges). Uses efficient SQL joins to retrieve
        successor information with their attributes in a single query.

        Parameters
        ----------
        node_ids : list[int] | int
            The IDs of the nodes to get successors for.
        attr_keys : Sequence[str] | str | None, optional
            The attribute keys to retrieve for successor nodes. If None,
            all attributes are retrieved.

        Returns
        -------
        dict[int, pl.DataFrame] | pl.DataFrame
            If a list of node_ids is provided, returns a dictionary mapping
            each node_id to a DataFrame of its successors. If a single node_id
            is provided, returns the DataFrame directly.

        Examples
        --------
        ```python
        successors_df = graph.sucessors(node_id)
        successors_dict = graph.sucessors([node1, node2, node3])
        ```
        """
        return self._get_neighbors(
            node_key=DEFAULT_ATTR_KEYS.EDGE_SOURCE,
            neighbor_key=DEFAULT_ATTR_KEYS.EDGE_TARGET,
            node_ids=node_ids,
            attr_keys=attr_keys,
        )

    def predecessors(
        self,
        node_ids: list[int] | int,
        attr_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        """
        Get the predecessor nodes of given nodes.

        Predecessors are nodes that are sources of edges targeting the
        given nodes (incoming edges). Uses efficient SQL joins to retrieve
        predecessor information with their attributes in a single query.

        Parameters
        ----------
        node_ids : list[int] | int
            The IDs of the nodes to get predecessors for.
        attr_keys : Sequence[str] | str | None, optional
            The attribute keys to retrieve for predecessor nodes. If None,
            all attributes are retrieved.

        Returns
        -------
        dict[int, pl.DataFrame] | pl.DataFrame
            If a list of node_ids is provided, returns a dictionary mapping
            each node_id to a DataFrame of its predecessors. If a single node_id
            is provided, returns the DataFrame directly.

        Examples
        --------
        ```python
        predecessors_df = graph.predecessors(node_id)
        predecessors_dict = graph.predecessors([node1, node2, node3])
        ```
        """
        return self._get_neighbors(
            node_key=DEFAULT_ATTR_KEYS.EDGE_TARGET,
            neighbor_key=DEFAULT_ATTR_KEYS.EDGE_SOURCE,
            node_ids=node_ids,
            attr_keys=attr_keys,
        )

    def filter_nodes_by_attrs(
        self,
        *attrs: AttrComparison,
    ) -> list[int]:
        """
        Filter nodes by their attribute values.

        Performs an SQL query with WHERE clauses for each specified attribute,
        providing efficient filtering for large graphs.

        Parameters
        ----------
        attrs : dict[str, Any]
            Dictionary of attribute-value pairs to filter by. Only nodes
            that match all specified attributes will be returned.

        Returns
        -------
        list[int]
            List of node IDs that match the filter criteria.

        Examples
        --------
        ```python
        node_ids = graph.filter_nodes_by_attribute({"t": 0})
        node_ids = graph.filter_nodes_by_attribute({"t": 1, "label": "A"})
        ```
        """
        with Session(self._engine) as session:
            query = session.query(self.Node.node_id)
            query = _filter_query(query, self.Node, attrs)
            return [i for (i,) in query.all()]

    def node_ids(self) -> list[int]:
        """
        Get the IDs of all nodes in the graph.

        Returns
        -------
        list[int]
            List of all node IDs in the graph.

        Examples
        --------
        ```python
        all_nodes = graph.node_ids()
        print(f"Graph contains {len(all_nodes)} nodes")
        ```
        """
        with Session(self._engine) as session:
            return [i for (i,) in session.query(self.Node.node_id).all()]

    def edge_ids(self) -> list[int]:
        with Session(self._engine) as session:
            return [i for (i,) in session.query(self.Edge.edge_id).all()]

    def subgraph(
        self,
        *attr_filters: AttrComparison,
        node_ids: Sequence[int] | None = None,
        node_attr_keys: Sequence[str] | str | None = None,
        edge_attr_keys: Sequence[str] | str | None = None,
    ) -> "GraphView":
        from tracksdata.graph._graph_view import GraphView

        node_attr_comps, edge_attr_comps = split_attr_comps(attr_filters)
        self._validate_subgraph_args(node_ids, node_attr_comps, edge_attr_comps)

        if hasattr(node_ids, "tolist"):
            node_ids = node_ids.tolist()

        with Session(self._engine) as session:
            node_query = sa.select(self.Node)
            edge_query = sa.select(self.Edge)

            node_filtered = False

            if node_ids is not None:
                node_query = node_query.filter(self.Node.node_id.in_(node_ids))
                node_filtered = True

                edge_query = edge_query.filter(
                    self.Edge.source_id.in_(node_ids),
                    self.Edge.target_id.in_(node_ids),
                )

            if node_attr_comps:
                node_query = _filter_query(node_query, self.Node, node_attr_comps)

                node_ids = list(session.execute(node_query.with_only_columns(self.Node.node_id)).scalars().all())
                node_filtered = True

                SourceNode = aliased(self.Node)
                TargetNode = aliased(self.Node)

                edge_query = edge_query.join(
                    SourceNode,
                    self.Edge.source_id == SourceNode.node_id,
                ).join(
                    TargetNode,
                    self.Edge.target_id == TargetNode.node_id,
                )
                edge_query = _filter_query(edge_query, SourceNode, node_attr_comps)
                edge_query = _filter_query(edge_query, TargetNode, node_attr_comps)

            if edge_attr_comps:
                edge_query = _filter_query(edge_query, self.Edge, edge_attr_comps)

                if not node_filtered:
                    node_ids = (
                        session.execute(edge_query.with_only_columns(self.Edge.source_id, self.Edge.target_id))
                        .scalars()
                        .all()
                    )
                    node_ids = np.unique(node_ids).tolist()
                    node_query = node_query.filter(self.Node.node_id.in_(node_ids))

            if node_attr_keys is not None:
                if DEFAULT_ATTR_KEYS.NODE_ID not in node_attr_keys:
                    node_attr_keys.append(DEFAULT_ATTR_KEYS.NODE_ID)

                node_query = node_query.options(
                    load_only(
                        *[getattr(self.Node, key) for key in node_attr_keys],
                    ),
                )

            if edge_attr_keys is not None:
                edge_attr_keys = set(edge_attr_keys)
                # we always return the source and target id by default
                edge_attr_keys.add(DEFAULT_ATTR_KEYS.EDGE_ID)
                edge_attr_keys.add(DEFAULT_ATTR_KEYS.EDGE_SOURCE)
                edge_attr_keys.add(DEFAULT_ATTR_KEYS.EDGE_TARGET)
                edge_attr_keys = list(edge_attr_keys)

                edge_query = edge_query.options(
                    load_only(
                        *[getattr(self.Edge, key) for key in edge_attr_keys],
                    ),
                )

        node_map_to_root = {}
        node_map_from_root = {}
        rx_graph = rx.PyDiGraph()

        node_query = session.execute(node_query)
        edge_query = session.execute(edge_query)

        for row in node_query.scalars().all():
            data = {k: v for k, v in row.__dict__.items() if not k.startswith("_")}
            root_node_id = data.pop(DEFAULT_ATTR_KEYS.NODE_ID)
            node_id = rx_graph.add_node(data)
            node_map_to_root[node_id] = root_node_id
            node_map_from_root[root_node_id] = node_id

        for row in edge_query.scalars().all():
            data = {k: v for k, v in row.__dict__.items() if not k.startswith("_")}
            source_id = node_map_from_root[data.pop(DEFAULT_ATTR_KEYS.EDGE_SOURCE)]
            target_id = node_map_from_root[data.pop(DEFAULT_ATTR_KEYS.EDGE_TARGET)]
            rx_graph.add_edge(source_id, target_id, data)

        graph = GraphView(
            rx_graph=rx_graph,
            node_map_to_root=node_map_to_root,
            root=self,
        )

        return graph

    def time_points(self) -> list[int]:
        with Session(self._engine) as session:
            return [t for (t,) in session.query(self.Node.t).distinct().all()]

    def _reorder_by_indices(
        self,
        df: pl.DataFrame,
        indices: Sequence[int],
        id_key: str,
    ) -> pl.DataFrame:
        # NOTE: maybe we should avoid doing this and return the indices
        order_df = pl.DataFrame(
            {id_key: indices, "order": np.arange(0, len(indices))},
        )
        return order_df.join(df, on=id_key, how="left").drop("order")

    def _raw_query(self, query: sa.Select) -> str:
        # for some reason, the query.statement is not working with polars
        raw_query = str(query.compile(dialect=self._engine.dialect, compile_kwargs={"literal_binds": True}))
        LOG.info("Raw query:\n%s", raw_query)
        return raw_query

    def node_attrs(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        attr_keys: Sequence[str] | str | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        if isinstance(attr_keys, str):
            attr_keys = [attr_keys]

        with Session(self._engine) as session:
            query = session.query(self.Node)

            if node_ids is not None:
                if hasattr(node_ids, "tolist"):
                    node_ids = node_ids.tolist()

                query = query.filter(self.Node.node_id.in_(node_ids))

            if attr_keys is not None:
                # making them unique
                attr_keys = list(dict.fromkeys(attr_keys))

                query = query.options(
                    load_only(
                        *[getattr(self.Node, key) for key in attr_keys],
                    ),
                )

            LOG.info("Query: %s", query.statement)

        nodes_df = pl.read_database(
            query.statement,
            connection=session.connection(),
        )
        nodes_df = self._cast_boolean_columns(self.Node, nodes_df)
        nodes_df = unpickle_bytes_columns(nodes_df)

        # match node_ids ordering
        if node_ids is not None and not nodes_df.is_empty():
            nodes_df = self._reorder_by_indices(nodes_df, node_ids, "node_id")

        # indices are included by default and must be removed
        if attr_keys is not None:
            nodes_df = nodes_df.select([pl.col(c) for c in attr_keys])

        if unpack:
            nodes_df = unpack_array_attrs(nodes_df)

        return nodes_df

    def edge_attrs(
        self,
        *,
        node_ids: list[int] | None = None,
        attr_keys: Sequence[str] | None = None,
        include_targets: bool = False,
        unpack: bool = False,
    ) -> pl.DataFrame:
        if isinstance(attr_keys, str):
            attr_keys = [attr_keys]

        with Session(self._engine) as session:
            query = sa.select(self.Edge)

            if node_ids is not None:
                if hasattr(node_ids, "tolist"):
                    node_ids = node_ids.tolist()

                if include_targets:
                    query = query.filter(self.Edge.source_id.in_(node_ids))
                else:
                    query = query.filter(
                        self.Edge.source_id.in_(node_ids),
                        self.Edge.target_id.in_(node_ids),
                    )

            if attr_keys is not None:
                attr_keys = set(attr_keys)
                # we always return the source and target id by default
                attr_keys.add(DEFAULT_ATTR_KEYS.EDGE_ID)
                attr_keys.add(DEFAULT_ATTR_KEYS.EDGE_SOURCE)
                attr_keys.add(DEFAULT_ATTR_KEYS.EDGE_TARGET)
                attr_keys = list(attr_keys)

                LOG.info("Edge attribute keys: %s", attr_keys)

                query = query.with_only_columns(
                    *[getattr(self.Edge, key) for key in attr_keys],
                )

            edges_df = pl.read_database(
                self._raw_query(query),
                connection=session.connection(),
            )
            edges_df = self._cast_boolean_columns(self.Edge, edges_df)
            edges_df = unpickle_bytes_columns(edges_df)

        if unpack:
            edges_df = unpack_array_attrs(edges_df)

        return edges_df

    @property
    def node_attr_keys(self) -> list[str]:
        keys = list(self.Node.__table__.columns.keys())
        keys.remove(DEFAULT_ATTR_KEYS.NODE_ID)
        return keys

    @property
    def edge_attr_keys(self) -> list[str]:
        keys = list(self.Edge.__table__.columns.keys())
        for k in [DEFAULT_ATTR_KEYS.EDGE_ID, DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]:
            keys.remove(k)
        return keys

    def _sqlalchemy_type_inference(self, default_value: Any) -> TypeEngine:
        if np.isscalar(default_value) and hasattr(default_value, "item"):
            default_value = default_value.item()

        if isinstance(default_value, float):
            return sa.Float

        # must come before integer, otherwise it will be interpreted as it
        elif isinstance(default_value, bool):
            return sa.Boolean

        elif isinstance(default_value, int):
            return sa.Integer

        elif isinstance(default_value, str):
            return sa.String

        elif isinstance(default_value, Enum):
            return sa.Enum(default_value.__class__)

        elif default_value is None or not _is_builtin(default_value):
            return sa.PickleType

        else:
            raise ValueError(f"Unsupported default value type: {type(default_value)}")

    def _add_new_column(
        self,
        table_class: type[DeclarativeBase],
        key: str,
        default_value: Any,
    ) -> None:
        sa_type = self._sqlalchemy_type_inference(default_value)

        if sa_type == sa.Boolean:
            self._boolean_columns[table_class.__tablename__].append(key)

        sa_column = sa.Column(key, sa_type, default=default_value)

        str_dialect_type = sa_column.type.compile(dialect=self._engine.dialect)

        add_column_stmt = sa.DDL(
            f"ALTER TABLE {table_class.__table__} ADD "
            f"COLUMN {sa_column.name} {str_dialect_type} "
            f"DEFAULT {default_value}",
        )
        LOG.info("add %s column statement:\n'%s'", table_class.__table__, add_column_stmt)

        # create the new column in the database
        with Session(self._engine) as session:
            session.execute(add_column_stmt)
            session.commit()

        # register the new column in the Node class
        setattr(table_class, key, sa_column)
        table_class.__table__.append_column(sa_column)

    def add_node_attr_key(self, key: str, default_value: Any) -> None:
        if key in self.node_attr_keys:
            raise ValueError(f"Node attribute key {key} already exists")

        self._add_new_column(self.Node, key, default_value)

    def add_edge_attr_key(self, key: str, default_value: Any) -> None:
        if key in self.edge_attr_keys:
            raise ValueError(f"Edge attribute key {key} already exists")

        self._add_new_column(self.Edge, key, default_value)

    @property
    def num_edges(self) -> int:
        with Session(self._engine) as session:
            return int(session.query(self.Edge).count())

    @property
    def num_nodes(self) -> int:
        with Session(self._engine) as session:
            return int(session.query(self.Node).count())

    def _update_table(
        self,
        table_class: type[DeclarativeBase],
        ids: Sequence[int] | None,
        id_key: str,
        attrs: dict[str, Any],
    ) -> None:
        if ids is not None:
            if len(ids) == 0:
                LOG.info("No ids to update, skipping")
                return

            if hasattr(ids, "tolist"):
                ids = ids.tolist()

        # Handle array values with bulk_update_mappings
        attrs = attrs.copy()
        _data_numpy_to_native(attrs)

        # specialized case for scalar values - use simple bulk update
        if all(np.isscalar(v) for v in attrs.values()):
            LOG.info("update %s table with scalar values: %s", table_class.__table__, attrs)

            with Session(self._engine) as session:
                query = session.query(table_class)
                if ids is not None:
                    query = query.filter(getattr(table_class, id_key).in_(ids))
                query.update(attrs)
                session.commit()

            return

        if ids is None:
            raise ValueError("`ids` must be provided to update with variable values.")

        # Prepare values for bulk update
        update_data = []

        for k, v in attrs.items():
            if np.isscalar(v):
                # Convert scalar to list of same length as ids
                attrs[k] = [v] * len(ids)
            else:
                if hasattr(v, "tolist"):
                    attrs[k] = v.tolist()

                # Validate length matches ids
                if len(attrs[k]) != len(ids):
                    raise ValueError(f"Length mismatch: {len(attrs[k])} values for {len(ids)} {id_key}s")

        # Create list of dictionaries for bulk update (simple primary key)
        for i, row_id in enumerate(ids):
            row_data = {id_key: row_id}

            # Add the attribute updates
            for k, v_list in attrs.items():
                row_data[k] = v_list[i]
            update_data.append(row_data)

        LOG.info("update %s table with %d rows", table_class.__table__, len(update_data))
        LOG.info("update data sample: %s", update_data[:2])

        with Session(self._engine) as session:
            session.execute(sa.update(table_class), update_data)
            session.commit()

    def update_node_attrs(
        self,
        *,
        attrs: dict[str, Any],
        node_ids: Sequence[int] | None = None,
    ) -> None:
        if "t" in attrs:
            raise ValueError("Node attribute 't' cannot be updated.")

        self._update_table(self.Node, node_ids, DEFAULT_ATTR_KEYS.NODE_ID, attrs)

    def update_edge_attrs(
        self,
        *,
        attrs: dict[str, Any],
        edge_ids: Sequence[int] | None = None,
    ) -> None:
        self._update_table(self.Edge, edge_ids, DEFAULT_ATTR_KEYS.EDGE_ID, attrs)

    def _get_degree(
        self,
        node_ids: list[int] | int | None,
        node_key: str,
    ) -> list[int] | int:
        if isinstance(node_ids, int):
            with Session(self._engine) as session:
                query = (
                    session.query(
                        getattr(self.Edge, node_key),
                    )
                    .filter(getattr(self.Edge, node_key) == node_ids)
                    .count()
                )
            return int(query)

        with Session(self._engine) as session:
            # get the number of edges for each using group by and count
            node_id_col = getattr(self.Edge, node_key)
            query = session.query(node_id_col, sa.func.count(node_id_col)).group_by(node_id_col)
            if node_ids is not None:
                query = query.filter(node_id_col.in_(node_ids))
            degree = dict(query.all())

        if node_ids is None:
            # this is necessary to make sure it's the same order as node_ids
            return [degree.get(node_id, 0) for node_id in self.node_ids()]

        return [degree.get(node_id, 0) for node_id in node_ids]

    def in_degree(self, node_ids: list[int] | int | None = None) -> list[int] | int:
        """
        Get the in-degree of a list of nodes.
        """
        return self._get_degree(node_ids, DEFAULT_ATTR_KEYS.EDGE_TARGET)

    def out_degree(self, node_ids: list[int] | int | None = None) -> list[int] | int:
        """
        Get the out-degree of a list of nodes.
        """
        return self._get_degree(node_ids, DEFAULT_ATTR_KEYS.EDGE_SOURCE)

    def __getstate__(self) -> dict:
        data_dict = self.__dict__.copy()
        for k in ["Base", "Node", "Edge", "Overlap", "_engine"]:
            del data_dict[k]
        return data_dict

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # recreate deleted objects
        self._engine = sa.create_engine(self._url, **self._engine_kwargs)
        self._define_schema(overwrite=False)

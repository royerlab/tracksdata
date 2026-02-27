import binascii
from collections.abc import Callable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

import cloudpickle
import numpy as np
import polars as pl
import rustworkx as rx
import sqlalchemy as sa
from polars._typing import SchemaDict
from sqlalchemy.orm import DeclarativeBase, Session, aliased, load_only
from sqlalchemy.orm.query import Query
from sqlalchemy.sql.type_api import TypeEngine

from tracksdata.attrs import AttrComparison, split_attr_comps
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.graph.filters._base_filter import BaseFilter
from tracksdata.utils._cache import cache_method
from tracksdata.utils._dataframe import unpack_array_attrs, unpickle_bytes_columns
from tracksdata.utils._dtypes import (
    AttrSchema,
    deserialize_attr_schema,
    polars_dtype_to_sqlalchemy_type,
    process_attr_key_args,
    serialize_attr_schema,
    sqlalchemy_type_to_polars_dtype,
)
from tracksdata.utils._logging import LOG
from tracksdata.utils._signal import is_signal_on

if TYPE_CHECKING:
    from tracksdata.graph._graph_view import GraphView


T = TypeVar("T")


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


def _coerce_json_field_expr(lhs: Any, dtype: pl.DataType | None) -> Any:
    if dtype is None:
        return lhs

    dtype_base = dtype.base_type()

    if dtype_base == pl.Boolean:
        if hasattr(lhs, "as_boolean"):
            return lhs.as_boolean()
        return sa.cast(lhs, sa.Boolean)
    if dtype_base in {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}:
        if hasattr(lhs, "as_integer"):
            return lhs.as_integer()
        return sa.cast(lhs, sa.BigInteger)
    if dtype_base in {pl.Float16, pl.Float32, pl.Float64}:
        if hasattr(lhs, "as_float"):
            return lhs.as_float()
        return sa.cast(lhs, sa.Float)
    if dtype_base in {pl.String, pl.Utf8}:
        if hasattr(lhs, "as_string"):
            return lhs.as_string()
        return sa.cast(lhs, sa.String)
    return lhs


def _field_dtype_from_schema(
    attr_filter: AttrComparison,
    attr_schemas: dict[str, AttrSchema] | None,
) -> pl.DataType | None:
    if attr_schemas is None:
        return None

    schema = attr_schemas.get(str(attr_filter.column))
    if schema is None:
        return None

    dtype = schema.dtype
    for field in attr_filter.attr.field_path:
        if not isinstance(dtype, pl.Struct):
            return None

        dtype = dtype.to_schema().get(field)
        if dtype is None:
            return None

    return dtype


def _resolve_attr_filter_column(
    table: type[DeclarativeBase],
    attr_filter: AttrComparison,
    attr_schemas: dict[str, AttrSchema] | None = None,
) -> Any:
    lhs = getattr(table, str(attr_filter.column))

    if not attr_filter.attr.field_path:
        return lhs

    for field in attr_filter.attr.field_path:
        lhs = lhs[field]

    field_dtype = _field_dtype_from_schema(attr_filter, attr_schemas)
    return _coerce_json_field_expr(lhs, field_dtype)


def _json_decode_safe_dtype(dtype: pl.DataType) -> pl.DataType:
    """
    Return a JSON-decodable dtype by replacing fixed-size arrays with lists recursively.
    """
    if isinstance(dtype, pl.Array):
        return pl.List(_json_decode_safe_dtype(dtype.inner))

    if isinstance(dtype, pl.List):
        return pl.List(_json_decode_safe_dtype(dtype.inner))

    if isinstance(dtype, pl.Struct):
        return pl.Struct({key: _json_decode_safe_dtype(inner) for key, inner in dtype.to_schema().items()})

    return dtype


def _struct_json_decode_expr(column: str, target_dtype: pl.Struct) -> pl.Expr:
    decode_dtype = _json_decode_safe_dtype(target_dtype)
    decoded_expr = pl.when(pl.col(column).is_null()).then(None).otherwise(pl.col(column).str.json_decode(decode_dtype))
    if decode_dtype != target_dtype:
        decoded_expr = decoded_expr.cast(target_dtype)
    return decoded_expr.alias(column)


def _filter_query(
    query: sa.Select,
    table: type[DeclarativeBase],
    attr_filters: list[AttrComparison],
    attr_schemas: dict[str, AttrSchema] | None = None,
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
    attr_schemas : dict[str, AttrSchema] | None, optional
        Attribute schema map used to resolve nested struct field dtypes.

    Returns
    -------
    sa.Select
        The filtered query.
    """
    LOG.info("Filter query:\n%s", attr_filters)
    query = query.filter(
        *[
            attr_filter.op(
                _resolve_attr_filter_column(table, attr_filter, attr_schemas=attr_schemas),
                attr_filter.other,
            )
            for attr_filter in attr_filters
        ]
    )
    return query


class SQLFilter(BaseFilter):
    def __init__(
        self,
        *attr_filters: AttrComparison,
        graph: "SQLGraph",
        node_ids: Sequence[int] | None = None,
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
        node_filtered = False
        node_attr_schemas = self._graph._node_attr_schemas()
        edge_attr_schemas = self._graph._edge_attr_schemas()

        if node_ids is not None:
            if hasattr(node_ids, "tolist"):
                node_ids = node_ids.tolist()

            self._node_query = self._node_query.filter(self._graph.Node.node_id.in_(node_ids))
            if not self._include_targets:
                self._edge_query = self._edge_query.filter(
                    self._graph.Edge.target_id.in_(node_ids),
                )
            if not self._include_sources:
                self._edge_query = self._edge_query.filter(
                    self._graph.Edge.source_id.in_(node_ids),
                )
            node_filtered = True

        if self._node_attr_comps:
            node_filtered = True
            # filtering nodes by attributes
            self._node_query = _filter_query(
                self._node_query,
                self._graph.Node,
                self._node_attr_comps,
                attr_schemas=node_attr_schemas,
            )

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
                self._edge_query = _filter_query(
                    self._edge_query,
                    SourceNode,
                    self._node_attr_comps,
                    attr_schemas=node_attr_schemas,
                )

            if self._include_sources or include_none:
                self._edge_query = self._edge_query.join(
                    TargetNode,
                    self._graph.Edge.target_id == TargetNode.node_id,
                )
                self._edge_query = _filter_query(
                    self._edge_query,
                    TargetNode,
                    self._node_attr_comps,
                    attr_schemas=node_attr_schemas,
                )

        if self._edge_attr_comps:
            self._edge_query = _filter_query(
                self._edge_query,
                self._graph.Edge,
                self._edge_attr_comps,
                attr_schemas=edge_attr_schemas,
            )

            # we haven't filtered the nodes by attributes
            # so we only return the nodes that are in the edges
            if not node_filtered:
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

    @cache_method
    def node_ids(self) -> list[int]:
        """
        Get the ids of the nodes resulting from the filter.
        """
        with Session(self._graph._engine) as session:
            if isinstance(self._node_query, sa.CompoundSelect):
                return session.execute(self._node_query.options(load_only(self._graph.Node.node_id))).scalars().all()
            else:
                return session.execute(self._node_query.with_only_columns(self._graph.Node.node_id)).scalars().all()

    @cache_method
    def edge_ids(self) -> list[int]:
        """
        Get the ids of the edges resulting from the filter.
        """
        with Session(self._graph._engine) as session:
            if isinstance(self._edge_query, sa.CompoundSelect):
                return session.execute(self._edge_query.options(load_only(self._graph.Edge.edge_id))).scalars().all()
            else:
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
                schema_overrides=self._graph._polars_schema_override(self._graph.Node),
            )

        if attr_keys is not None:
            nodes_attrs = nodes_attrs.select(attr_keys)

        nodes_attrs = unpickle_bytes_columns(nodes_attrs)
        nodes_attrs = self._graph._cast_array_columns(self._graph.Node, nodes_attrs)

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
                schema_overrides=self._graph._polars_schema_override(self._graph.Edge),
            )

        edges_df = unpickle_bytes_columns(edges_df)
        edges_df = self._graph._cast_array_columns(self._graph.Edge, edges_df)

        if unpack:
            edges_df = unpack_array_attrs(edges_df)

        return edges_df

    @cache_method
    def subgraph(
        self,
        node_attr_keys: Sequence[str] | None = None,
        edge_attr_keys: Sequence[str] | None = None,
    ) -> "GraphView":
        from tracksdata.graph._graph_view import GraphView

        # Give the node_attr_keys as a list, since otherwise the SQL results return the
        # Ensure the time key is in the node attributes
        if node_attr_keys is not None:
            node_attr_keys = [DEFAULT_ATTR_KEYS.T, DEFAULT_ATTR_KEYS.NODE_ID, *node_attr_keys]
        else:
            node_attr_keys = [DEFAULT_ATTR_KEYS.T, *self._graph.node_attr_keys(return_ids=True)]

        node_attr_keys = list(dict.fromkeys(node_attr_keys))

        if edge_attr_keys is None:
            edge_attr_keys = self._graph.edge_attr_keys(return_ids=True).copy()

        node_query = self._query_from_attr_keys(
            query=self._node_query,
            table=self._graph.Node,
            attr_keys=node_attr_keys,
            extra_columns=[DEFAULT_ATTR_KEYS.NODE_ID],
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

            for row in node_query.mappings().all():
                data = dict(row)
                root_node_id = data.pop(DEFAULT_ATTR_KEYS.NODE_ID)
                node_id = rx_graph.add_node(data)
                node_map_to_root[node_id] = root_node_id
                node_map_from_root[root_node_id] = node_id

            for row in edge_query.mappings().all():
                data = dict(row)
                source_id = node_map_from_root[data.pop(DEFAULT_ATTR_KEYS.EDGE_SOURCE)]
                target_id = node_map_from_root[data.pop(DEFAULT_ATTR_KEYS.EDGE_TARGET)]
                rx_graph.add_edge(source_id, target_id, data)

        graph = GraphView(
            rx_graph=rx_graph,
            node_map_to_root=node_map_to_root,
            root=self._graph,
            node_attr_keys=node_attr_keys,
            edge_attr_keys=edge_attr_keys,
        )

        return graph


def blob_default(engine: sa.Engine, value: bytes) -> sa.Text:
    """Return a dialect-correct server_default for LargeBinary."""
    hex_blob = binascii.hexlify(value).decode("ascii")
    if engine.dialect.name == "sqlite":
        return sa.text(f"X'{hex_blob}'")
    elif engine.dialect.name in {"postgresql", "postgres"}:
        # either works:
        # return sa.text(f"decode('{hex_blob}','hex')")
        return sa.text(f"'\\x{hex_blob}'::bytea")
    else:
        raise RuntimeError(f"Unsupported dialect {engine.dialect.name}")


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
    _PRIVATE_SQL_NODE_SCHEMA_STORE_KEY = "__private_sql_node_attr_schema_store"
    _PRIVATE_SQL_EDGE_SCHEMA_STORE_KEY = "__private_sql_edge_attr_schema_store"
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

        if overwrite:
            self.Base.metadata.drop_all(self._engine)

        self.Base.metadata.create_all(self._engine)

        self._max_id_per_time = {}
        self._update_max_id_per_time()

    def supports_custom_indices(self) -> bool:
        return True

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
            for table in metadata.tables.values():
                self._restore_pickled_column_types(table)
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
            edge_id = sa.Column(sa.Integer, sa.Identity(always=True), primary_key=True, unique=True)
            source_id = sa.Column(sa.BigInteger, sa.ForeignKey(f"{node_tb_name}.node_id"), index=True, nullable=False)
            target_id = sa.Column(sa.BigInteger, sa.ForeignKey(f"{node_tb_name}.node_id"), index=True, nullable=False)

        class Overlap(Base):
            __tablename__ = "Overlap"
            overlap_id = sa.Column(sa.Integer, sa.Identity(always=True), primary_key=True, unique=True)
            source_id = sa.Column(sa.BigInteger, sa.ForeignKey(f"{node_tb_name}.node_id"), index=True, nullable=False)
            target_id = sa.Column(sa.BigInteger, sa.ForeignKey(f"{node_tb_name}.node_id"), index=True, nullable=False)

        class Metadata(Base):
            __tablename__ = "Metadata"
            key = sa.Column(sa.String, primary_key=True)
            value = sa.Column(sa.JSON)

        # Assign to instance variables
        self.Base = Base
        self.Node = Node
        self.Edge = Edge
        self.Overlap = Overlap
        self.Metadata = Metadata

    @staticmethod
    def _default_node_attr_schemas() -> dict[str, AttrSchema]:
        return {
            DEFAULT_ATTR_KEYS.T: AttrSchema(key=DEFAULT_ATTR_KEYS.T, dtype=pl.Int32),
            DEFAULT_ATTR_KEYS.NODE_ID: AttrSchema(key=DEFAULT_ATTR_KEYS.NODE_ID, dtype=pl.Int64),
        }

    @staticmethod
    def _default_edge_attr_schemas() -> dict[str, AttrSchema]:
        return {
            DEFAULT_ATTR_KEYS.EDGE_ID: AttrSchema(key=DEFAULT_ATTR_KEYS.EDGE_ID, dtype=pl.Int32),
            DEFAULT_ATTR_KEYS.EDGE_SOURCE: AttrSchema(key=DEFAULT_ATTR_KEYS.EDGE_SOURCE, dtype=pl.Int64),
            DEFAULT_ATTR_KEYS.EDGE_TARGET: AttrSchema(key=DEFAULT_ATTR_KEYS.EDGE_TARGET, dtype=pl.Int64),
        }

    def _attr_schemas_from_metadata(
        self,
        *,
        table_class: type[DeclarativeBase],
        metadata_key: str,
        default_schemas: dict[str, AttrSchema],
        preferred_order: Sequence[str],
    ) -> dict[str, AttrSchema]:
        encoded_schemas = self._private_metadata.get(metadata_key, {})
        schemas = default_schemas.copy()
        schemas.update(
            {key: deserialize_attr_schema(encoded_schema, key=key) for key, encoded_schema in encoded_schemas.items()}
        )

        # Legacy databases may not have schema metadata for all columns.
        for column_name, column in table_class.__table__.columns.items():
            if column_name not in schemas:
                schemas[column_name] = AttrSchema(
                    key=column_name,
                    dtype=sqlalchemy_type_to_polars_dtype(column.type),
                )

        ordered_keys = [key for key in preferred_order if key in schemas]
        ordered_keys.extend(key for key in table_class.__table__.columns.keys() if key not in ordered_keys)
        ordered_keys.extend(key for key in schemas if key not in ordered_keys)
        return {key: schemas[key] for key in ordered_keys}

    def _attr_schemas_for_table(self, table_class: type[DeclarativeBase]) -> dict[str, AttrSchema]:
        if table_class.__tablename__ == self.Node.__tablename__:
            return self._node_attr_schemas()
        return self._edge_attr_schemas()

    @staticmethod
    def _is_pickled_sql_type(column_type: TypeEngine) -> bool:
        return isinstance(column_type, sa.PickleType | sa.LargeBinary)

    @staticmethod
    def _is_json_sql_type(column_type: TypeEngine) -> bool:
        return isinstance(column_type, sa.JSON)

    @property
    def __node_attr_schemas(self) -> dict[str, AttrSchema]:
        return self._attr_schemas_from_metadata(
            table_class=self.Node,
            metadata_key=self._PRIVATE_SQL_NODE_SCHEMA_STORE_KEY,
            default_schemas=self._default_node_attr_schemas(),
            preferred_order=[DEFAULT_ATTR_KEYS.T, DEFAULT_ATTR_KEYS.NODE_ID],
        )

    @__node_attr_schemas.setter
    def __node_attr_schemas(self, schemas: dict[str, AttrSchema]) -> None:
        merged_schemas = self._default_node_attr_schemas()
        merged_schemas.update(schemas)
        schemas = merged_schemas
        encoded_schemas = {key: serialize_attr_schema(schema) for key, schema in schemas.items()}
        self._private_metadata[self._PRIVATE_SQL_NODE_SCHEMA_STORE_KEY] = encoded_schemas

    @property
    def __edge_attr_schemas(self) -> dict[str, AttrSchema]:
        return self._attr_schemas_from_metadata(
            table_class=self.Edge,
            metadata_key=self._PRIVATE_SQL_EDGE_SCHEMA_STORE_KEY,
            default_schemas=self._default_edge_attr_schemas(),
            preferred_order=[
                DEFAULT_ATTR_KEYS.EDGE_ID,
                DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                DEFAULT_ATTR_KEYS.EDGE_TARGET,
            ],
        )

    @__edge_attr_schemas.setter
    def __edge_attr_schemas(self, schemas: dict[str, AttrSchema]) -> None:
        merged_schemas = self._default_edge_attr_schemas()
        merged_schemas.update(schemas)
        schemas = merged_schemas
        encoded_schemas = {key: serialize_attr_schema(schema) for key, schema in schemas.items()}
        self._private_metadata[self._PRIVATE_SQL_EDGE_SCHEMA_STORE_KEY] = encoded_schemas

    def _restore_pickled_column_types(self, table: sa.Table) -> None:
        for column in table.columns:
            if isinstance(column.type, sa.LargeBinary):
                column.type = sa.PickleType()

    def _polars_schema_override(self, table_class: type[DeclarativeBase]) -> SchemaDict:
        schemas = self._attr_schemas_for_table(table_class)

        # Return schema overrides for columns safely represented in SQL.
        # Pickled columns are unpickled and casted in a second pass.
        return {
            key: schema.dtype
            for key, schema in schemas.items()
            if (
                key in table_class.__table__.columns
                and not self._is_pickled_sql_type(table_class.__table__.columns[key].type)
                and not isinstance(schema.dtype, pl.Struct)
            )
        }

    def _cast_array_columns(self, table_class: type[DeclarativeBase], df: pl.DataFrame) -> pl.DataFrame:
        schemas = self._attr_schemas_for_table(table_class)

        decode_exprs: list[pl.Expr] = []
        casts: list[pl.Series] = []
        for key, schema in schemas.items():
            if key not in df.columns or key not in table_class.__table__.columns:
                continue

            column_type = table_class.__table__.columns[key].type
            source_dtype = df.schema[key]

            if isinstance(schema.dtype, pl.Struct) and self._is_json_sql_type(column_type):
                if source_dtype == pl.String:
                    decode_exprs.append(_struct_json_decode_expr(key, schema.dtype))
                elif source_dtype != schema.dtype:
                    casts.append(pl.Series(key, df[key].to_list(), dtype=schema.dtype))
                continue

            if not self._is_pickled_sql_type(column_type):
                continue

            try:
                casts.append(pl.Series(key, df[key].to_list(), dtype=schema.dtype))
            except Exception:
                # Keep original dtype when values cannot be casted to the target schema.
                continue

        if decode_exprs:
            df = df.with_columns(decode_exprs)
        if casts:
            df = df.with_columns(casts)
        return df

    def _update_max_id_per_time(self) -> None:
        """
        Update the maximum node ID for each time point.

        Scans the database to find the current maximum node ID for each time
        point and updates the internal cache to ensure newly created nodes
        have unique IDs.
        """
        with Session(self._engine) as session:
            stmt = sa.select(self.Node.t, sa.func.max(self.Node.node_id)).group_by(self.Node.t)
            self._max_id_per_time = {int(time): int(max_id) for time, max_id in session.execute(stmt).all()}

    def filter(
        self,
        *attr_filters: AttrComparison,
        node_ids: Sequence[int] | None = None,
        include_targets: bool = False,
        include_sources: bool = False,
    ) -> "SQLFilter":
        return SQLFilter(
            *attr_filters,
            graph=self,
            node_ids=node_ids,
            include_targets=include_targets,
            include_sources=include_sources,
        )

    def add_node(
        self,
        attrs: dict[str, Any],
        validate_keys: bool = True,
        index: int | None = None,
    ) -> int:
        """
        Add a node to the graph at time t.

        Node IDs are automatically generated based on the time point and
        the node_id_time_multiplier to ensure uniqueness across time points,
        unless an explicit index is provided.

        Parameters
        ----------
        attrs : dict[str, Any]
            The attributes of the node to be added. Must contain a "t" key
            specifying the time point. Additional keys will be stored as
            node attributes.
        validate_keys : bool, default True
            Whether to check if the attribute keys are valid against the
            current schema. If False, validation is skipped for performance.
        index : int | None, default None
            Optional specific node ID to use. If provided, this will be used
            as the node_id instead of the auto-generated value.

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
        node_id = graph.add_node({"t": 0, "x": 20.0, "y": 30.0}, index=12345)
        ```
        """
        if validate_keys:
            self._validate_attributes(attrs, self.node_attr_keys(), "node")

            if "t" not in attrs:
                raise ValueError(f"Node attributes must have a 't' key. Got {attrs.keys()}")

        time = attrs["t"]

        if index is None:
            default_node_id = (time * self.node_id_time_multiplier) - 1
            node_id = self._max_id_per_time.get(time, default_node_id) + 1
        else:
            node_id = index

        node = self.Node(
            node_id=node_id,
            **attrs,
        )

        with Session(self._engine) as session:
            session.add(node)
            session.commit()

        # Update max_id tracking only if using auto-generated IDs
        if index is None:
            self._max_id_per_time[time] = node_id

        if is_signal_on(self.node_added):
            self.node_added.emit(node_id, attrs)

        return node_id

    def bulk_add_nodes(
        self,
        nodes: list[dict[str, Any]],
        indices: list[int] | None = None,
    ) -> list[int]:
        """
        Add multiple nodes to the graph efficiently.

        Provides better performance than calling add_node multiple times by
        using SQLAlchemy's bulk insert functionality and reducing database
        transactions. Automatically assigns node IDs and handles time-based
        ID generation, or uses provided indices.

        Parameters
        ----------
        nodes : list[dict[str, Any]]
            List of node attribute dictionaries. Each dictionary must contain
            a "t" key and any additional attribute keys.
        indices : list[int] | None, default None
            Optional list of specific node IDs to use. If provided, must be
            the same length as nodes. If None, IDs are auto-generated.

        Examples
        --------
        ```python
        nodes = [
            {"t": 0, "x": 10, "y": 20, "label": "A"},
            {"t": 0, "x": 15, "y": 25, "label": "B"},
        ]
        graph.bulk_add_nodes(nodes)

        # With specific indices
        graph.bulk_add_nodes(nodes, indices=[1000, 2000])
        ```

        Returns
        -------
        list[int]
            The IDs of the added nodes.
        """
        if len(nodes) == 0:
            return []

        self._validate_indices_length(nodes, indices)

        node_ids = []
        for i, node in enumerate(nodes):
            time = node["t"]

            if indices is None:
                default_node_id = (time * self.node_id_time_multiplier) - 1
                node_id = self._max_id_per_time.get(time, default_node_id) + 1
                # Update max_id tracking only for auto-generated IDs
                self._max_id_per_time[time] = node_id
            else:
                node_id = indices[i]

            node[DEFAULT_ATTR_KEYS.NODE_ID] = node_id
            node_ids.append(node_id)

        self._chunked_sa_write(Session.bulk_insert_mappings, nodes, self.Node)

        if is_signal_on(self.node_added):
            for node_id, node_attrs in zip(node_ids, nodes, strict=True):
                new_attrs = {key: value for key, value in node_attrs.items() if key != DEFAULT_ATTR_KEYS.NODE_ID}
                self.node_added.emit(node_id, new_attrs)

        return node_ids

    def remove_node(self, node_id: int) -> None:
        """
        Remove a node from the graph.

        This method removes the specified node and all edges connected to it
        (both incoming and outgoing edges). Also removes any overlaps
        involving this node.

        Parameters
        ----------
        node_id : int
            The ID of the node to remove.

        Raises
        ------
        ValueError
            If the node_id does not exist in the graph.
        """
        with Session(self._engine) as session:
            # Check if the node exists
            node = session.query(self.Node).filter(self.Node.node_id == node_id).first()
            if node is None:
                raise ValueError(f"Node {node_id} does not exist in the graph.")

            if is_signal_on(self.node_removed):
                old_attrs = {key: getattr(node, key) for key in self.node_attr_keys()}

            # Remove all edges where this node is source or target
            session.query(self.Edge).filter(
                sa.or_(self.Edge.source_id == node_id, self.Edge.target_id == node_id)
            ).delete()

            # Remove all overlaps involving this node
            session.query(self.Overlap).filter(
                sa.or_(self.Overlap.source_id == node_id, self.Overlap.target_id == node_id)
            ).delete()

            # Remove the node itself
            session.delete(node)
            session.commit()
            if is_signal_on(self.node_removed):
                self.node_removed.emit(node_id, old_attrs)

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
            self._validate_attributes(attrs, self.edge_attr_keys(), "edge")

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

        if return_ids:
            with Session(self._engine) as session:
                result = session.execute(sa.insert(self.Edge).returning(self.Edge.edge_id), edges)
                session.commit()
                return list(result.scalars().all())

        else:
            self._chunked_sa_write(Session.bulk_insert_mappings, edges, self.Edge)
            return None

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
        self._chunked_sa_write(Session.bulk_insert_mappings, overlaps, self.Overlap)

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
        node_ids: list[int] | int | None,
        attr_keys: Sequence[str] | str | None = None,
        *,
        return_attrs: bool = False,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]:
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
        node_ids : list[int] | int | None
            The IDs of the nodes to get neighbors for.
            If None, all nodes are used.
        attr_keys : Sequence[str] | str | None, optional
            The attribute keys to retrieve for neighbor nodes when ``return_attrs`` is True.
            If None, all attributes are retrieved.
        return_attrs : bool, default False
            Whether to return the attributes DataFrame. When False only neighbor
            node IDs are returned.

        Returns
        -------
        dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]
            When ``return_attrs`` is True, returns a DataFrame for a single node or a dictionary
            mapping each node ID to a DataFrame of neighbor attributes. Otherwise returns a list
            of neighbor node IDs for a single node or a dictionary mapping each node ID to its
            neighbor ID list.
        """
        single_node = False
        filter_node_ids: list[int] | None
        if isinstance(node_ids, int):
            node_ids = [node_ids]
            filter_node_ids = node_ids
            single_node = True
        elif node_ids is None:
            node_ids = self.node_ids()
            filter_node_ids = None
        else:
            filter_node_ids = node_ids

        if isinstance(attr_keys, str):
            attr_keys = [attr_keys]

        with Session(self._engine) as session:
            if not return_attrs:
                # only neighbor IDs
                node_columns = [self.Node.node_id]
                if attr_keys is not None:
                    LOG.warning("attr_keys is ignored when return_attrs is False.")
                    attr_keys = None
            elif attr_keys is None:
                # all columns
                node_columns = [self.Node]
            else:
                node_columns = [getattr(self.Node, key) for key in attr_keys]

            query = session.query(getattr(self.Edge, node_key), *node_columns)
            query = query.join(self.Edge, getattr(self.Edge, neighbor_key) == self.Node.node_id)
            if filter_node_ids is None or len(filter_node_ids) == 0:
                node_df = pl.read_database(
                    query.statement,
                    connection=session.connection(),
                    schema_overrides=self._polars_schema_override(self.Node),
                )
            else:
                node_df = self._chunked_sa_read(
                    session,
                    lambda x: query.filter(getattr(self.Edge, node_key).in_(x)),
                    filter_node_ids,
                    self.Node,
                )
            node_df = unpickle_bytes_columns(node_df)
            node_df = self._cast_array_columns(self.Node, node_df)

        if single_node:
            if not return_attrs:
                return node_df[DEFAULT_ATTR_KEYS.NODE_ID].to_list()
            else:
                return node_df
        else:
            neighbors_dict = {node_id: group for (node_id,), group in node_df.group_by(node_key)}
            for node_id in node_ids:
                neighbors_dict.setdefault(node_id, pl.DataFrame(schema=node_df.schema))
            if not return_attrs:
                return {node_id: df[DEFAULT_ATTR_KEYS.NODE_ID].to_list() for node_id, df in neighbors_dict.items()}
            else:
                return neighbors_dict

    def successors(
        self,
        node_ids: list[int] | int | None,
        attr_keys: Sequence[str] | str | None = None,
        *,
        return_attrs: bool = False,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]:
        """
        Get the successor nodes of given nodes.

        Successors are nodes that are targets of edges originating from the
        given nodes (outgoing edges). Uses efficient SQL joins to retrieve
        successor information with their attributes in a single query.

        Parameters
        ----------
        node_ids : list[int] | int | None
            The IDs of the nodes to get successors for.
            If None, all nodes are used.
        attr_keys : Sequence[str] | str | None, optional
            The attribute keys to retrieve for successor nodes when ``return_attrs`` is True.
            If None, all attributes are retrieved.
        return_attrs : bool, default False
            Whether to return the attributes DataFrame. When False only successor
            node IDs are returned.

        Returns
        -------
        dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]
            When ``return_attrs`` is True, returns a DataFrame for a single node or a dictionary
            mapping each node ID to a DataFrame of successor attributes. Otherwise returns a list
            of successor node IDs for a single node or a dictionary mapping each node ID to its
            successor ID list.

        Examples
        --------
        ```python
        successors_df = graph.sucessors(node_id, return_attrs=True)
        successors_dict = graph.sucessors([node1, node2, node3], return_attrs=True)
        ```
        """
        return self._get_neighbors(
            node_key=DEFAULT_ATTR_KEYS.EDGE_SOURCE,
            neighbor_key=DEFAULT_ATTR_KEYS.EDGE_TARGET,
            node_ids=node_ids,
            attr_keys=attr_keys,
            return_attrs=return_attrs,
        )

    def predecessors(
        self,
        node_ids: list[int] | int | None,
        attr_keys: Sequence[str] | str | None = None,
        *,
        return_attrs: bool = False,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]:
        """
        Get the predecessor nodes of given nodes.

        Predecessors are nodes that are sources of edges targeting the
        given nodes (incoming edges). Uses efficient SQL joins to retrieve
        predecessor information with their attributes in a single query.

        Parameters
        ----------
        node_ids : list[int] | int | None
            The IDs of the nodes to get predecessors for.
            If None, all nodes are used.
        attr_keys : Sequence[str] | str | None, optional
            The attribute keys to retrieve for predecessor nodes when ``return_attrs`` is True.
            If None, all attributes are retrieved.
        return_attrs : bool, default False
            Whether to return the attributes DataFrame. When False only predecessor
            node IDs are returned.

        Returns
        -------
        dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]
            When ``return_attrs`` is True, returns a DataFrame for a single node or a dictionary
            mapping each node ID to a DataFrame of predecessor attributes. Otherwise returns a list
            of predecessor node IDs for a single node or a dictionary mapping each node ID to its
            predecessor ID list.

        Examples
        --------
        ```python
        predecessors_df = graph.predecessors(node_id, return_attrs=True)
        predecessors_dict = graph.predecessors([node1, node2, node3], return_attrs=True)
        ```
        """
        return self._get_neighbors(
            node_key=DEFAULT_ATTR_KEYS.EDGE_TARGET,
            neighbor_key=DEFAULT_ATTR_KEYS.EDGE_SOURCE,
            node_ids=node_ids,
            attr_keys=attr_keys,
            return_attrs=return_attrs,
        )

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
        attr_keys: Sequence[str] | str | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        if isinstance(attr_keys, str):
            attr_keys = [attr_keys]

        with Session(self._engine) as session:
            query = sa.select(self.Node)

            if attr_keys is not None:
                # making them unique
                attr_keys = list(dict.fromkeys(attr_keys))

                query = query.with_only_columns(
                    *[getattr(self.Node, key) for key in attr_keys],
                )

            nodes_df = pl.read_database(
                self._raw_query(query),
                connection=session.connection(),
                schema_overrides=self._polars_schema_override(self.Node),
            )
            nodes_df = unpickle_bytes_columns(nodes_df)
            nodes_df = self._cast_array_columns(self.Node, nodes_df)

        # indices are included by default and must be removed
        if attr_keys is not None:
            nodes_df = nodes_df.select([pl.col(c) for c in attr_keys])
        else:
            nodes_df = nodes_df.select([pl.col(c) for c in self._node_attr_schemas() if c in nodes_df.columns])

        if unpack:
            nodes_df = unpack_array_attrs(nodes_df)

        return nodes_df

    def edge_attrs(
        self,
        *,
        attr_keys: Sequence[str] | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        if isinstance(attr_keys, str):
            attr_keys = [attr_keys]

        with Session(self._engine) as session:
            query = sa.select(self.Edge)

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
                schema_overrides=self._polars_schema_override(self.Edge),
            )
            edges_df = unpickle_bytes_columns(edges_df)
            edges_df = self._cast_array_columns(self.Edge, edges_df)

        if unpack:
            edges_df = unpack_array_attrs(edges_df)
        elif attr_keys is None:
            edges_df = edges_df.select([pl.col(c) for c in self._edge_attr_schemas() if c in edges_df.columns])

        return edges_df

    def _node_attr_schemas(self) -> dict[str, AttrSchema]:
        return self.__node_attr_schemas

    def _edge_attr_schemas(self) -> dict[str, AttrSchema]:
        return self.__edge_attr_schemas

    def node_attr_keys(self, return_ids: bool = False) -> list[str]:
        """
        Get the keys of the attributes of the nodes.

        Parameters
        ----------
        return_ids : bool, optional
            Whether to include NODE_ID in the returned keys. Defaults to False.
            If True, NODE_ID will be included in the list.
        """
        keys = list(self.Node.__table__.columns.keys())
        if not return_ids and DEFAULT_ATTR_KEYS.NODE_ID in keys:
            keys.remove(DEFAULT_ATTR_KEYS.NODE_ID)
        return keys

    def edge_attr_keys(self, return_ids: bool = False) -> list[str]:
        """
        Get the keys of the attributes of the edges.

        Parameters
        ----------
        return_ids : bool, optional
            Whether to include EDGE_ID, EDGE_SOURCE, and EDGE_TARGET in the returned keys.
            Defaults to False. If True, these ID fields will be included in the list.
        """
        keys = list(self.Edge.__table__.columns.keys())
        if not return_ids:
            for id_key in [DEFAULT_ATTR_KEYS.EDGE_ID, DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]:
                if id_key in keys:
                    keys.remove(id_key)
        return keys

    def _resolve_attr_keys(
        self,
        table_class: type[DeclarativeBase],
        attr_keys: Sequence[str] | str,
    ) -> tuple[list[sa.Column], str]:
        """Check that the given attribute keys exist on the table class and
        generate resolved columns and an index name based on them.

        Parameters
        ----------
        table_class : type[DeclarativeBase]
            The SQLAlchemy table class.
        attr_keys : Sequence[str] | str
            The attribute keys to include in the index name.

        Returns
        -------
        str
            The generated index name.
        """
        if isinstance(attr_keys, str):
            attr_keys = [attr_keys]

        if len(attr_keys) == 0:
            raise ValueError("attr_keys must contain at least one column name")

        missing = [key for key in attr_keys if key not in table_class.__table__.columns]
        if missing:
            raise ValueError(f"Columns {missing} do not exist on table {table_class.__tablename__}")
        resolved_columns = [getattr(table_class, key) for key in attr_keys]

        if isinstance(attr_keys, str):
            attr_keys = [attr_keys]

        cols_fragment = "_".join(attr_keys)
        name = f"ix_{table_class.__tablename__.lower()}_{cols_fragment}"
        return resolved_columns, name

    def _create_attr_index(
        self,
        table_class: type[DeclarativeBase],
        attr_keys: Sequence[str] | str,
        *,
        unique: bool = False,
    ) -> str:
        resolved_columns, name = self._resolve_attr_keys(table_class, attr_keys)

        index = sa.Index(name, *resolved_columns, unique=unique)

        LOG.info(
            "Ensuring index '%s' on table %s (columns=%s, unique=%s)",
            name,
            table_class.__tablename__,
            attr_keys,
            unique,
        )
        index.create(bind=self._engine, checkfirst=True)
        return name

    def _drop_attr_index(
        self,
        table_class: type[DeclarativeBase],
        attr_keys: Sequence[str] | str,
    ) -> str:
        resolved_columns, name = self._resolve_attr_keys(table_class, attr_keys)
        index = sa.Index(name, *resolved_columns)

        LOG.info(
            "Dropping index '%s' on table %s (columns=%s)",
            name,
            table_class.__tablename__,
            attr_keys,
        )
        index.drop(bind=self._engine, checkfirst=True)
        return name

    def create_node_attr_index(
        self,
        attr_keys: Sequence[str] | str,
        *,
        unique: bool = False,
    ) -> str:
        """
        Ensure an index exists for the given node attribute columns.
        If they are already indexed, they are kept as they are.

        Parameters
        ----------
        attr_keys : Sequence[str] | str
            Column names to include in the index. Can be a single column name
            or a list of multiple columns for a composite index.
        unique : bool, default False
            Whether the index should enforce uniqueness.

        Returns
        -------
        str
            The name of the index.

        """

        return self._create_attr_index(self.Node, attr_keys, unique=unique)

    def create_edge_attr_index(
        self,
        attr_keys: Sequence[str] | str,
        *,
        unique: bool = False,
    ) -> str:
        """Ensure an index exists for the given edge attribute columns.

        Parameters
        ----------
        attr_keys : Sequence[str] | str
            Column names to include in the index. Can be a single column name
            or a list of multiple columns for a composite index.
        unique : bool, default False
            Whether the index should enforce uniqueness.

        Returns
        -------
        str
            The name of the index.

        """

        return self._create_attr_index(self.Edge, attr_keys, unique=unique)

    def drop_node_attr_index(self, attr_keys: Sequence[str] | str) -> str:
        """Drop an index for the given node attribute columns.

        Parameters
        ----------
        attr_keys : Sequence[str] | str
            Column names to include in the index. Can be a single column name
            or a list of multiple columns for a composite index.

        Returns
        -------
        str
            The dropped index name.
        """
        return self._drop_attr_index(self.Node, attr_keys)

    def drop_edge_attr_index(self, attr_keys: Sequence[str] | str) -> str:
        """Drop an index for the given edge attribute columns.

        Parameters
        ----------
        attr_keys : Sequence[str] | str
            Column names to include in the index. Can be a single column name
            or a list of multiple columns for a composite index.

        Returns
        -------
        str
            The dropped index name.
        """
        return self._drop_attr_index(self.Edge, attr_keys)

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
        schema: AttrSchema,
    ) -> None:
        # Convert polars dtype to SQLAlchemy type
        sa_type = polars_dtype_to_sqlalchemy_type(schema.dtype)

        # Handle special cases for default value encoding
        default_value = schema.default_value
        if isinstance(sa_type, sa.PickleType) and default_value is not None:
            # Pickle complex types for database storage
            default_value = blob_default(self._engine, cloudpickle.dumps(default_value))

        sa_column = sa.Column(schema.key, sa_type, default=default_value)

        str_dialect_type = sa_column.type.compile(dialect=self._engine.dialect)
        identifier_preparer = self._engine.dialect.identifier_preparer
        quoted_table_name = identifier_preparer.format_table(table_class.__table__)
        quoted_column_name = identifier_preparer.quote(sa_column.name)

        # Properly quote default values based on type
        if isinstance(default_value, str):
            quoted_default = f"'{default_value}'"
        elif default_value is None:
            quoted_default = "NULL"
        else:
            quoted_default = str(default_value)

        add_column_stmt = sa.DDL(
            f"ALTER TABLE {quoted_table_name} ADD "
            f"COLUMN {quoted_column_name} {str_dialect_type} "
            f"DEFAULT {quoted_default}",
        )
        LOG.info("add %s column statement:\n'%s'", table_class.__table__, add_column_stmt)

        # create the new column in the database
        with Session(self._engine) as session:
            session.execute(add_column_stmt)
            session.commit()

        # register the new column in the Node class
        setattr(table_class, schema.key, sa_column)
        table_class.__table__.append_column(sa_column)

    def _drop_column(self, table_class: type[DeclarativeBase], key: str) -> None:
        identifier_preparer = self._engine.dialect.identifier_preparer
        quoted_table_name = identifier_preparer.format_table(table_class.__table__)
        quoted_column_name = identifier_preparer.quote(key)
        drop_column_stmt = sa.DDL(f"ALTER TABLE {quoted_table_name} DROP COLUMN {quoted_column_name}")
        LOG.info("drop %s column statement:\n'%s'", table_class.__table__, drop_column_stmt)

        with Session(self._engine) as session:
            session.execute(drop_column_stmt)
            session.commit()

        # refresh ORM schema to reflect database changes
        self._define_schema(overwrite=False)

    def add_node_attr_key(
        self,
        key_or_schema: str | AttrSchema,
        dtype: pl.DataType | None = None,
        default_value: Any = None,
    ) -> None:
        node_schemas = self.__node_attr_schemas
        # Process arguments and create validated schema
        schema = process_attr_key_args(key_or_schema, dtype, default_value, node_schemas)

        # Add column to database
        self._add_new_column(self.Node, schema)
        node_schemas[schema.key] = schema
        self.__node_attr_schemas = node_schemas

    def remove_node_attr_key(self, key: str) -> None:
        if key not in self.node_attr_keys():
            raise ValueError(f"Node attribute key {key} does not exist")

        if key in (DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.T):
            raise ValueError(f"Cannot remove required node attribute key {key}")

        node_schemas = self.__node_attr_schemas
        self._drop_column(self.Node, key)
        node_schemas.pop(key, None)
        self.__node_attr_schemas = node_schemas

    def add_edge_attr_key(
        self,
        key_or_schema: str | AttrSchema,
        dtype: pl.DataType | None = None,
        default_value: Any = None,
    ) -> None:
        edge_schemas = self.__edge_attr_schemas
        # Process arguments and create validated schema
        schema = process_attr_key_args(key_or_schema, dtype, default_value, edge_schemas)

        # Add column to database
        self._add_new_column(self.Edge, schema)
        edge_schemas[schema.key] = schema
        self.__edge_attr_schemas = edge_schemas

    def remove_edge_attr_key(self, key: str) -> None:
        if key not in self.edge_attr_keys():
            raise ValueError(f"Edge attribute key {key} does not exist")

        edge_schemas = self.__edge_attr_schemas
        self._drop_column(self.Edge, key)
        edge_schemas.pop(key, None)
        self.__edge_attr_schemas = edge_schemas

    def num_edges(self) -> int:
        with Session(self._engine) as session:
            return int(session.query(self.Edge).count())

    def num_nodes(self) -> int:
        with Session(self._engine) as session:
            return int(session.query(self.Node).count())

    def _sql_chunk_size(self) -> int:
        if self._engine.dialect.name == "postgresql":
            chunk_size = 30_000
        else:  # for now everything else will use sqlite chunk size
            # elif self._engine.dialect.name == "sqlite":
            chunk_size = 900

        return chunk_size

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
                if ids is None:
                    query = session.query(table_class)
                    query.update(attrs)
                    session.commit()
                else:
                    # a bit custom and cannot be used with `_chunked_sa_operation`
                    chunk_size = max(1, int(self._sql_chunk_size() / len(attrs)))
                    table_id_attrs = getattr(table_class, id_key)
                    for i in range(0, len(ids), chunk_size):
                        query = session.query(table_class).filter(table_id_attrs.in_(ids[i : i + chunk_size]))
                        query.update(attrs)
                    session.commit()
            return

        if ids is None:
            raise ValueError("`ids` must be provided to update with variable values.")

        for k, v in attrs.items():
            if not np.isscalar(v) and len(attrs[k]) != len(ids):
                raise ValueError(f"Length mismatch: {len(attrs[k])} values for {len(ids)} {id_key}s")

        attrs[id_key] = ids
        # using polars is faster and simpler than iterating in python
        update_data = pl.DataFrame(attrs).to_dicts()

        LOG.info("update %s table with %d rows", table_class.__table__, len(update_data))
        LOG.info("update data sample: %s", update_data[:2])

        self._chunked_sa_write(Session.bulk_update_mappings, update_data, table_class)

    def _chunked_sa_write(
        self,
        session_op: Callable[[Session, type[DeclarativeBase], list[dict[str, Any]]], None],
        data: list[dict[str, Any]],
        table_class: type[DeclarativeBase],
    ) -> None:
        if len(data) == 0:
            return

        # chunked update is faster
        # keep under 1000 values including every column
        chunk_size = max(1, int(self._sql_chunk_size() / len(data[0])))
        with Session(self._engine) as session:
            for i in range(0, len(data), chunk_size):
                session_op(session, table_class, data[i : i + chunk_size])
            session.commit()

    def _chunked_sa_read(
        self,
        session: Session,
        query_filter_op: Callable[[T], Query],
        data: list[T],
        table_class: type[DeclarativeBase],
    ) -> pl.DataFrame:
        """
        Apply a query filter in chunks and concatenate the results.

        Parameters
        ----------
        session : Session
            The SQLAlchemy session.
        query_filter_op : Callable[[T], Query]
            The function to apply a query filter to the data. It must return a SQLAlchemy Query object.
        data : list[T]
            List of data to passed into the query_filter_op function.
        table_class : type[DeclarativeBase]
            The SQLAlchemy table class.

        Examples
        --------
        ```python
        data = [1, 2, 3, 4, 5]
        query_filter_op = lambda x: query.filter(x.id.in_(data))
        data_df = self._chunked_sa_read(session, query_filter_op, data, Node)
        ```

        Returns
        -------
        pl.DataFrame
            The data as a Polars DataFrame.
        """
        if len(data) == 0:
            raise ValueError("Data is empty")

        chunk_size = max(1, self._sql_chunk_size())
        chunks = []
        for i in range(0, len(data), chunk_size):
            query = query_filter_op(data[i : i + chunk_size])
            data_df = pl.read_database(
                query.statement,
                connection=session.connection(),
                schema_overrides=self._polars_schema_override(table_class),
            )
            chunks.append(data_df)
        return pl.concat(chunks)

    def update_node_attrs(
        self,
        *,
        attrs: dict[str, Any],
        node_ids: Sequence[int] | None = None,
    ) -> None:
        if "t" in attrs:
            raise ValueError("Node attribute 't' cannot be updated.")

        updated_node_ids = self.node_ids() if node_ids is None else list(node_ids)
        if len(updated_node_ids) == 0:
            return

        attr_keys = self.node_attr_keys()
        if is_signal_on(self.node_updated):
            old_df = self.filter(node_ids=updated_node_ids).node_attrs(
                attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, *attr_keys]
            )
            old_attrs_by_id = {row[DEFAULT_ATTR_KEYS.NODE_ID]: row for row in old_df.rows(named=True)}

        self._update_table(self.Node, node_ids, DEFAULT_ATTR_KEYS.NODE_ID, attrs)

        if is_signal_on(self.node_updated):
            new_df = self.filter(node_ids=updated_node_ids).node_attrs(
                attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, *attr_keys]
            )
            new_attrs_by_id = {row[DEFAULT_ATTR_KEYS.NODE_ID]: row for row in new_df.rows(named=True)}
            for node_id in updated_node_ids:
                self.node_updated.emit(node_id, old_attrs_by_id[node_id], new_attrs_by_id[node_id])

    def update_edge_attrs(
        self,
        *,
        attrs: dict[str, Any],
        edge_ids: Sequence[int] | None = None,
    ) -> None:
        self._update_table(self.Edge, edge_ids, DEFAULT_ATTR_KEYS.EDGE_ID, attrs)

    def assign_tracklet_ids(
        self,
        output_key: str = DEFAULT_ATTR_KEYS.TRACKLET_ID,
        reset: bool = True,
        tracklet_id_offset: int | None = None,
        node_ids: list[int] | None = None,
        return_id_update: bool = False,
        allow_frame_skip: bool = False,
    ) -> rx.PyDiGraph | tuple[rx.PyDiGraph, pl.DataFrame]:
        if node_ids is not None:
            track_node_ids = list(set(self.tracklet_nodes(node_ids)))
        else:
            track_node_ids = None
        if output_key in self.node_attr_keys():
            node_attr_keys = [output_key]
        else:
            node_attr_keys = []

        return (
            self.filter(node_ids=track_node_ids)
            .subgraph(node_attr_keys=node_attr_keys)
            .assign_tracklet_ids(
                output_key=output_key,
                reset=reset,
                tracklet_id_offset=tracklet_id_offset,
                return_id_update=return_id_update,
                allow_frame_skip=allow_frame_skip,
            )
        )

    def _get_degree(
        self,
        node_ids: list[int] | int | None,
        node_key: str,
    ) -> list[int] | int:
        edge_key_col = getattr(self.Edge, node_key)

        if isinstance(node_ids, int):
            stmt = sa.select(sa.func.count()).where(edge_key_col == node_ids)
            with Session(self._engine) as session:
                return int(session.execute(stmt).scalar())

        stmt = sa.select(edge_key_col, sa.func.count()).group_by(edge_key_col)
        if node_ids is not None:
            stmt = stmt.where(edge_key_col.in_(node_ids))

        with Session(self._engine) as session:
            # get the number of edges for each using group by and count
            degree = dict(session.execute(stmt).all())

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
        for k in ["Base", "Node", "Edge", "Overlap", "Metadata", "_engine"]:
            del data_dict[k]
        return data_dict

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # recreate deleted objects
        self._engine = sa.create_engine(self._url, **self._engine_kwargs)
        self._define_schema(overwrite=False)

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
        """

        if tracklet_id_key not in self.node_attr_keys():
            raise ValueError(f"Track id key '{tracklet_id_key}' not found in graph. Expected '{self.node_attr_keys()}'")

        with Session(self._engine) as session:
            node_query = sa.select(getattr(self.Node, tracklet_id_key)).distinct()

            SourceNode = aliased(self.Node)
            TargetNode = aliased(self.Node)

            edge_query = (
                sa.select(
                    getattr(self.Edge, DEFAULT_ATTR_KEYS.EDGE_SOURCE),
                    getattr(self.Edge, DEFAULT_ATTR_KEYS.EDGE_TARGET),
                )
                .join(
                    SourceNode,
                    SourceNode.node_id == self.Edge.source_id,
                )
                .join(
                    TargetNode,
                    TargetNode.node_id == self.Edge.target_id,
                )
                .filter(
                    getattr(SourceNode, tracklet_id_key) != getattr(TargetNode, tracklet_id_key),
                )
            )

            if ignore_tracklet_id is not None:
                node_query = node_query.filter(getattr(self.Node, tracklet_id_key) != ignore_tracklet_id)
                edge_query = edge_query.filter(
                    getattr(SourceNode, tracklet_id_key) != ignore_tracklet_id,
                    getattr(TargetNode, tracklet_id_key) != ignore_tracklet_id,
                )

            edge_query = edge_query.with_only_columns(
                getattr(SourceNode, tracklet_id_key).label("source_tracklet_id"),
                getattr(TargetNode, tracklet_id_key).label("target_tracklet_id"),
            )

            nodes_df = pl.read_database(
                self._raw_query(node_query),
                connection=session.connection(),
            )

            edges_df = pl.read_database(
                self._raw_query(edge_query),
                connection=session.connection(),
            )

        graph = rx.PyDiGraph()
        tracklet_ids = nodes_df[tracklet_id_key].to_list()
        tracklet_id_to_rx = dict(zip(tracklet_ids, graph.add_nodes_from(tracklet_ids), strict=False))
        graph.add_edges_from(
            zip(
                edges_df["source_tracklet_id"].map_elements(tracklet_id_to_rx.__getitem__, return_dtype=int).to_list(),
                edges_df["target_tracklet_id"].map_elements(tracklet_id_to_rx.__getitem__, return_dtype=int).to_list(),
                zip(edges_df["source_tracklet_id"].to_list(), edges_df["target_tracklet_id"].to_list(), strict=False),
                strict=True,
            )
        )

        return graph

    def has_node(self, node_id: int) -> bool:
        """
        Check if the graph has a node with the given id.
        """
        with Session(self._engine) as session:
            return session.scalar(sa.sql.expression.exists().where(self.Node.node_id == node_id).select())

    def has_edge(self, source_id: int, target_id: int) -> bool:
        """
        Check if the graph has an edge between two nodes.
        """
        with Session(self._engine) as session:
            return (
                session.query(self.Edge)
                .filter(self.Edge.source_id == source_id, self.Edge.target_id == target_id)
                .count()
                > 0
            )

    def edge_id(self, source_id: int, target_id: int) -> int:
        """
        Return the edge id between two nodes.
        """
        with Session(self._engine) as session:
            edge_id = (
                session.query(self.Edge.edge_id)
                .filter(self.Edge.source_id == source_id, self.Edge.target_id == target_id)
                .scalar()
            )
            if edge_id is None:
                raise ValueError(f"Edge {source_id}->{target_id} does not exist in the graph.")
            return edge_id

    def remove_edge(
        self,
        source_id: int | None = None,
        target_id: int | None = None,
        *,
        edge_id: int | None = None,
    ) -> None:
        """
        Remove an edge from the graph either by its ID or by its endpoints.
        """
        with Session(self._engine) as session:
            if edge_id is None:
                if source_id is None or target_id is None:
                    raise ValueError("Provide either edge_id or both source_id and target_id.")
                deleted = (
                    session.query(self.Edge)
                    .filter(self.Edge.source_id == source_id, self.Edge.target_id == target_id)
                    .delete()
                )
                if not deleted:
                    raise ValueError(f"Edge {source_id}->{target_id} does not exist in the graph.")
            else:
                deleted = session.query(self.Edge).filter(self.Edge.edge_id == edge_id).delete()
                if not deleted:
                    raise ValueError(f"Edge {edge_id} does not exist in the graph.")
            session.commit()

    def _metadata(self) -> dict[str, Any]:
        with Session(self._engine) as session:
            result = session.query(self.Metadata).all()
            return {row.key: row.value for row in result}

    def _private_metadata_for_copy(self) -> dict[str, Any]:
        private_metadata = super()._private_metadata_for_copy()
        private_metadata.pop(self._PRIVATE_SQL_NODE_SCHEMA_STORE_KEY, None)
        private_metadata.pop(self._PRIVATE_SQL_EDGE_SCHEMA_STORE_KEY, None)
        return private_metadata

    def _update_metadata(self, **kwargs) -> None:
        with Session(self._engine) as session:
            for key, value in kwargs.items():
                metadata_entry = self.Metadata(key=key, value=value)
                session.merge(metadata_entry)
            session.commit()

    def _remove_metadata(self, key: str) -> None:
        with Session(self._engine) as session:
            session.query(self.Metadata).filter(self.Metadata.key == key).delete()
            session.commit()

    def edge_list(self) -> list[list[int, int]]:
        """
        Get the edge list of the graph.
        """
        with Session(self._engine) as session:
            return [
                [source_id, target_id]
                for source_id, target_id in session.query(self.Edge.source_id, self.Edge.target_id).all()
            ]

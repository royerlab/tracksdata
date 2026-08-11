"""Read-only SQL graph backed lazily by GEFF Zarr arrays."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

import dask.array as da
import numpy as np
import polars as pl
import rustworkx as rx
import sqlalchemy as sa
import xarray as xr
import zarr
from geff_spec import GeffMetadata
from sqlalchemy.orm import DeclarativeBase
from xarray_sql import XarrayContext
from zarr.storage import StoreLike

from tracksdata.attrs import Filter, attr_comps_to_strs, split_attr_comps
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.graph._sql_graph import SQLFilter, SQLGraph
from tracksdata.utils._cache import cache_method
from tracksdata.utils._dataframe import unpack_array_attrs
from tracksdata.utils._dtypes import AttrSchema, polars_dtype_to_sqlalchemy_type
from tracksdata.utils._logging import LOG

if TYPE_CHECKING:
    from tracksdata.graph._graph_view import GraphView

_NODE_ROW = "node_row"
_EDGE_ROW = DEFAULT_ATTR_KEYS.EDGE_ID


def _polars_dtype(dtype: np.dtype[Any]) -> pl.DataType:
    """Return the Polars dtype corresponding to a NumPy dtype."""
    return pl.Series("value", np.empty(0, dtype=dtype)).dtype


def _array_dtype(inner: pl.DataType, shape: tuple[int, ...]) -> pl.Array:
    """Build a fixed-shape Polars Array dtype."""
    return pl.Array(inner, shape)


def _masked_scalar(values: da.Array, missing: da.Array) -> da.Array:
    """Apply a GEFF missing mask without computing either array."""
    kind = values.dtype.kind
    if kind in "biu":
        return da.where(missing, np.nan, values.astype(np.float64))
    if kind in "fc":
        return da.where(missing, np.nan, values)
    if kind in "mM":
        return da.where(missing, np.datetime64("NaT"), values)
    return da.where(missing, None, values.astype(object))


class ZarrSQLFilter(SQLFilter):
    """A :class:`SQLFilter` whose collection is executed by xarray-sql."""

    @cache_method
    def node_ids(self) -> list[int]:
        query = self._graph._select_columns(self._node_query, self._graph.Node, [DEFAULT_ATTR_KEYS.NODE_ID])
        return self._graph._read_database(query, self._graph.Node)[DEFAULT_ATTR_KEYS.NODE_ID].to_list()

    @cache_method
    def edge_ids(self) -> list[int]:
        query = self._graph._select_columns(self._edge_query, self._graph.Edge, [DEFAULT_ATTR_KEYS.EDGE_ID])
        return self._graph._read_database(query, self._graph.Edge)[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()

    @cache_method
    def num_nodes(self) -> int:
        query = sa.select(sa.func.count()).select_from(self._node_query.subquery())
        return int(self._graph._execute(query).item())

    @cache_method
    def num_edges(self) -> int:
        query = sa.select(sa.func.count()).select_from(self._edge_query.subquery())
        return int(self._graph._execute(query).item())

    @cache_method
    def node_attrs(
        self,
        *,
        attr_keys: list[str] | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        result = self._graph._attrs_from_query(self._node_query, self._graph.Node, attr_keys)
        return unpack_array_attrs(result) if unpack else result

    @cache_method
    def edge_attrs(self, attr_keys: list[str] | None = None, unpack: bool = False) -> pl.DataFrame:
        requested = None
        if attr_keys is not None:
            requested = list(
                dict.fromkeys(
                    [
                        *attr_keys,
                        DEFAULT_ATTR_KEYS.EDGE_ID,
                        DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                        DEFAULT_ATTR_KEYS.EDGE_TARGET,
                    ]
                )
            )
        result = self._graph._attrs_from_query(self._edge_query, self._graph.Edge, requested)
        return unpack_array_attrs(result) if unpack else result

    @cache_method
    def subgraph(
        self,
        node_attr_keys: Sequence[str] | None = None,
        edge_attr_keys: Sequence[str] | None = None,
    ) -> GraphView:
        from tracksdata.graph._graph_view import GraphView

        if node_attr_keys is None:
            requested_node_keys = [DEFAULT_ATTR_KEYS.T, *self._graph.node_attr_keys(return_ids=True)]
        else:
            requested_node_keys = [DEFAULT_ATTR_KEYS.T, DEFAULT_ATTR_KEYS.NODE_ID, *node_attr_keys]
        requested_node_keys = list(dict.fromkeys(requested_node_keys))

        if edge_attr_keys is None:
            requested_edge_keys = self._graph.edge_attr_keys(return_ids=True)
        else:
            requested_edge_keys = list(edge_attr_keys)

        nodes_df = self.node_attrs(attr_keys=requested_node_keys)
        edges_df = self.edge_attrs(attr_keys=requested_edge_keys)

        node_map_to_root: dict[int, int] = {}
        node_map_from_root: dict[int, int] = {}
        rx_graph = rx.PyDiGraph()
        for data in nodes_df.iter_rows(named=True):
            root_node_id = data.pop(DEFAULT_ATTR_KEYS.NODE_ID)
            node_id = rx_graph.add_node(data)
            node_map_to_root[node_id] = root_node_id
            node_map_from_root[root_node_id] = node_id

        for data in edges_df.iter_rows(named=True):
            source_root = data.pop(DEFAULT_ATTR_KEYS.EDGE_SOURCE)
            target_root = data.pop(DEFAULT_ATTR_KEYS.EDGE_TARGET)
            if source_root not in node_map_from_root or target_root not in node_map_from_root:
                continue
            rx_graph.add_edge(node_map_from_root[source_root], node_map_from_root[target_root], data)

        return GraphView(
            rx_graph=rx_graph,
            node_map_to_root=node_map_to_root,
            root=self._graph,
            node_attr_keys=requested_node_keys,
            edge_attr_keys=requested_edge_keys,
        )


class ZarrSQLGraph(SQLGraph):
    """Read-only graph that queries scalar GEFF properties through xarray-sql.

    Construction reads only Zarr/GEFF metadata. Array payloads remain Dask-backed
    until a query is collected. Fixed-shape and variable-length properties are
    fetched lazily for only the row positions selected by SQL. Non-scalar properties
    can be returned but cannot themselves be used as SQL filter predicates.
    """

    def __init__(
        self,
        geff_store: StoreLike,
        *,
        chunks: Any = None,
        node_attr_key_map: dict[str, str] | None = None,
        edge_attr_key_map: dict[str, str] | None = None,
    ) -> None:
        BaseGraph.__init__(self)
        self._geff_store = geff_store
        self._chunks = chunks
        self._geff_metadata = GeffMetadata.read(geff_store)
        self._zarr_group = zarr.open_group(geff_store, mode="r")
        self._engine = SimpleNamespace(dialect=sa.engine.default.DefaultDialect())
        self._node_attr_key_map = dict(node_attr_key_map or {})
        self._edge_attr_key_map = dict(edge_attr_key_map or {})
        self._fixed_arrays: dict[Literal["node", "edge"], dict[str, da.Array]] = {"node": {}, "edge": {}}
        self._varlength_arrays: dict[
            Literal["node", "edge"],
            dict[str, tuple[da.Array, zarr.Array]],
        ] = {"node": {}, "edge": {}}
        self._missing_arrays: dict[Literal["node", "edge"], dict[str, da.Array]] = {"node": {}, "edge": {}}
        self._nullable_keys: dict[Literal["node", "edge"], set[str]] = {"node": set(), "edge": set()}

        source_metadata = self._geff_metadata
        node_vars, node_schemas = self._prepare_node_table()
        edge_vars, edge_schemas = self._prepare_edge_table()
        self._node_schemas = node_schemas
        self._edge_schemas = edge_schemas
        if DEFAULT_ATTR_KEYS.T not in self._node_schemas:
            raise ValueError(f"GEFF must expose the required node property '{DEFAULT_ATTR_KEYS.T}'.")
        self._define_models(node_vars, edge_vars)

        # Match BaseGraph.from_geff: key mappings are also reflected in the
        # returned metadata; registered arrays still point at the original
        # on-disk Zarr paths.
        self._geff_metadata = source_metadata.model_copy(deep=True)
        for source_key, key in self._node_attr_key_map.items():
            if source_key in self._geff_metadata.node_props_metadata:
                self._geff_metadata.node_props_metadata[key] = self._geff_metadata.node_props_metadata.pop(source_key)
        for source_key, key in self._edge_attr_key_map.items():
            if source_key in self._geff_metadata.edge_props_metadata:
                self._geff_metadata.edge_props_metadata[key] = self._geff_metadata.edge_props_metadata.pop(source_key)

        node_ds = xr.Dataset({key: ((_NODE_ROW,), value) for key, value in node_vars.items()})
        edge_ds = xr.Dataset({key: ((_EDGE_ROW,), value) for key, value in edge_vars.items()})
        self._context = XarrayContext()
        self._context.from_dataset("node", node_ds, chunks=chunks)
        self._context.from_dataset("edge", edge_ds, chunks=chunks)

        geff_dict = self._geff_metadata.model_dump(mode="json")
        extra = dict(geff_dict.get("extra") or {})
        tracksdata_metadata = dict(extra.pop("tracksdata", {}) or {})
        geff_dict["extra"] = extra
        self._metadata_data = {"geff": geff_dict, **tracksdata_metadata}

    def _prepare_node_table(self) -> tuple[dict[str, da.Array], dict[str, AttrSchema]]:
        node_ids = da.from_zarr(self._zarr_group["nodes/ids"]).astype(np.int64)
        variables = {DEFAULT_ATTR_KEYS.NODE_ID: node_ids}
        schemas: dict[str, AttrSchema] = {DEFAULT_ATTR_KEYS.NODE_ID: AttrSchema(DEFAULT_ATTR_KEYS.NODE_ID, pl.Int64)}
        prop_schemas, prop_vars = self._prepare_properties(
            "node", self._geff_metadata.node_props_metadata, self._node_attr_key_map
        )
        variables.update(prop_vars)
        schemas.update(prop_schemas)
        ordered = [key for key in (DEFAULT_ATTR_KEYS.T, DEFAULT_ATTR_KEYS.NODE_ID) if key in schemas]
        ordered.extend(key for key in schemas if key not in ordered)
        return variables, {key: schemas[key] for key in ordered}

    def _prepare_edge_table(self) -> tuple[dict[str, da.Array], dict[str, AttrSchema]]:
        edge_ids = da.from_zarr(self._zarr_group["edges/ids"]).astype(np.int64)
        variables = {
            DEFAULT_ATTR_KEYS.EDGE_SOURCE: edge_ids[:, 0],
            DEFAULT_ATTR_KEYS.EDGE_TARGET: edge_ids[:, 1],
        }
        schemas: dict[str, AttrSchema] = {
            DEFAULT_ATTR_KEYS.EDGE_ID: AttrSchema(DEFAULT_ATTR_KEYS.EDGE_ID, pl.Int64),
            DEFAULT_ATTR_KEYS.EDGE_SOURCE: AttrSchema(DEFAULT_ATTR_KEYS.EDGE_SOURCE, pl.Int64),
            DEFAULT_ATTR_KEYS.EDGE_TARGET: AttrSchema(DEFAULT_ATTR_KEYS.EDGE_TARGET, pl.Int64),
        }
        prop_schemas, prop_vars = self._prepare_properties(
            "edge", self._geff_metadata.edge_props_metadata, self._edge_attr_key_map
        )
        variables.update(prop_vars)
        schemas.update(prop_schemas)
        return variables, schemas

    def _prepare_properties(
        self,
        mode: Literal["node", "edge"],
        metadata: dict[str, Any],
        key_map: dict[str, str],
    ) -> tuple[dict[str, AttrSchema], dict[str, da.Array]]:
        schemas: dict[str, AttrSchema] = {}
        variables: dict[str, da.Array] = {}
        reserved = (
            {DEFAULT_ATTR_KEYS.NODE_ID, _NODE_ROW}
            if mode == "node"
            else {DEFAULT_ATTR_KEYS.EDGE_ID, DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET}
        )
        for source_key, prop_metadata in metadata.items():
            key = key_map.get(source_key, source_key)
            if key in reserved or key in schemas:
                raise ValueError(f"Mapped {mode} property key '{key}' conflicts with a required column.")
            prop_path = f"{mode}s/props/{source_key}"
            if prop_metadata.varlength:
                values_path = f"{prop_path}/values"
                data_path = f"{prop_path}/data"
                if values_path not in self._zarr_group or data_path not in self._zarr_group:
                    raise ValueError(f"Variable-length GEFF {mode} property '{source_key}' is incomplete.")
                # Object preserves each element's individual ndarray shape.
                schemas[key] = AttrSchema(key, pl.Object)
                self._varlength_arrays[mode][key] = (
                    da.from_zarr(self._zarr_group[values_path]),
                    self._zarr_group[data_path],
                )
                if f"{prop_path}/missing" in self._zarr_group:
                    self._missing_arrays[mode][key] = da.from_zarr(self._zarr_group[f"{prop_path}/missing"]).astype(
                        bool
                    )
                continue
            if f"{prop_path}/values" not in self._zarr_group:
                raise ValueError(f"GEFF {mode} property '{source_key}' has no values array.")
            values = da.from_zarr(self._zarr_group[f"{prop_path}/values"])
            inner = _polars_dtype(np.dtype(prop_metadata.dtype))
            missing_path = f"{prop_path}/missing"
            # Nullable integer/bool has no NumPy representation that both
            # preserves nulls and its exact dtype for Arrow/DataFusion. Defer
            # it instead of coercing through float (which would also lose
            # integer precision above 2**53).
            defer_scalar = values.dtype.kind in "OUS" or (
                values.dtype.kind in "biu" and missing_path in self._zarr_group
            )
            if values.ndim == 1 and not defer_scalar:
                schemas[key] = AttrSchema(key, inner)
                if missing_path in self._zarr_group:
                    missing = da.from_zarr(self._zarr_group[missing_path]).astype(bool)
                    self._missing_arrays[mode][key] = missing
                    self._nullable_keys[mode].add(key)
                    values = _masked_scalar(values, missing)
                variables[key] = values
            elif values.ndim == 1:
                # xarray-sql must inspect object/string variables eagerly to
                # derive their Arrow schema. Keep them outside the SQL table
                # so construction remains metadata-only and fetch selected
                # rows directly from Zarr at collection time.
                schemas[key] = AttrSchema(key, inner)
                self._fixed_arrays[mode][key] = values
                if f"{prop_path}/missing" in self._zarr_group:
                    self._missing_arrays[mode][key] = da.from_zarr(self._zarr_group[f"{prop_path}/missing"]).astype(
                        bool
                    )
            else:
                schemas[key] = AttrSchema(key, _array_dtype(inner, tuple(values.shape[1:])))
                self._fixed_arrays[mode][key] = values
                if f"{prop_path}/missing" in self._zarr_group:
                    self._missing_arrays[mode][key] = da.from_zarr(self._zarr_group[f"{prop_path}/missing"]).astype(
                        bool
                    )
        return schemas, variables

    def _define_models(self, node_vars: dict[str, da.Array], edge_vars: dict[str, da.Array]) -> None:
        class Base(DeclarativeBase):
            pass

        node_columns: dict[str, Any] = {
            "__tablename__": "node",
            _NODE_ROW: sa.Column(sa.BigInteger, primary_key=True),
        }
        node_columns.update(
            {key: sa.Column(polars_dtype_to_sqlalchemy_type(self._node_schemas[key].dtype)) for key in node_vars}
        )
        edge_columns: dict[str, Any] = {
            "__tablename__": "edge",
            DEFAULT_ATTR_KEYS.EDGE_ID: sa.Column(sa.BigInteger, primary_key=True),
        }
        edge_columns.update(
            {key: sa.Column(polars_dtype_to_sqlalchemy_type(self._edge_schemas[key].dtype)) for key in edge_vars}
        )
        self.Base = Base
        self.Node = type("Node", (Base,), node_columns)
        self.Edge = type("Edge", (Base,), edge_columns)

    @classmethod
    def from_geff(
        cls,
        geff_store: StoreLike,
        geff_read_kwargs: dict[str, Any] | None = None,
        node_attr_key_map: dict[str, str] | None = None,
        edge_attr_key_map: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[ZarrSQLGraph, GeffMetadata]:
        if geff_read_kwargs:
            raise ValueError("ZarrSQLGraph.from_geff does not support nonempty geff_read_kwargs.")
        graph = cls(
            geff_store,
            node_attr_key_map=node_attr_key_map,
            edge_attr_key_map=edge_attr_key_map,
            **kwargs,
        )
        return graph, graph._geff_metadata

    def supports_custom_indices(self) -> bool:
        return False

    def _sql_chunk_size(self) -> int:
        return 2**60

    def _raw_query(self, query: sa.Select) -> str:
        raw_query = str(query.compile(compile_kwargs={"literal_binds": True}))
        LOG.info("xarray-sql query:\n%s", raw_query)
        return raw_query

    def _execute(self, query: sa.Select) -> pl.DataFrame:
        arrow = self._context.sql(self._raw_query(query)).to_arrow_table()
        return pl.from_arrow(arrow)

    def _read_database(
        self,
        query: sa.Select,
        table_class: type[DeclarativeBase],
        connection: Any = None,
    ) -> pl.DataFrame:
        del connection
        df = self._execute(query)
        mode = self._mode_for_table(table_class)
        schemas = self._node_schemas if mode == "node" else self._edge_schemas
        expressions: list[pl.Expr] = []
        for key, schema in schemas.items():
            if key not in df.columns or isinstance(schema.dtype, pl.Array | pl.List):
                continue
            expr = pl.col(key)
            if key in self._nullable_keys[mode] and df.schema[key].is_float():
                expr = expr.fill_nan(None)
            expressions.append(expr.cast(schema.dtype, strict=False))
        return df.with_columns(expressions) if expressions else df

    def _select_columns(
        self,
        query: sa.Select,
        table: type[DeclarativeBase],
        names: Sequence[str],
    ) -> sa.Select:
        if isinstance(query, sa.CompoundSelect):
            alias = query.alias("selected_rows")
            return sa.select(*[getattr(alias.c, name) for name in names])
        return query.with_only_columns(*[getattr(table, name) for name in names])

    def _attrs_from_query(
        self,
        query: sa.Select,
        table: type[DeclarativeBase],
        attr_keys: Sequence[str] | None,
    ) -> pl.DataFrame:
        mode = self._mode_for_table(table)
        schemas = self._node_schemas if mode == "node" else self._edge_schemas
        requested = list(schemas) if attr_keys is None else list(dict.fromkeys(attr_keys))
        self._validate_attr_keys(requested, mode)
        deferred = [key for key in requested if key in self._fixed_arrays[mode] or key in self._varlength_arrays[mode]]
        sql_names = [key for key in requested if hasattr(table, key)]
        row_key = _NODE_ROW if mode == "node" else _EDGE_ROW
        if deferred and row_key not in sql_names:
            sql_names.append(row_key)
        selected_query = self._select_columns(query, table, sql_names)
        df = self._read_database(selected_query, table)
        if deferred:
            rows = df[row_key].to_numpy().astype(np.int64, copy=False)
            df = self._add_deferred_properties(df, mode, deferred, rows)
        if row_key not in requested and row_key in df.columns:
            df = df.drop(row_key)
        return df.select([key for key in requested if key in df.columns])

    def _add_deferred_properties(
        self,
        df: pl.DataFrame,
        mode: Literal["node", "edge"],
        keys: Sequence[str],
        rows: np.ndarray,
    ) -> pl.DataFrame:
        """Add non-SQL properties for selected physical GEFF rows."""
        for key in keys:
            if key in self._fixed_arrays[mode]:
                series = self._read_fixed_property(mode, key, rows)
            else:
                series = self._read_varlength_property(mode, key, rows)
            df = df.with_columns(series)

        if mode == "node" and DEFAULT_ATTR_KEYS.MASK in keys:
            bbox_key = DEFAULT_ATTR_KEYS.BBOX
            if bbox_key not in self._node_schemas:
                raise ValueError("A GEFF mask property requires a matching bbox property.")
            if bbox_key in df.columns:
                bboxes = df[bbox_key].to_list()
            elif bbox_key in self._fixed_arrays["node"]:
                bboxes = self._read_fixed_property("node", bbox_key, rows).to_list()
            elif bbox_key in self._varlength_arrays["node"]:
                bboxes = self._read_varlength_property("node", bbox_key, rows).to_list()
            else:
                raise ValueError("The GEFF bbox property is not readable as an array.")

            from tracksdata.nodes._mask import Mask

            masks = df[DEFAULT_ATTR_KEYS.MASK].to_list()
            df = df.with_columns(
                pl.Series(
                    DEFAULT_ATTR_KEYS.MASK,
                    [
                        None if mask is None else Mask(np.asarray(mask, dtype=bool), bbox)
                        for mask, bbox in zip(masks, bboxes, strict=True)
                    ],
                    dtype=pl.Object,
                )
            )
        return df

    def _read_fixed_property(
        self,
        mode: Literal["node", "edge"],
        key: str,
        rows: np.ndarray,
    ) -> pl.Series:
        dtype = (self._node_schemas if mode == "node" else self._edge_schemas)[key].dtype
        if len(rows) == 0:
            return pl.Series(key, [], dtype=dtype)
        values = np.asarray(self._fixed_arrays[mode][key][rows].compute())
        missing_array = self._missing_arrays[mode].get(key)
        if missing_array is None:
            return pl.Series(key, values, dtype=dtype)
        missing = np.asarray(missing_array[rows].compute()).astype(bool)
        if missing.ndim > 1:
            missing = missing.reshape(len(rows), -1).all(axis=1)
        data = [None if is_missing else value for value, is_missing in zip(values, missing, strict=True)]
        return pl.Series(key, data, dtype=dtype)

    def _read_varlength_property(
        self,
        mode: Literal["node", "edge"],
        key: str,
        rows: np.ndarray,
    ) -> pl.Series:
        """Deserialize selected GEFF variable-length values without a full scan."""
        if len(rows) == 0:
            return pl.Series(key, [], dtype=pl.Object)

        encoded_array, data_array = self._varlength_arrays[mode][key]
        encoded = np.asarray(encoded_array[rows].compute(), dtype=np.uint64)
        missing_array = self._missing_arrays[mode].get(key)
        missing = (
            np.zeros(len(rows), dtype=bool)
            if missing_array is None
            else np.asarray(missing_array[rows].compute(), dtype=bool)
        )
        values: list[np.ndarray | None] = []
        for encoded_value, is_missing in zip(encoded, missing, strict=True):
            if is_missing:
                values.append(None)
                continue
            offset = int(encoded_value[0])
            shape = tuple(int(value) for value in encoded_value[1:])
            size = int(np.prod(shape, dtype=np.int64))
            values.append(np.asarray(data_array[offset : offset + size]).reshape(shape))
        return pl.Series(key, values, dtype=pl.Object)

    def filter(
        self,
        *attr_filters: Filter,
        node_ids: Sequence[int] | None = None,
        include_targets: bool = False,
        include_sources: bool = False,
    ) -> ZarrSQLFilter:
        node_comps, edge_comps = split_attr_comps(attr_filters)
        unsupported = [
            key
            for mode, comps in (("node", node_comps), ("edge", edge_comps))
            for key in attr_comps_to_strs(comps)
            if key in self._fixed_arrays[mode] or key in self._varlength_arrays[mode]
        ]
        if unsupported:
            raise NotImplementedError(f"Non-scalar GEFF properties cannot be used in SQL filters: {unsupported}")
        return ZarrSQLFilter(
            *attr_filters,
            graph=self,
            node_ids=node_ids,
            include_targets=include_targets,
            include_sources=include_sources,
        )

    def node_ids(self) -> list[int]:
        return self.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID])[DEFAULT_ATTR_KEYS.NODE_ID].to_list()

    def edge_ids(self) -> list[int]:
        return self.edge_attrs(attr_keys=[])[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()

    def time_points(self) -> list[int]:
        query = sa.select(self.Node.t).distinct()
        return self._read_database(query, self.Node)[DEFAULT_ATTR_KEYS.T].to_list()

    def node_attrs(
        self,
        *,
        attr_keys: Sequence[str] | str | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        if isinstance(attr_keys, str):
            attr_keys = [attr_keys]
        result = self._attrs_from_query(sa.select(self.Node), self.Node, attr_keys)
        return unpack_array_attrs(result) if unpack else result

    def edge_attrs(
        self,
        *,
        attr_keys: Sequence[str] | str | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        if isinstance(attr_keys, str):
            attr_keys = [attr_keys]
        requested = None
        if attr_keys is not None:
            requested = list(
                dict.fromkeys(
                    [
                        *attr_keys,
                        DEFAULT_ATTR_KEYS.EDGE_ID,
                        DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                        DEFAULT_ATTR_KEYS.EDGE_TARGET,
                    ]
                )
            )
        result = self._attrs_from_query(sa.select(self.Edge), self.Edge, requested)
        return unpack_array_attrs(result) if unpack else result

    def _node_attr_schemas(self) -> dict[str, AttrSchema]:
        return self._node_schemas

    def _edge_attr_schemas(self) -> dict[str, AttrSchema]:
        return self._edge_schemas

    def _get_neighbors(
        self,
        node_key: str,
        neighbor_key: str,
        node_ids: list[int] | int | None,
        attr_keys: Sequence[str] | str | None = None,
        *,
        return_attrs: bool = False,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]:
        single = isinstance(node_ids, int)
        requested_ids = self.node_ids() if node_ids is None else ([node_ids] if single else list(node_ids))
        if isinstance(attr_keys, str):
            attr_keys = [attr_keys]
        if return_attrs:
            requested_attrs = list(self._node_schemas) if attr_keys is None else list(attr_keys)
            self._validate_attr_keys(requested_attrs, "node")
        else:
            requested_attrs = [DEFAULT_ATTR_KEYS.NODE_ID]
        deferred = [
            key for key in requested_attrs if key in self._fixed_arrays["node"] or key in self._varlength_arrays["node"]
        ]
        columns = [
            getattr(self.Edge, node_key),
            *[getattr(self.Node, key) for key in requested_attrs if hasattr(self.Node, key)],
        ]
        if deferred:
            columns.append(getattr(self.Node, _NODE_ROW))
        query = sa.select(*columns).join(self.Edge, getattr(self.Edge, neighbor_key) == self.Node.node_id)
        if node_ids is not None:
            query = query.where(getattr(self.Edge, node_key).in_(requested_ids))
        df = self._read_database(query, self.Node)
        if deferred:
            rows = df[_NODE_ROW].to_numpy().astype(np.int64, copy=False)
            df = self._add_deferred_properties(df, "node", deferred, rows).drop(_NODE_ROW)
        if single:
            return df.drop(node_key) if return_attrs else df[DEFAULT_ATTR_KEYS.NODE_ID].to_list()
        if return_attrs:
            groups = {int(key[0]): group.drop(node_key) for key, group in df.group_by(node_key)}
            empty = df.drop(node_key).clear()
            return {node_id: groups.get(node_id, empty) for node_id in requested_ids}
        groups = df.select(node_key, DEFAULT_ATTR_KEYS.NODE_ID).rows_by_key(node_key)
        return {node_id: [row[0] for row in groups.get(node_id, [])] for node_id in requested_ids}

    def successors(
        self,
        node_ids: list[int] | int | None,
        attr_keys: Sequence[str] | str | None = None,
        *,
        return_attrs: bool = False,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]:
        return self._get_neighbors(
            DEFAULT_ATTR_KEYS.EDGE_SOURCE,
            DEFAULT_ATTR_KEYS.EDGE_TARGET,
            node_ids,
            attr_keys,
            return_attrs=return_attrs,
        )

    def predecessors(
        self,
        node_ids: list[int] | int | None,
        attr_keys: Sequence[str] | str | None = None,
        *,
        return_attrs: bool = False,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]:
        return self._get_neighbors(
            DEFAULT_ATTR_KEYS.EDGE_TARGET,
            DEFAULT_ATTR_KEYS.EDGE_SOURCE,
            node_ids,
            attr_keys,
            return_attrs=return_attrs,
        )

    def _get_degree(self, node_ids: list[int] | int | None, node_key: str) -> list[int] | int:
        column = getattr(self.Edge, node_key)
        query = sa.select(column, sa.func.count().label("degree")).group_by(column)
        requested = (
            self.node_ids() if node_ids is None else ([node_ids] if isinstance(node_ids, int) else list(node_ids))
        )
        if node_ids is not None:
            query = query.where(column.in_(requested))
        result = self._execute(query)
        degrees = dict(result.iter_rows())
        values = [int(degrees.get(node_id, 0)) for node_id in requested]
        return values[0] if isinstance(node_ids, int) else values

    def in_degree(self, node_ids: list[int] | int | None = None) -> list[int] | int:
        return self._get_degree(node_ids, DEFAULT_ATTR_KEYS.EDGE_TARGET)

    def out_degree(self, node_ids: list[int] | int | None = None) -> list[int] | int:
        return self._get_degree(node_ids, DEFAULT_ATTR_KEYS.EDGE_SOURCE)

    def dividing_nodes(self) -> list[int]:
        column = self.Edge.source_id
        query = sa.select(column).group_by(column).having(sa.func.count() == 2)
        return self._execute(query)[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()

    def num_nodes(self) -> int:
        return int(self._execute(sa.select(sa.func.count()).select_from(self.Node)).item())

    def num_edges(self) -> int:
        return int(self._execute(sa.select(sa.func.count()).select_from(self.Edge)).item())

    def tracklet_graph(
        self,
        tracklet_id_key: str = DEFAULT_ATTR_KEYS.TRACKLET_ID,
        ignore_tracklet_id: int | None = None,
    ) -> rx.PyDiGraph:
        """Create a tracklet graph through the backend-independent query path."""
        return BaseGraph.tracklet_graph(self, tracklet_id_key, ignore_tracklet_id)

    def has_node(self, node_id: int) -> bool:
        query = sa.select(sa.func.count()).select_from(self.Node).where(self.Node.node_id == node_id)
        return bool(self._execute(query).item())

    def has_edge(self, source_id: int, target_id: int) -> bool:
        query = (
            sa.select(sa.func.count())
            .select_from(self.Edge)
            .where(self.Edge.source_id == source_id, self.Edge.target_id == target_id)
        )
        return bool(self._execute(query).item())

    def edge_id(self, source_id: int, target_id: int) -> int:
        query = sa.select(self.Edge.edge_id).where(
            self.Edge.source_id == source_id,
            self.Edge.target_id == target_id,
        )
        result = self._execute(query)
        if result.height == 0:
            raise ValueError(f"Edge {source_id}->{target_id} does not exist in the graph.")
        return int(result.item(0, 0))

    def edge_list(self) -> list[list[int]]:
        query = sa.select(self.Edge.source_id, self.Edge.target_id)
        return [list(row) for row in self._execute(query).iter_rows()]

    def overlaps(self, node_ids: list[int] | None = None) -> list[list[int]]:
        del node_ids
        return []

    def has_overlaps(self) -> bool:
        return False

    def _metadata(self) -> dict[str, Any]:
        return dict(self._metadata_data)

    def __getstate__(self) -> dict[str, Any]:
        """Persist only constructor inputs; query contexts are rebuilt."""
        return {
            "geff_store": self._geff_store,
            "chunks": self._chunks,
            "node_attr_key_map": self._node_attr_key_map,
            "edge_attr_key_map": self._edge_attr_key_map,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(**state)

    @staticmethod
    def _read_only(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("ZarrSQLGraph is read-only; the GEFF source cannot be modified.")

    bulk_add_nodes = _read_only
    bulk_remove_nodes = _read_only
    bulk_add_edges = _read_only
    bulk_remove_edges = _read_only
    add_overlap = _read_only
    bulk_add_overlaps = _read_only
    add_node_attr_key = _read_only
    remove_node_attr_key = _read_only
    add_edge_attr_key = _read_only
    remove_edge_attr_key = _read_only
    update_node_attrs = _read_only
    update_edge_attrs = _read_only
    create_node_attr_index = _read_only
    create_edge_attr_index = _read_only
    drop_node_attr_index = _read_only
    drop_edge_attr_index = _read_only
    assign_tracklet_ids = _read_only
    _update_metadata = _read_only
    _remove_metadata = _read_only

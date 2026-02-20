from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import sqlalchemy as sa
from cloudpickle import dumps, loads
from sqlalchemy.sql.type_api import TypeEngine

_POLARS_DTYPE_TO_NUMPY_DTYPE = {
    pl.Datetime: np.datetime64,
    pl.Boolean: np.bool_,
    pl.Float16: np.float16,
    pl.Float32: np.float32,
    pl.Float64: np.float64,
    pl.Int8: np.int8,
    pl.Int16: np.int16,
    pl.Int32: np.int32,
    pl.Int64: np.int64,
    pl.Duration: np.timedelta64,
    pl.UInt8: np.uint8,
    pl.UInt16: np.uint16,
    pl.UInt32: np.uint32,
    pl.UInt64: np.uint64,
}

_COMPATIBILITY_CONVERSION = {np.float16: np.float32}


def polars_dtype_to_numpy_dtype(
    polars_dtype: pl.DataType,
    allow_sequence: bool = True,
    compatibility: bool = False,
) -> np.dtype:
    """Convert a polars dtype to a numpy dtype.

    Parameters
    ----------
    polars_dtype : DataType
        The polars dtype to convert.
    allow_sequence : bool
        Whether to allow sequence types (List, Array). Default is True.
    compatibility : bool
        Return numpy dtype in compatibility mode, avoiding exotic data types (e.g. np.float16)

    Returns
    -------
    np.dtype
        The numpy dtype.
    """
    while isinstance(polars_dtype, pl.Array | pl.List):
        if not allow_sequence:
            raise ValueError(f"Sequence types are not allowed: {polars_dtype}. Set allow_sequence=True to allow.")
        polars_dtype = polars_dtype.inner

    try:
        np_dtype = _POLARS_DTYPE_TO_NUMPY_DTYPE[polars_dtype]
    except KeyError as e:
        raise ValueError(
            f"Invalid polars dtype: {polars_dtype}. Expected one of {_POLARS_DTYPE_TO_NUMPY_DTYPE.keys()}"
        ) from e

    if compatibility:
        np_dtype = _COMPATIBILITY_CONVERSION.get(np_dtype, np_dtype)

    return np_dtype


def column_to_bytes(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Convert a column of a DataFrame to bytes.
    Used to serialize columns for multiprocessing.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to convert.
    column : str
        The column to convert.

    Returns
    -------
    pl.DataFrame
        The converted DataFrame.
    """
    return df.with_columns(pl.col(column).map_elements(dumps, return_dtype=pl.Binary))


def column_from_bytes(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Convert a column of a DataFrame from bytes.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to convert.
    column : str
        The column to convert.

    Returns
    -------
    pl.DataFrame
        The converted DataFrame.
    """
    return df.with_columns(pl.col(column).map_elements(loads, return_dtype=pl.Object))


def column_to_numpy(series: pl.Series) -> np.ndarray:
    """
    Helper function to convert a polars series to a numpy array.
    It handles the case where the series is a binary column.

    Parameters
    ----------
    series : pl.Series
        The series to convert.

    Returns
    -------
    np.ndarray
        The converted numpy array.
    """
    if series.dtype == pl.Binary:
        return np.asarray(series.to_list())
    else:
        return series.to_numpy()


@dataclass
class AttrSchema:
    """
    Schema information for a graph attribute.

    Stores both the polars dtype and the default value for an attribute key.
    This is used to maintain consistent type information across graph operations
    and when converting to polars DataFrames.

    Parameters
    ----------
    key : str
        The attribute key name.
    dtype : pl.DataType
        The polars data type for this attribute.
    default_value : Any, optional
        The default value for this attribute. If None, will be inferred from dtype.

    Examples
    --------
    Create a schema with inferred default:

    ```python
    schema = AttrSchema(key="count", dtype=pl.UInt32)
    # default_value will be 0
    ```

    Create a schema with custom default:

    ```python
    schema = AttrSchema(key="score", dtype=pl.Float64, default_value=-99.0)
    ```

    Create an array schema:

    ```python
    schema = AttrSchema(key="bbox", dtype=pl.Array(pl.Float64, 4))
    # default_value will be np.zeros(4, dtype=np.float64)
    ```
    """

    key: str
    dtype: pl.DataType
    default_value: Any = None

    def __post_init__(self):
        """Infer default value if not provided and validate compatibility."""
        if self.default_value is None:
            self.default_value = infer_default_value_from_dtype(self.dtype)
        else:
            validate_default_value_dtype_compatibility(self.default_value, self.dtype)

    def copy(self) -> AttrSchema:
        """
        Create a defensive copy of this AttrSchema.

        Returns
        -------
        AttrSchema
            A new AttrSchema instance with the same key, dtype, and default_value.

        Examples
        --------
        ```python
        original = AttrSchema(key="score", dtype=pl.Float64, default_value=1.0)
        copied = original.copy()

        # Mutating original doesn't affect the copy
        original.default_value = 999.0
        assert copied.default_value == 1.0
        ```
        """
        return AttrSchema(key=self.key, dtype=self.dtype, default_value=self.default_value)


def process_attr_key_args(
    key_or_schema: str | AttrSchema,
    dtype: pl.DataType | None,
    default_value: Any,
    attr_schemas: dict[str, AttrSchema],
) -> AttrSchema:
    """
    Process arguments for add_node_attr_key/add_edge_attr_key and return a validated schema.

    This helper function handles both calling patterns (convenience and schema mode),
    validates arguments, and ensures the key doesn't already exist.

    Parameters
    ----------
    key_or_schema : str | AttrSchema
        Either a key name or an AttrSchema object.
    dtype : pl.DataType | None
        The polars data type (required when key_or_schema is a string).
    default_value : Any
        The default value (will be inferred from dtype if None).
    attr_schemas : dict[str, AttrSchema]
        The dictionary of existing attribute schemas (for duplicate check).

    Returns
    -------
    AttrSchema
        A validated AttrSchema ready to be stored.

    Raises
    ------
    TypeError
        If dtype is not provided when using string key.
    ValueError
        If the key already exists or if default_value and dtype are incompatible.

    Examples
    --------
    ```python
    # Convenience mode
    schema = process_attr_key_args("count", pl.UInt32, None, {})
    assert schema.default_value == 0

    # Schema mode
    original = AttrSchema(key="score", dtype=pl.Float64, default_value=1.0)
    schema = process_attr_key_args(original, None, None, {})
    assert schema is not original  # Defensive copy
    ```
    """
    # Handle both calling patterns
    if isinstance(key_or_schema, AttrSchema):
        # Schema mode: create a defensive copy to avoid mutation bugs
        schema = key_or_schema.copy()
        key = schema.key
    else:
        # Convenience mode: build schema from parameters
        key = key_or_schema
        if dtype is None:
            raise TypeError("dtype is required when not using AttrSchema")

        # Determine default_value if not provided
        if default_value is None:
            default_value = infer_default_value_from_dtype(dtype)
        else:
            # Validate compatibility if both provided
            validate_default_value_dtype_compatibility(default_value, dtype)

        # Create schema
        schema = AttrSchema(key=key, dtype=dtype, default_value=default_value)

    # Check key doesn't exist
    if key in attr_schemas:
        raise ValueError(f"Attribute key {key} already exists")

    return schema


# Default value mapping for polars dtypes
DTYPE_DEFAULT_MAP = {
    pl.Boolean: False,
    pl.UInt8: 0,
    pl.UInt16: 0,
    pl.UInt32: 0,
    pl.UInt64: 0,
    pl.Int8: -1,
    pl.Int16: -1,
    pl.Int32: -1,
    pl.Int64: -1,
    pl.Float16: -1.0,
    pl.Float32: -1.0,
    pl.Float64: -1.0,
    pl.String: "",
    pl.Utf8: "",
}


def infer_default_value_from_dtype(dtype: pl.DataType) -> Any:
    """
    Infer a sensible default value from a polars dtype.

    Parameters
    ----------
    dtype : pl.DataType
        The polars data type.

    Returns
    -------
    Any
        A sensible default value for the type.

    Examples
    --------
    >>> infer_default_value_from_dtype(pl.Int64)
    -1
    >>> infer_default_value_from_dtype(pl.UInt32)
    0
    >>> infer_default_value_from_dtype(pl.Boolean)
    False
    >>> infer_default_value_from_dtype(pl.Array(pl.Float64, 3))
    array([0., 0., 0.])
    """
    # Handle array types - create zeros with correct shape and dtype
    if isinstance(dtype, pl.Array):
        inner_dtype = dtype.inner
        numpy_dtype = polars_dtype_to_numpy_dtype(inner_dtype, allow_sequence=True)
        return np.zeros(dtype.shape, dtype=numpy_dtype)

    # Handle list types
    if isinstance(dtype, pl.List):
        return []

    # Use dictionary lookup for standard types
    return DTYPE_DEFAULT_MAP.get(dtype, None)


# SQLAlchemy type mapping for polars dtypes
_POLARS_TO_SQLALCHEMY_TYPE_MAP = {
    # Boolean
    pl.Boolean: sa.Boolean,
    # Small integer types
    pl.Int8: sa.SmallInteger,
    pl.UInt8: sa.SmallInteger,
    pl.Int16: sa.SmallInteger,
    pl.UInt16: sa.SmallInteger,
    # Integer types
    pl.Int32: sa.Integer,
    pl.UInt32: sa.Integer,
    # Big integer types
    pl.Int64: sa.BigInteger,
    pl.UInt64: sa.BigInteger,
    # Float types
    pl.Float16: sa.Float,
    pl.Float32: sa.Float,
    pl.Float64: sa.Float,
    # String types
    pl.String: sa.String,
    pl.Utf8: sa.String,
}


def polars_dtype_to_sqlalchemy_type(dtype: pl.DataType) -> TypeEngine:
    """
    Convert a polars dtype to SQLAlchemy type.

    Parameters
    ----------
    dtype : pl.DataType
        The polars data type.

    Returns
    -------
    sa.TypeEngine
        The corresponding SQLAlchemy type.

    Examples
    --------
    >>> polars_dtype_to_sqlalchemy_type(pl.Int64)
    <class 'sqlalchemy.sql.sqltypes.BigInteger'>
    >>> polars_dtype_to_sqlalchemy_type(pl.Boolean)
    <class 'sqlalchemy.sql.sqltypes.Boolean'>
    """
    # Handle struct types as JSON for backend-level field filtering.
    if isinstance(dtype, pl.Struct):
        return sa.JSON()

    # Handle sequence types - use PickleType for storage
    if isinstance(dtype, pl.Array | pl.List):
        return sa.PickleType()

    # Use dictionary lookup for standard types
    sa_type_class = _POLARS_TO_SQLALCHEMY_TYPE_MAP.get(dtype)
    if sa_type_class is not None:
        return sa_type_class()

    # Object and fallback
    return sa.PickleType()


# SQLAlchemy to polars type mapping for schema loading
# Order matters: more specific types must come before more general types
# (e.g., BigInteger before Integer, since BigInteger is a subclass of Integer)
_SQLALCHEMY_TO_POLARS_TYPE_MAP = [
    (sa.Boolean, pl.Boolean),
    (sa.BigInteger, pl.Int64),  # Must come before Integer
    (sa.SmallInteger, pl.Int16),  # Must come before Integer
    (sa.Integer, pl.Int32),
    (sa.Float, pl.Float64),
    (sa.Text, pl.String),  # Must come before String
    (sa.String, pl.String),
    (sa.JSON, pl.Object),
    (sa.PickleType, pl.Object),  # Must come before LargeBinary
    (sa.LargeBinary, pl.Object),
]


def sqlalchemy_type_to_polars_dtype(sa_type: TypeEngine) -> pl.DataType:
    """
    Convert a SQLAlchemy type to a polars dtype.

    This is a best-effort conversion for loading existing database schemas.

    Parameters
    ----------
    sa_type : TypeEngine
        The SQLAlchemy type.

    Returns
    -------
    pl.DataType
        The corresponding polars dtype.

    Examples
    --------
    >>> sqlalchemy_type_to_polars_dtype(sa.BigInteger())
    Int64
    >>> sqlalchemy_type_to_polars_dtype(sa.Boolean())
    Boolean
    """
    # Check the type map for known types
    # Order matters: more specific types are checked first
    for sa_type_class, pl_dtype in _SQLALCHEMY_TO_POLARS_TYPE_MAP:
        if isinstance(sa_type, sa_type_class):
            return pl_dtype

    # Fallback to Object for unknown types
    return pl.Object


def validate_default_value_dtype_compatibility(default_value: Any, dtype: pl.DataType) -> None:
    """
    Validate that a default value is compatible with a polars dtype.

    Parameters
    ----------
    default_value : Any
        The default value to validate.
    dtype : pl.DataType
        The polars dtype to validate against.

    Raises
    ------
    ValueError
        If the default value is incompatible with the dtype.

    Examples
    --------
    >>> validate_default_value_dtype_compatibility(42, pl.Int64)
    # No error

    >>> validate_default_value_dtype_compatibility("string", pl.Int64)
    ValueError: default_value 'string' (type: str) is incompatible with dtype Int64...
    """
    # Skip validation for Object and Binary types - they accept any value
    if dtype in (pl.Object, pl.Binary):
        return

    try:
        # Try to create a polars series and cast
        pl.Series([default_value], dtype=dtype)
    except Exception as e:
        raise ValueError(
            f"default_value {default_value!r} (type: {type(default_value).__name__}) "
            f"is incompatible with dtype {dtype}. "
            f"Cannot cast to specified type. Error: {e}"
        ) from e

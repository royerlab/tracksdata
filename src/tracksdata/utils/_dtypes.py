from typing import Any

import blosc2
import numpy as np
import polars as pl
from cloudpickle import dumps, loads
from polars.datatypes.classes import (
    Boolean,
    DataType,
    Datetime,
    Duration,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)

from tracksdata.utils._logging import LOG

_POLARS_DTYPE_TO_NUMPY_DTYPE = {
    Datetime: np.datetime64,
    Boolean: np.bool_,
    Float32: np.float32,
    Float64: np.float64,
    Int8: np.int8,
    Int16: np.int16,
    Int32: np.int32,
    Int64: np.int64,
    Duration: np.timedelta64,
    UInt8: np.uint8,
    UInt16: np.uint16,
    UInt32: np.uint32,
    UInt64: np.uint64,
}


def polars_dtype_to_numpy_dtype(polars_dtype: DataType) -> np.dtype:
    """Convert a polars dtype to a numpy dtype.

    Parameters
    ----------
    polars_dtype : DataType
        The polars dtype to convert.

    Returns
    -------
    np.dtype
        The numpy dtype.
    """
    try:
        return _POLARS_DTYPE_TO_NUMPY_DTYPE[polars_dtype]
    except KeyError as e:
        raise ValueError(
            f"Invalid polars dtype: {polars_dtype}. Expected one of {_POLARS_DTYPE_TO_NUMPY_DTYPE.keys()}"
        ) from e


def _try_packing_numpy_array(x: Any) -> bytes:
    if isinstance(x, np.ndarray):
        packed = blosc2.pack_array2(x)
    else:
        packed = dumps(x)
    return packed


def _try_unpacking_numpy_array(x: bytes) -> Any:
    try:
        unpacked = blosc2.unpack_array2(x)
    except Exception:
        unpacked = loads(x)

    return unpacked


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
    prev_nthreads = blosc2.set_nthreads(1)
    df = df.with_columns(pl.col(column).map_elements(_try_packing_numpy_array, return_dtype=pl.Binary))
    blosc2.set_nthreads(prev_nthreads)
    return df


def column_from_bytes(df: pl.DataFrame, column: str | None = None) -> pl.DataFrame:
    """
    Convert a column of a DataFrame from bytes.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to convert.
    column : str | None
        The column to convert. If not provided, all pl.Binary columns will be converted.

    Returns
    -------
    pl.DataFrame
        The converted DataFrame.
    """
    # This function used to be simple
    # but polars sometimes was failing to serialize numpy arrays

    if column is None:
        columns = [c for c, d in zip(df.columns, df.dtypes, strict=False) if d == pl.Binary]
    else:
        columns = [column]  # if column in df.columns and df[column].dtype == pl.Binary else []

    # If no binary columns found, return as-is (data already deserialized)
    if not columns:
        return df

    prev_nthreads = blosc2.set_nthreads(1)
    for c in columns:
        # Always use Object dtype to avoid issues with heterogeneous array shapes/types
        try:
            df = df.with_columns(pl.col(c).map_elements(_try_unpacking_numpy_array, return_dtype=pl.Object))
        except Exception:
            # If there's an issue with map_elements (e.g., polars conversion errors),
            # fall back to manual conversion
            LOG.warning(f"Error in map_elements for column {c}, falling back to manual conversion")
            values = [_try_unpacking_numpy_array(val) for val in df[c].to_list()]
            df = df.with_columns(pl.Series(name=c, values=values, dtype=pl.Object))

    blosc2.set_nthreads(prev_nthreads)
    return df

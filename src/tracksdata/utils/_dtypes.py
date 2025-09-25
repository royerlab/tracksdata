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
    if isinstance(polars_dtype, pl.Array | pl.List):
        polars_dtype = polars_dtype.inner

    try:
        return _POLARS_DTYPE_TO_NUMPY_DTYPE[polars_dtype]
    except KeyError as e:
        raise ValueError(
            f"Invalid polars dtype: {polars_dtype}. Expected one of {_POLARS_DTYPE_TO_NUMPY_DTYPE.keys()}"
        ) from e


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

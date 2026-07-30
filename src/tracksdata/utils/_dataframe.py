from collections.abc import Mapping

import cloudpickle
import polars as pl
import polars.selectors as cs


def unpack_array_attrs(df: pl.DataFrame) -> pl.DataFrame:
    """
    Unpack array attributesinto a dictionary, convert array columns into multiple scalar columns.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with array attributes.

    Returns
    -------
    pl.DataFrame
        DataFrame with unpacked array attributes.
    """

    array_cols = [name for name, dtype in df.schema.items() if isinstance(dtype, pl.Array)]

    if len(array_cols) == 0:
        return df

    for col in array_cols:
        df = df.with_columns(pl.col(col).arr.to_struct(lambda x: f"{col}_{x}")).unnest(col)  # noqa: B023

    return unpack_array_attrs(df)


def unpickle_bytes_columns(
    df: pl.DataFrame,
    dtypes: Mapping[str, pl.DataType] | None = None,
) -> pl.DataFrame:
    """
    Unpickle bytes columns from the database.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to unpickle the bytes columns from.
    dtypes : Mapping[str, pl.DataType] | None
        Declared dtype per column, used to build the unpickled columns.
        Columns without a declared dtype fall back to polars' inference, which
        only looks at the leading rows and therefore silently truncates a
        `Float64` column whose first rows happen to hold whole numbers.

    Returns
    -------
    pl.DataFrame
        The DataFrame with the bytes columns unpickled.
    """
    if dtypes is None:
        dtypes = {}

    df = df.map_columns(cs.binary(), lambda x: x.map_elements(cloudpickle.loads, return_dtype=pl.Object))
    for col, dtype in zip(df.columns, df.dtypes, strict=True):
        if not isinstance(dtype, pl.Object):
            continue
        values = df[col].to_list()
        # `None` falls back to polars' inference, either because the column has no
        # declared dtype or because its values turned out not to fit it.
        candidates = (dtypes[col], None) if col in dtypes else (None,)
        for target_dtype in candidates:
            try:
                df = df.with_columns(pl.Series(col, values, dtype=target_dtype))
                break
            except Exception:
                # values that fit neither the declared dtype nor an inferred one
                # (e.g. `Mask` objects) are left as an object column.
                pass
    return df

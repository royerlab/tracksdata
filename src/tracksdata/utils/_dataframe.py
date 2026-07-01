from collections.abc import Collection

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


def unpickle_columns(df: pl.DataFrame, columns: Collection[str]) -> pl.DataFrame:
    """
    Unpickle pickled bytes columns read from the database.

    Only the columns in *columns* are unpickled. Raw-binary columns (e.g. the
    blosc2-compressed ``data`` leaf of a Mask struct attribute) are stored
    natively as ``pl.Binary`` and must be left untouched, so callers pass the
    explicit set of genuinely-pickled physical columns rather than relying on
    all binary columns being pickled.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to unpickle the bytes columns from.
    columns : Collection[str]
        The physical column names that hold pickled values.

    Returns
    -------
    pl.DataFrame
        The DataFrame with the pickled columns unpickled.
    """
    # `columns` lists columns that are *defined* as pickled (SQL ``PickleType``),
    # but the runtime dtype is inferred per query result since pickle columns are
    # excluded from the polars schema override. A genuinely-pickled column can
    # therefore come back as something other than ``pl.Binary`` (e.g. an all-NULL
    # result is inferred as ``pl.Null``). Restrict to actual binary columns so
    # ``cloudpickle.loads`` is only ever applied to real bytes.
    targets = [col for col in columns if col in df.columns and df.schema[col] == pl.Binary]
    if not targets:
        return df

    df = df.map_columns(cs.by_name(targets), lambda x: x.map_elements(cloudpickle.loads, return_dtype=pl.Object))
    for col in targets:
        if isinstance(df.schema[col], pl.Object):
            try:
                df = df.with_columns(pl.Series(df[col].to_list()).alias(col))
            except Exception:
                pass
    return df

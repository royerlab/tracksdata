from collections.abc import Collection

import cloudpickle
import polars as pl


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

    # `map_elements` converts a returned dict into polars-native containers even with
    # `return_dtype=pl.Object`, turning e.g. a `Mask`'s numpy arrays into python lists.
    # Building the object series directly keeps the unpickled values untouched.
    df = df.with_columns(
        pl.Series(col, [None if v is None else cloudpickle.loads(v) for v in df[col]], dtype=pl.Object)
        for col in targets
    )
    for col in targets:
        if isinstance(df.schema[col], pl.Object):
            try:
                inferred = pl.Series(df[col].to_list())
            except Exception:
                continue
            # Dict values (e.g. a `Mask`) infer as a struct, and polars converts the
            # numpy arrays they hold into nested python lists. Keep those as objects.
            if isinstance(inferred.dtype, pl.Struct):
                continue
            df = df.with_columns(inferred.alias(col))
    return df

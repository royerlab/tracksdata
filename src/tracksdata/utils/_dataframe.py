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


def unpickle_bytes_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Unpickle bytes columns from the database.

    The result is left as :class:`polars.Object`. Every caller pairs this with
    ``SQLGraph._cast_columns``, which casts each pickled column to its declared
    schema dtype, so narrowing the dtype here would only build an intermediate
    that is immediately rebuilt -- and for a column of opaque payloads (masks)
    the attempt materializes every value just to fail and be discarded.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to unpickle the bytes columns from.

    Returns
    -------
    pl.DataFrame
        The DataFrame with the bytes columns unpickled.
    """
    binary_cols = [name for name, dtype in df.schema.items() if dtype == pl.Binary]
    if not binary_cols:
        return df

    # A plain comprehension rather than `map_elements`: these are opaque Python
    # objects either way, so routing them through the expression engine adds
    # per-element dispatch without buying any vectorization.
    return df.with_columns(
        pl.Series(
            name,
            [None if value is None else cloudpickle.loads(value) for value in df[name]],
            dtype=pl.Object,
        )
        for name in binary_cols
    )

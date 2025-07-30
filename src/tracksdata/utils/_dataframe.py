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

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to unpickle the bytes columns from.

    Returns
    -------
    pl.DataFrame
        The DataFrame with the bytes columns unpickled.
    """
    df = df.with_columns(pl.col(pl.Binary).map_elements(cloudpickle.loads, return_dtype=pl.Object))
    for col, dtype in zip(df.columns, df.dtypes, strict=True):
        if isinstance(dtype, pl.Object):
            try:
                df = df.with_columns(pl.Series(df[col].to_list()).alias(col))
            except Exception:
                pass
    return df

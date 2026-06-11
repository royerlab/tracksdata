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


def unpickle_bytes_columns(
    df: pl.DataFrame,
    schema_overrides: dict[str, pl.DataType] | None = None,
) -> pl.DataFrame:
    """
    Unpickle bytes columns from the database.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to unpickle the bytes columns from.
    schema_overrides : dict[str, pl.DataType] | None, optional
        Optional mapping from column name to target polars dtype. When a name
        matches a binary column, the decoded values are placed directly into a
        Series of that dtype, skipping the intermediate ``pl.Object`` rebuild.

    Returns
    -------
    pl.DataFrame
        The DataFrame with the bytes columns unpickled.
    """
    if schema_overrides is None:
        schema_overrides = {}

    new_series: list[pl.Series] = []
    for name, dtype in zip(df.columns, df.dtypes, strict=True):
        if dtype != pl.Binary:
            continue
        raw = df[name].to_list()
        decoded = [None if v is None else cloudpickle.loads(v) for v in raw]
        target = schema_overrides.get(name)
        try:
            new_series.append(pl.Series(name, decoded, dtype=target))
        except Exception:
            new_series.append(pl.Series(name, decoded, dtype=pl.Object))

    if new_series:
        df = df.with_columns(new_series)
    return df

import cloudpickle
import numpy as np
import polars as pl

from tracksdata.utils._dataframe import unpack_array_attrs, unpickle_bytes_columns


def test_unpack_array_attrs() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "attributes": np.asarray([[0.1, 0.2], [1.1, 1.2], [2.1, 2.2]]),
            "other": np.asarray([[[3.1, 3.2]], [[4.1, 4.2]], [[5.1, 5.2]]]),
        }
    )

    unpackaged_df = unpack_array_attrs(df)

    expected_df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "attributes_0": [0.1, 1.1, 2.1],
            "attributes_1": [0.2, 1.2, 2.2],
            "other_0_0": [3.1, 4.1, 5.1],
            "other_0_1": [3.2, 4.2, 5.2],
        }
    )

    assert np.all(unpackaged_df.columns == expected_df.columns)

    np.testing.assert_array_equal(
        unpackaged_df.to_numpy(),
        expected_df.to_numpy(),
    )


def test_unpickle_bytes_columns_variable_size_arrays() -> None:
    """Regression: unpickling a binary column with variable-size numpy arrays must not crash.

    This reproduces the production bug triggered by GEFF import: masks are stored as raw
    numpy arrays (not Mask objects) after a zarr roundtrip, so different nodes have arrays
    of different shapes. Without return_dtype=pl.Object, polars infers the type from the
    first array and raises SchemaError on the next differently-shaped one.
    """
    arrays = [np.ones((41, 41), dtype=bool), np.ones((4, 4), dtype=bool)]
    df = pl.DataFrame(
        {"mask": pl.Series([cloudpickle.dumps(a) for a in arrays], dtype=pl.Binary)}
    )

    result = unpickle_bytes_columns(df)  # must not raise SchemaError

    for actual, expected in zip(result["mask"].to_list(), arrays):
        np.testing.assert_array_equal(actual, expected)

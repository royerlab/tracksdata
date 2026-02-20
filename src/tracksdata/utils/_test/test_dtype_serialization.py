import base64
import binascii
import io

import numpy as np
import polars as pl
import pytest

from tracksdata.utils._dtypes import (
    AttrSchema,
    deserialize_attr_schema,
    serialize_attr_schema,
)


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int64,
        pl.Float32,
        pl.Boolean,
        pl.String,
        pl.List(pl.Int16),
        pl.Array(pl.Float64, 4),
        pl.Array(pl.Int32, (2, 3)),
        pl.Struct({"x": pl.Int64, "y": pl.List(pl.String)}),
        pl.Datetime("us", "UTC"),
    ],
)
def test_serialize_deserialize_attr_schema_dtype_roundtrip(dtype: pl.DataType) -> None:
    schema = AttrSchema(key="dummy", dtype=dtype)
    encoded = serialize_attr_schema(schema)

    assert isinstance(encoded, str)
    assert encoded
    assert base64.b64decode(encoded)

    restored = deserialize_attr_schema(encoded, key=schema.key)

    assert restored == schema


def test_deserialize_attr_schema_invalid_base64_raises() -> None:
    with pytest.raises(binascii.Error):
        deserialize_attr_schema("not-base64", key="dummy")


def test_deserialize_attr_schema_non_ipc_payload_raises() -> None:
    encoded = base64.b64encode(b"not-arrow-ipc").decode("utf-8")

    with pytest.raises((OSError, pl.exceptions.PolarsError)):
        deserialize_attr_schema(encoded, key="dummy")


@pytest.mark.parametrize(
    "schema",
    [
        AttrSchema(key="score", dtype=pl.Float64, default_value=1.25),
        AttrSchema(
            key="vector",
            dtype=pl.Array(pl.Float32, 3),
            default_value=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        ),
        AttrSchema(key="payload", dtype=pl.Object, default_value={"nested": [1, 2, 3]}),
    ],
)
def test_serialize_deserialize_attr_schema_roundtrip(schema: AttrSchema) -> None:
    encoded = serialize_attr_schema(schema)
    restored = deserialize_attr_schema(encoded, key=schema.key)
    assert restored == schema


def test_serialize_attr_schema_stores_default_in_dummy_row() -> None:
    schema = AttrSchema(key="score", dtype=pl.Float64, default_value=1.25)
    encoded = serialize_attr_schema(schema)

    payload = base64.b64decode(encoded)
    df = pl.read_ipc(io.BytesIO(payload))

    assert "__attr_schema_value__" in df.columns
    assert df.schema["__attr_schema_value__"] == pl.Float64
    assert df["__attr_schema_value__"][0] == 1.25
    assert "__attr_schema_dtype_pickle__" not in df.columns

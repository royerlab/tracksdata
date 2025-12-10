import numpy as np
import pytest

from tracksdata.utils._dtypes import infer_default_value


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        (True, False),
        (42, -1),
        (3.14, -1.0),
        (np.uint8(5), 0),
        (np.int32(7), -1),
        (np.float32(3.14), -1.0),
        ("foo", None),
    ],
)
def test_infer_default_value(sample: object, expected: object) -> None:
    assert infer_default_value(sample) == expected

import pytest

from tracksdata.options import options_context
from tracksdata.utils._multiprocessing import multiprocessing_apply


def _square(x: int) -> int:
    return x * x


@pytest.mark.parametrize("n_workers", [1, 2])
def test_multiprocessing_apply_empty_sequence(n_workers: int) -> None:
    """An empty sequence must be a no-op regardless of the worker count."""
    with options_context(n_workers=n_workers):
        assert list(multiprocessing_apply(_square, [], desc="empty")) == []


@pytest.mark.parametrize("n_workers", [1, 2])
def test_multiprocessing_apply_results(n_workers: int) -> None:
    with options_context(n_workers=n_workers):
        results = sorted(multiprocessing_apply(_square, [1, 2, 3], desc="squares"))
    assert results == [1, 4, 9]

import numpy as np
import pytest
from tracksdata.array._nd_chunk_cache import NDChunkCache

def simple_compute_func(t: int, chunk_slices: tuple[slice, ...], buffer: np.ndarray) -> None:
    """Dummy I/O-heavy function."""
    mesh = np.meshgrid(
        *[np.arange(s.start, s.stop) for s in chunk_slices], 
        indexing='ij'
    )
    buffer[chunk_slices] = t*1e6+sum([100**(i)*mesh[i] for i in range(len(mesh))])
    
def test_nd_chunk_cache_checks_dim():
    cache = NDChunkCache(
        compute_func=simple_compute_func,
        shape=(256, 256, 256),
        chunk_shape=(64, 64, 64),
        max_buffers=2,
    )
    assert cache.ndim == 3
    assert cache.shape == (256, 256, 256)
    assert cache.chunk_shape == (64, 64, 64)
    with pytest.raises(ValueError):
        NDChunkCache(
            compute_func=simple_compute_func,
            shape=(256, 256),
            chunk_shape=(64, 64, 64),
            max_buffers=2,
        )


def test_nd_chunk_cache_data_integrity_non_divisible():
    """Test that data is computed correctly for non-divisible edge chunks."""
    cache = NDChunkCache(
        compute_func=simple_compute_func,
        shape=(5,),
        chunk_shape=(2,),
        max_buffers=2
    )
    
    # Get full array
    result = cache.get(0, (slice(0, 5),))
    
    # Verify specific values to ensure correct computation
    # Element at index 4 should have value: 0 * 1000 + 4 = 4
    assert result[4] == 4.0, f"Expected value 4.0 at index 4, got {result[4]}"
    
    # Test consistency - getting the same slice twice should return identical data
    partial1 = cache.get(0, (slice(3, 5),))
    partial2 = cache.get(0, (slice(3, 5),))
    np.testing.assert_array_equal(partial1, partial2, "Data should be consistent")


@pytest.mark.parametrize("shape,chunk_size", [
    ((10,), (3,)),
    ((7, 5), (3, 2)),
    ((8, 6, 4), (3, 2, 3)),
    ((5,), (2,)),
    ((11, 7), (4, 3)),
    ((1,), (2,)),
    ((3, 1), (2, 3)),
])
def test_nd_chunk_cache_various_non_divisible(shape, chunk_size):
    """Parametrized test for various non-divisible shape combinations."""
    cache = NDChunkCache(
        compute_func=simple_compute_func,
        shape=shape,
        chunk_shape=chunk_size,
        max_buffers=2
    )
    
    # Grid shape should use ceiling division
    expected_grid_shape = tuple((fs + cs - 1) // cs for fs, cs in zip(shape, chunk_size))
    assert cache.grid_shape == expected_grid_shape, \
        f"Expected grid_shape {expected_grid_shape}, got {cache.grid_shape}"
    
    # Should be able to access full array
    full_slices = tuple(slice(0, fs) for fs in shape)
    result = cache.get(0, full_slices)
    assert result.shape == shape, f"Expected shape {shape}, got {result.shape}"
    
    # Should be able to access edge elements
    edge_slices = tuple(slice(fs - 1, fs) for fs in shape)
    edge_result = cache.get(0, edge_slices)
    expected_edge_shape = tuple(1 for _ in shape)
    assert edge_result.shape == expected_edge_shape, \
        f"Expected edge shape {expected_edge_shape}, got {edge_result.shape}"


@pytest.fixture
def array_nd_chunk_cache():
    max_size=100
    time_count = 2
    shape=(max_size, max_size, max_size)
    chunk_size=(64,64,64)

    np_array = np.empty([time_count]+list(shape))
    for t in range(time_count):
        simple_compute_func(t, (slice(0, max_size), slice(0, max_size), slice(0, max_size)), np_array[t])

    # 3-D data example: full 256×256×256 volume, 64×64×64 chunks
    cache = NDChunkCache(
        compute_func=simple_compute_func,
        shape=shape,
        chunk_shape=chunk_size,
        max_buffers=3,
    )
    return cache, np_array

@pytest.mark.parametrize("volume_slicing", 
    [
        (slice(10, 50), slice(0, 20), slice(30, 45)),
        (slice(6, 30), slice(10, 22), slice(0, 80)),
        (slice(0, 100), 50, slice(0, 30)),
        (99, 50, slice(0, 30)),
        (19, 50, 45),
    ]
)
def test_nd_chunk_cache_correctly_slice(array_nd_chunk_cache, volume_slicing):
    cache, np_array = array_nd_chunk_cache

    # First request → triggers chunk computations
    vol1 = cache.get(1, volume_slicing)
    np.testing.assert_array_equal(vol1, np_array[1][volume_slicing],
                                  "First request should match original data")
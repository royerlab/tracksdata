"""Tests for the options system."""

import pytest

from tracksdata.options import Options, get_options, options_context, set_options


def test_default_options() -> None:
    """Test that default options are created with expected values."""
    options = Options()
    assert options.show_progress is True
    assert options.n_workers == 1


def test_get_options() -> None:
    """Test getting the global options."""
    options = get_options()
    assert isinstance(options, Options)
    assert options.show_progress is True


def test_set_options() -> None:
    """Test setting the global options."""
    original_options = get_options()

    new_options = Options(show_progress=False)
    set_options(new_options)

    assert get_options().show_progress is False

    # Restore original
    set_options(original_options)
    assert get_options().show_progress is True


def test_set_options_with_kwargs() -> None:
    """Test setting options using keyword arguments."""
    original_options = get_options()

    # Set options using kwargs
    set_options(show_progress=False, n_workers=4)
    assert get_options().show_progress is False
    assert get_options().n_workers == 4

    # Restore original
    set_options(original_options)
    assert get_options().show_progress is True
    assert get_options().n_workers == 1


def test_set_options_error_both_provided() -> None:
    """Test that providing both options and kwargs raises an error."""

    with pytest.raises(ValueError, match="Cannot provide both 'options' and keyword arguments"):
        set_options(Options(show_progress=False), show_progress=True)


def test_set_options_error_neither_provided() -> None:
    """Test that providing neither options nor kwargs raises an error."""

    with pytest.raises(ValueError, match="Must provide either 'options' or keyword arguments"):
        set_options()


def test_options_context_manager() -> None:
    """Test using Options as a context manager."""
    original_show_progress = get_options().show_progress

    with Options(show_progress=False):
        assert get_options().show_progress is False

    # Should restore original value
    assert get_options().show_progress == original_show_progress


def test_options_context_manager_nested() -> None:
    """Test nested context managers."""
    original_show_progress = get_options().show_progress

    with Options(show_progress=False):
        assert get_options().show_progress is False

        with Options(show_progress=True):
            assert get_options().show_progress is True

        # Should restore to outer context
        assert get_options().show_progress is False

    # Should restore original value
    assert get_options().show_progress == original_show_progress


def test_options_context_function() -> None:
    """Test using the options_context function."""
    original_show_progress = get_options().show_progress

    with options_context(show_progress=False):
        assert get_options().show_progress is False

    # Should restore original value
    assert get_options().show_progress == original_show_progress


def test_options_context_function_nested() -> None:
    """Test nested options_context function calls."""
    original_show_progress = get_options().show_progress

    with options_context(show_progress=False):
        assert get_options().show_progress is False

        with options_context(show_progress=True):
            assert get_options().show_progress is True

        # Should restore to outer context
        assert get_options().show_progress is False

    # Should restore original value
    assert get_options().show_progress == original_show_progress


def test_options_context_exception() -> None:
    """Test that options are restored even if an exception occurs."""
    original_show_progress = get_options().show_progress

    try:
        with Options(show_progress=False):
            assert get_options().show_progress is False
            raise ValueError("Test exception")
    except ValueError:
        pass

    # Should restore original value even after exception
    assert get_options().show_progress == original_show_progress

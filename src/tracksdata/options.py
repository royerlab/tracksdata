"""Global options system for TracksData."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

# Module-private mutable state
_options_stack: list["Options"] = []


@dataclass
class Options:
    """
    Global options for TracksData.

    This class provides a centralized way to control various behaviors
    across the library, such as progress display and multiprocessing.

    Parameters
    ----------
    show_progress : bool, default True
        Whether to display progress bars during operations.
    n_workers : int
        Number of worker processes to use for multiprocessing operations.
        - 0 or 1: use default behavior (sequential)
        - > 1: use exactly this many worker processes
        NOTE: Overhead of multiprocessing is significant, experiment with 1 before increasing.
    """

    show_progress: bool = True
    n_workers: int = 1

    def __enter__(self) -> "Options":
        """Enter the context manager."""
        # Push current options to stack
        _options_stack.append(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager."""
        # Pop this options from stack
        if _options_stack:
            _options_stack.pop()

    def update(self, **kwargs: Any) -> None:
        """Update the options with the given keyword arguments."""
        valid_keys = set(self.__dict__.keys())
        for key, value in kwargs.items():
            if key not in valid_keys:
                raise ValueError(f"Invalid option: {key}. Expected one of {valid_keys}")
            setattr(self, key, value)

    def copy(self) -> "Options":
        """Return a copy of the options."""
        return Options(**self.__dict__)


# Default options - always at the bottom of the stack
_default_options = Options()


def get_options() -> Options:
    """
    Get the current global options.

    Returns
    -------
    Options
        The current global options instance.
    """
    # Return the top of the stack, or default if stack is empty
    return _options_stack[-1] if _options_stack else _default_options


def set_options(options: Options | None = None, **kwargs: Any) -> None:
    """
    Set the global options pushing a new Options object to the stack.

    Parameters
    ----------
    options : Options | None, optional
        The options object to set as global. If None, kwargs will be used to create a new Options object.
    **kwargs : Any
        Individual option parameters to set. Only used if options is None.

    Examples
    --------
    Set options using an Options object:

    >>> from tracksdata.options import Options, set_options
    >>> set_options(Options(show_progress=False))

    Set options using keyword arguments:

    >>> set_options(show_progress=False)

    Raises
    ------
    ValueError
        If both options and kwargs are provided, or if neither are provided.
    """
    if options is not None and kwargs:
        raise ValueError("Cannot provide both 'options' and keyword arguments")

    if options is None and not kwargs:
        raise ValueError("Must provide either 'options' or keyword arguments")

    if options is None:
        options = get_options().copy()
        options.update(**kwargs)

    _options_stack.append(options)


@contextmanager
def options_context(**kwargs: Any) -> Generator[Options, None, None]:
    """
    Context manager for temporarily modifying options.

    Parameters
    ----------
    **kwargs : Any
        Options parameters to temporarily set.

    Yields
    ------
    Options
        The options object with the temporary settings.

    Examples
    --------
    ```python
    from tracksdata.options import options_context

    with options_context(show_progress=False):
        # Operations here will not show progress
        pass
    ```

    See Also
    --------
    [Options][tracksdata.options.Options]:
        The global options class.

    """
    temp_options = Options(**kwargs)
    with temp_options:
        yield temp_options

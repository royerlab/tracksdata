import multiprocessing as mp
from collections.abc import Callable, Generator, Sequence
from multiprocessing import connection
from typing import TypeVar

import cloudpickle
from tqdm import tqdm

from tracksdata.options import get_options

R = TypeVar("R")
T = TypeVar("T")

"""
Configures multiprocessing to use multiprocessing for pickling.
This allows function to be pickled.
Reference: https://stackoverflow.com/a/69253561/6748803
"""
cloudpickle.Pickler.dumps, cloudpickle.Pickler.loads = (
    cloudpickle.dumps,
    cloudpickle.loads,
)
connection._ForkingPickler = cloudpickle.Pickler


def multiprocessing_apply(
    func: Callable[[T], R],
    sequence: Sequence[T],
    desc: str | None = None,
    sorted: bool = False,
) -> Generator[R, None, None]:
    """Applies `func` for each item in `sequence`.

    Parameters
    ----------
    func : Callable[[T], R]
        Function to be executed.
    sequence : Sequence[T]
        Sequence of parameters.
    desc : Optional[str], optional
        Description to tqdm progress bar, by default None
    sorted : bool, optional
        Whether to keep the order of the returned results.
        Sorted output is a bit slower.

    Returns
    -------
    Generator[R, None, None]
        Generator of `func` outputs.
    """
    length = len(sequence)
    options = get_options()
    disable_tqdm = not options.show_progress

    if length == 1:
        # skipping iteration overhead
        yield func(sequence[0])

    elif options.n_workers > 1:
        ctx = mp.get_context("spawn")
        chunksize = max(1, length // (options.n_workers * 2))
        with ctx.Pool(min(options.n_workers, length)) as pool:
            map_func = pool.imap if sorted else pool.imap_unordered
            for result in tqdm(
                map_func(func, sequence, chunksize=chunksize),
                desc=desc,
                total=length,
                disable=disable_tqdm,
            ):
                yield result

    else:
        for result in tqdm(sequence, desc=desc, disable=disable_tqdm):
            yield func(result)

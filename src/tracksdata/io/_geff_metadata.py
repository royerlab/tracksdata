"""Read tracksdata graph metadata from a geff store without building a graph.

`BaseGraph.to_geff` writes the graph metadata (`graph.metadata`) into the geff
metadata extras, and `BaseGraph.from_geff` hoists it back onto the graph. Callers
that need a metadata value *before* they have a graph object -- for example the
`shape` of the dense segmentation, which is required to construct a
`GraphArrayView` -- cannot use either. `read_graph_metadata` closes that gap so
downstream libraries do not have to know where tracksdata stores the extras.
"""

from __future__ import annotations

from typing import Any

from geff_spec import GeffMetadata
from zarr.storage import StoreLike

from tracksdata.graph._base_graph import BaseGraph

__all__ = ["read_graph_metadata"]

_EXTRA_KEY = "tracksdata"


def read_graph_metadata(source: StoreLike | GeffMetadata) -> dict[str, Any]:
    """
    Read the tracksdata graph metadata of a geff dataset without loading the graph.

    Returns the same metadata that `graph.metadata` would hold after
    `BaseGraph.from_geff`, minus the `geff` key. Note that the values went through a
    JSON round-trip, so tuples come back as lists.

    Parameters
    ----------
    source : StoreLike | GeffMetadata
        The store or path of the geff dataset, or an already parsed `GeffMetadata`.

    Returns
    -------
    dict[str, Any]
        The graph metadata, empty if the dataset was not written by tracksdata.

    Examples
    --------
    ```python
    graph.metadata["shape"] = (5, 100, 100)
    graph.to_geff("tracks.geff")

    shape = read_graph_metadata("tracks.geff")["shape"]  # [5, 100, 100]
    ```
    """
    if not isinstance(source, GeffMetadata):
        source = GeffMetadata.read(source)

    metadata = source.extra.get(_EXTRA_KEY, {})

    return {k: v for k, v in metadata.items() if not BaseGraph._is_private_metadata_key(k)}

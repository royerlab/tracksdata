"""
Array representation of graphical data.

Provides read-only array views of graph attributes, with lazy loading from original data sources.
"""

from tracksdata.array._graph_array import GraphArrayView

__all__ = ["GraphArrayView"]

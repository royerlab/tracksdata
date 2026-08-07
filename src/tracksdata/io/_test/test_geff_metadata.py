from pathlib import Path

import polars as pl
from geff_spec import Axis, GeffMetadata, PropMetadata
from zarr.storage import MemoryStore

from tracksdata.graph import RustWorkXGraph
from tracksdata.io import read_graph_metadata

SHAPE = (5, 100, 100)


def _make_graph() -> RustWorkXGraph:
    graph = RustWorkXGraph()
    graph.add_node_attr_key("y", pl.Float64)
    graph.add_node_attr_key("x", pl.Float64)
    graph.add_node({"t": 0, "y": 1.0, "x": 2.0})
    graph.add_node({"t": 1, "y": 3.0, "x": 4.0})
    graph.metadata["shape"] = SHAPE
    return graph


def _minimal_geff_metadata() -> GeffMetadata:
    """A `GeffMetadata` as a downstream library would build it: no tracksdata extras."""
    return GeffMetadata(
        directed=True,
        axes=[
            Axis(name="t", type="time"),
            Axis(name="y", type="space", scale=0.5),
            Axis(name="x", type="space", scale=0.5),
        ],
        node_props_metadata={
            "t": PropMetadata(identifier="t", dtype="int64"),
            "y": PropMetadata(identifier="y", dtype="float64"),
            "x": PropMetadata(identifier="x", dtype="float64"),
        },
        edge_props_metadata={},
        extra={"downstream": {"hello": "world"}},
    )


def test_read_graph_metadata_from_path(tmp_path: Path) -> None:
    """The shape is readable from a store path, without building a graph."""
    graph = _make_graph()
    geff_path = tmp_path / "tracks.geff"
    graph.to_geff(geff_store=geff_path)

    # tuples become lists through the JSON round-trip
    assert read_graph_metadata(geff_path) == {"shape": list(SHAPE)}


def test_read_graph_metadata_custom_geff_metadata() -> None:
    """The shape survives a write with caller-supplied metadata and is readable back."""
    graph = _make_graph()
    store = MemoryStore()
    graph.to_geff(geff_store=store, geff_metadata=_minimal_geff_metadata())

    assert read_graph_metadata(store) == {"shape": list(SHAPE)}


def test_read_graph_metadata_from_geff_metadata_instance() -> None:
    """An already parsed `GeffMetadata` is accepted, so the store is not reopened."""
    graph = _make_graph()
    store = MemoryStore()
    graph.to_geff(geff_store=store)

    assert read_graph_metadata(GeffMetadata.read(store)) == {"shape": list(SHAPE)}


def test_read_graph_metadata_without_tracksdata_extras() -> None:
    """A geff not written by tracksdata yields an empty dict rather than raising."""
    # foreign extras only
    assert read_graph_metadata(_minimal_geff_metadata()) == {}

    no_extra = _minimal_geff_metadata()
    no_extra.extra = {}
    assert read_graph_metadata(no_extra) == {}


def test_read_graph_metadata_excludes_private_keys() -> None:
    """Private metadata is written to the store but not exposed by the reader."""
    graph = _make_graph()
    graph._private_metadata["__private_secret"] = 42

    store = MemoryStore()
    graph.to_geff(geff_store=store)

    assert "__private_secret" in GeffMetadata.read(store).extra["tracksdata"]
    assert read_graph_metadata(store) == {"shape": list(SHAPE)}

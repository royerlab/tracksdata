"""Tests for dtype parameter in add_node_attr_key and add_edge_attr_key."""

import numpy as np
import polars as pl
import pytest

from tracksdata.graph import RustWorkXGraph
from tracksdata.utils._dtypes import AttrSchema


class TestRustWorkXGraphDtype:
    """Test dtype functionality in RustWorkXGraph."""

    def test_add_node_attr_key_with_dtype_only(self):
        """Test adding node attribute with dtype only (default value inferred)."""
        graph = RustWorkXGraph()

        # Add various dtypes
        graph.add_node_attr_key("count", pl.UInt32)
        graph.add_node_attr_key("score", pl.Float64)
        graph.add_node_attr_key("flag", pl.Boolean)
        graph.add_node_attr_key("name", pl.String)

        # Verify schemas are stored
        assert "count" in graph._node_attr_schemas
        assert graph._node_attr_schemas["count"].dtype == pl.UInt32
        assert graph._node_attr_schemas["count"].default_value == 0

        assert graph._node_attr_schemas["score"].dtype == pl.Float64
        assert graph._node_attr_schemas["score"].default_value == -1.0

        assert graph._node_attr_schemas["flag"].dtype == pl.Boolean
        assert graph._node_attr_schemas["flag"].default_value is False

        assert graph._node_attr_schemas["name"].dtype == pl.String
        assert graph._node_attr_schemas["name"].default_value == ""

    def test_add_node_attr_key_with_dtype_and_default(self):
        """Test adding node attribute with both dtype and default value."""
        graph = RustWorkXGraph()

        graph.add_node_attr_key("score", pl.Float64, default_value=0.0)

        assert graph._node_attr_schemas["score"].dtype == pl.Float64
        assert graph._node_attr_schemas["score"].default_value == 0.0

    def test_add_node_attr_key_with_array_dtype(self):
        """Test adding node attribute with array dtype (zeros default)."""
        graph = RustWorkXGraph()

        graph.add_node_attr_key("bbox", pl.Array(pl.Float64, 4))

        assert graph._node_attr_schemas["bbox"].dtype == pl.Array(pl.Float64, 4)
        default = graph._node_attr_schemas["bbox"].default_value
        assert isinstance(default, np.ndarray)
        assert default.shape == (4,)
        assert default.dtype == np.float64
        np.testing.assert_array_equal(default, np.zeros(4, dtype=np.float64))

    def test_add_node_attr_key_with_nd_array_dtype(self):
        """Test adding node attribute with ndarray dtype."""
        graph = RustWorkXGraph()
        graph.add_node_attr_key("something", pl.Array(pl.Float64, (5, 3, 2)))

        assert graph._node_attr_schemas["something"].dtype == pl.Array(pl.Float64, (5, 3, 2))
        default = graph._node_attr_schemas["something"].default_value

        assert isinstance(default, np.ndarray)
        assert default.shape == (5, 3, 2)
        assert default.dtype == np.float64
        np.testing.assert_array_equal(default, np.zeros((5, 3, 2), dtype=np.float64))

    def test_add_node_attr_key_with_schema_object(self):
        """Test adding node attribute using AttrSchema object."""
        graph = RustWorkXGraph()

        schema = AttrSchema(key="intensity", dtype=pl.Float64)
        graph.add_node_attr_key(schema)

        assert "intensity" in graph._node_attr_schemas
        assert graph._node_attr_schemas["intensity"].dtype == pl.Float64
        assert graph._node_attr_schemas["intensity"].default_value == -1.0

    def test_add_node_attr_key_missing_dtype_raises(self):
        """Test that missing dtype raises TypeError."""
        graph = RustWorkXGraph()

        with pytest.raises(TypeError, match="dtype is required"):
            graph.add_node_attr_key("score")

    def test_add_node_attr_key_duplicate_raises(self):
        """Test that adding duplicate key raises ValueError."""
        graph = RustWorkXGraph()

        graph.add_node_attr_key("score", pl.Float64)

        with pytest.raises(ValueError, match="already exists"):
            graph.add_node_attr_key("score", pl.Float64)

    def test_add_node_attr_key_incompatible_default_raises(self):
        """Test that incompatible dtype and default raises ValueError."""
        graph = RustWorkXGraph()

        with pytest.raises(ValueError, match="incompatible"):
            graph.add_node_attr_key("score", pl.Int64, default_value="string")

    def test_defaults_applied_to_existing_nodes(self):
        """Test that defaults are applied to existing nodes."""
        graph = RustWorkXGraph()

        # Add a node
        graph.add_node({"t": 0})

        # Add new attribute
        graph.add_node_attr_key("score", pl.Float64)

        # Verify the default was applied
        node_attrs = graph.node_attrs()
        assert "score" in node_attrs.columns
        assert node_attrs["score"].item() == -1.0

    def test_add_edge_attr_key_with_dtype_only(self):
        """Test adding edge attribute with dtype only."""
        graph = RustWorkXGraph()

        graph.add_edge_attr_key("weight", pl.Float64)

        assert "weight" in graph._edge_attr_schemas
        assert graph._edge_attr_schemas["weight"].dtype == pl.Float64
        assert graph._edge_attr_schemas["weight"].default_value == -1.0

    def test_add_edge_attr_key_with_schema(self):
        """Test adding edge attribute using AttrSchema."""
        graph = RustWorkXGraph()

        schema = AttrSchema(key="distance", dtype=pl.Float64, default_value=0.0)
        graph.add_edge_attr_key(schema)

        assert "distance" in graph._edge_attr_schemas
        assert graph._edge_attr_schemas["distance"].default_value == 0.0

    def test_defaults_applied_to_existing_edges(self):
        """Test that defaults are applied to existing edges."""
        graph = RustWorkXGraph()

        # Add nodes and edge
        n1 = graph.add_node({"t": 0})
        n2 = graph.add_node({"t": 1})
        graph.add_edge(n1, n2, {})

        # Add new edge attribute
        graph.add_edge_attr_key("weight", pl.Float64, default_value=1.0)

        # Verify the default was applied
        edge_attrs = graph.edge_attrs()
        assert "weight" in edge_attrs.columns
        assert edge_attrs["weight"].item() == 1.0

    def test_node_attr_keys_returns_keys(self):
        """Test that node_attr_keys returns the correct keys."""
        graph = RustWorkXGraph()

        graph.add_node_attr_key("score", pl.Float64)
        graph.add_node_attr_key("count", pl.UInt32)

        keys = graph.node_attr_keys()
        assert "node_id" in keys
        assert "t" in keys
        assert "score" in keys
        assert "count" in keys

    def test_edge_attr_keys_returns_keys(self):
        """Test that edge_attr_keys returns the correct keys."""
        graph = RustWorkXGraph()

        graph.add_edge_attr_key("weight", pl.Float64)
        graph.add_edge_attr_key("distance", pl.Float64)

        keys = graph.edge_attr_keys()
        assert "weight" in keys
        assert "distance" in keys

    def test_signed_vs_unsigned_int_defaults(self):
        """Test that signed and unsigned integers get different defaults."""
        graph = RustWorkXGraph()

        graph.add_node_attr_key("unsigned", pl.UInt32)
        graph.add_node_attr_key("signed", pl.Int32)

        assert graph._node_attr_schemas["unsigned"].default_value == 0
        assert graph._node_attr_schemas["signed"].default_value == -1

    def test_schema_defensive_copy(self):
        """Test that passing AttrSchema creates a defensive copy to prevent mutation."""
        graph = RustWorkXGraph()

        # Create a schema object
        original_schema = AttrSchema(key="score", dtype=pl.Float64, default_value=1.0)

        # Add it to the graph
        graph.add_node_attr_key(original_schema)

        # Verify the schema was stored
        stored_schema = graph._node_attr_schemas["score"]
        assert stored_schema.key == "score"
        assert stored_schema.dtype == pl.Float64
        assert stored_schema.default_value == 1.0

        # Mutate the original schema
        original_schema.default_value = 999.0

        # Verify the stored schema wasn't affected (defensive copy worked)
        assert graph._node_attr_schemas["score"].default_value == 1.0
        assert stored_schema.default_value == 1.0

        # Verify the original was changed
        assert original_schema.default_value == 999.0

    def test_schema_copy_method(self):
        """Test that AttrSchema.copy() creates an independent copy."""
        original = AttrSchema(key="value", dtype=pl.Int32, default_value=42)

        # Create a copy
        copied = original.copy()

        # Verify the copy has the same values
        assert copied.key == "value"
        assert copied.dtype == pl.Int32
        assert copied.default_value == 42

        # Verify they are different objects
        assert copied is not original

        # Mutate the original
        original.default_value = -999

        # Verify the copy wasn't affected
        assert copied.default_value == 42

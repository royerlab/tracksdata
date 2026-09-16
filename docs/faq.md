# Frequently Asked Questions

## General Questions

### What is TracksData?

TracksData is a Python library that provides a common data structure and tools for multi-object tracking. It uses a graph-based representation where objects are nodes and tracks are edges connecting objects across time.

### When should I use TracksData?

TracksData is ideal for:

- Multi-object tracking in microscopy or computer vision
- Biological cell tracking and lineage analysis
- Particle tracking in physics simulations
- Any scenario requiring temporal object associations

### How does TracksData differ from other tracking libraries?

TracksData focuses on providing a **general** unified data structure and modular components that can be combined for different tracking scenarios and scale to large datasets (e.g. millions of nodes in terabytes of 3D + time imaging data).

### Which graph backend should I use?

- **RustWorkXGraph**: For most applications where data fits in memory
- **SQLGraph**: For large datasets or when you need persistent storage
- **GraphView**: You shouldn't instantiate this directly, it is used internally by the library when you use `graph.subgraph()`

### Can TracksData handle cell divisions?

Yes! The :class:`tracksdata.solvers.NearestNeighborsSolver` lets you defined the maximum number of children nodes and :class:`tracksdata.solvers.ILPSolver` specifically supports division events with configurable division costs.

### How do I add custom attributes?
```python
# Add new attribute keys to the graph
graph.add_node_attr_key("my_feature", 0.0)
graph.add_edge_attr_key("confidence", 1.0)

# Use them when adding nodes/edges
graph.add_node({"t": 0, "x": 10, "my_feature": 42.0})
graph.add_edge(source_id, target_id, {"confidence": 0.95})
```

### How do I create custom operators?

Inherit from :class:`tracksdata.edges.BaseEdgesOperator` or :class:`tracksdata.nodes.BaseNodesOperator` and implement `_add_edges_per_time` or `_add_nodes_per_time`:

```python
import tracksdata as td


class CustomNodes(td.nodes.BaseNodesOperator):
    def add_nodes(
        self,
        graph: td.graph.BaseGraph,
        *,
        t: int | None = None,
        **kwargs: Any,
    ) -> None:
        # Your custom logic here to add nodes to the graph
        pass
```

### How do I visualize results?

TracksData provides utilities for converting to napari format:

```python
import tracksdata as td

labels = ...

tracks_df, track_graph, track_labels = td.functional.to_napari_format(
    solution_graph,
    shape=labels.shape,
    mask_key="mask",
)

viewer = napari.Viewer()
viewer.add_labels(track_labels, scale=track_labels.scale)
viewer.add_tracks(tracks_df, graph=track_graph)
napari.run()
```

Large 3D data materializes a lot of voxels per timepoint, which is slow to render. Pass
`downscale` to draw the labels coarsely, and pick per-axis factors that equalize the
*physical* voxel size — for anisotropic data that usually means leaving `z` alone:

```python
import numpy as np

tracks_df, track_graph, track_labels = td.functional.to_napari_format(
    solution_graph,
    shape=labels.shape,
    mask_key="mask",
    downscale=(1, 4, 4),
    dtype=np.uint32,
)
```

Downscaling is nearest-neighbor. Objects thinner than the factor are drawn as a single
voxel so that they stay visible and selectable, but their shape is meaningless, and a
voxel covered by several objects may show either of them. Use the full-resolution view
for anything quantitative, and for editing: coordinates taken from a downscaled layer
refer to downscaled voxels and must not be written back into the graph as-is.

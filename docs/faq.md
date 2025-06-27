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
TracksData focuses on providing a unified data structure and modular components that can be combined for different tracking scenarios. It emphasizes optimal solutions through ILP while providing fast alternatives.

## Technical Questions

### Which graph backend should I use?

- **RustWorkXGraph**: For most applications where data fits in memory
- **SQLGraph**: For large datasets or when you need persistent storage
- **GraphView**: For working with subsets of larger graphs

### How do I handle missing detections?
TracksData solvers naturally handle missing detections through:

- Appearance/disappearance costs in ILPSolver
- Gap closing by allowing edges to skip time points
- Track termination and re-initialization

### Can TracksData handle cell divisions?
Yes! The ILPSolver specifically supports division events with configurable division costs. This makes it suitable for cell lineage tracking.

### How do I add custom attributes?
```python
# Add new attribute keys to the graph
graph.add_node_attr_key("my_feature", 0.0)
graph.add_edge_attr_key("confidence", 1.0)

# Use them when adding nodes/edges
graph.add_node({"t": 0, "x": 10, "my_feature": 42.0})
graph.add_edge(source_id, target_id, {"confidence": 0.95})
```

### How do I create custom edge operators?
Inherit from `BaseEdgesOperator` and implement `_add_edges_per_time`:

```python
from tracksdata.edges import BaseEdgesOperator

class CustomEdges(BaseEdgesOperator):
    def _add_edges_per_time(self, graph, *, t):
        # Your custom logic here
        pass
```

## Performance Questions

### How fast is TracksData?
Performance depends on the components used:

- **RustWorkXGraph**: Very fast for in-memory operations
- **ILPSolver**: Optimal but slower for large problems
- **NearestNeighborsSolver**: Fast heuristic for simple scenarios

### How do I optimize for large datasets?

- Use SQLGraph for datasets that don't fit in memory
- Consider NearestNeighborsSolver for speed over optimality
- Use distance thresholds to limit edge creation
- Process data in temporal chunks when possible

### Can I use multiple cores?
Yes, ILPSolver supports multi-threading via the `num_threads` parameter. Some operations also benefit from numpy's multithreading.

## Integration Questions

### How do I visualize results?
TracksData provides utilities for converting to napari format:
```python
from tracksdata.functional import to_napari_format
tracks_data = to_napari_format(solution)
```

### Can I export to other formats?
Yes, graphs can be converted to:

- Pandas/Polars DataFrames via `node_attrs()` and `edge_attrs()`
- NetworkX graphs for further analysis
- Custom formats by iterating over nodes and edges

### How do I integrate with scikit-image?
TracksData works well with scikit-image:
```python
from skimage.measure import label, regionprops
from tracksdata.nodes import RegionPropsNodes

# Segment image
labels = label(binary_image)

# Extract nodes
node_op = RegionPropsNodes()
node_op.add_nodes(graph, labels=labels, t=0)
```

## Troubleshooting

### Installation fails with rustworkx errors
Make sure you have Rust installed:
```bash
conda install -c conda-forge rust
```

### Solver fails with "no solution found"

- Check that edges exist between time points
- Verify edge weights are reasonable (not too high)
- Consider adjusting appearance/disappearance costs

### Memory usage is too high

- Use SQLGraph instead of RustWorkXGraph
- Process data in smaller chunks
- Limit the number of edges with distance thresholds

### Documentation not building
Make sure you have docs dependencies:
```bash
pip install .[docs]
mkdocs serve
```

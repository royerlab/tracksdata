# Examples

This section provides practical examples of using TracksData for multi-object tracking tasks.

## Basic Tracking Example

Here's a complete basic example that demonstrates the core workflow of TracksData. This example is available as an executable Python file at [`docs/examples/basic.py`](examples/basic.py).

```python
--8<-- "docs/examples/basic.py"
```

## Key Components Explained

- **Graph**: The core data structure holding nodes (objects) and edges (connections)
- **Nodes Operators**: Extract object features from segmented images (RegionPropsNodes, etc.)
- **Edges Operators**: Create temporal connections between objects (DistanceEdges, IoUEdges, etc.)
- **Solvers**: Optimize a minimization problem to find the best tracking assignments (NearestNeighborsSolver, ILPSolver)
- **Functional**: Utilities for format conversion and visualization

## Next Steps

- Check the [Getting Started](getting_started.md) guide for more detailed explanations
- Explore the [Concepts](concepts.md) page to understand the architecture
- See the API reference for complete documentation of all components

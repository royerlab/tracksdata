# Getting Started

## Basic Concepts

TracksData is built around a graph-based representation of multi-object tracking data:

- **Nodes** represent objects at specific time points (detections)
- **Edges** represent connections between objects across time (connections)
- **Attributes** store additional data like coordinates, features, or costs

## Quick Example

Here's a simple example of creating a graph and adding tracking data:

```python
import numpy as np
import tracksdata as td

# Create a graph
graph = td.graph.InMemoryGraph()

# Generate random nodes for testing
node_generator = td.nodes.RandomNodes(
    n_time_points=5,
    n_nodes_per_tp=(10, 15),
    n_dim=2
)
node_generator.add_nodes(graph)

# Connect nearby nodes across time
edge_generator = td.edges.DistanceEdges(
    distance_threshold=0.3,
    n_neighbors=3
)
edge_generator.add_edges(graph)

# Solve the tracking problem
solver = td.solvers.NearestNeighborsSolver(edge_weight="distance")
solution = solver.solve(graph)

print(f"Original graph has {graph.num_nodes} nodes and {graph.num_edges} edges")
print(f"Solution has {solution.num_nodes} nodes and {solution.num_edges} edges")
```

## Working with Real Data

For real tracking applications, you'll typically:

1. **Create nodes from detections** using `RegionPropsNodes` for segmented images
2. **Add edges** using `DistanceEdges` or custom edge operators
3. **Solve tracking** using `ILPSolver` for optimal results or `NearestNeighborsSolver` for speed
4. **Analyze results** using the graph's filtering and querying capabilities

```python
import tracksdata as td

# Extract nodes from labeled images
node_op = td.nodes.RegionPropsNodes(extra_properties=["area", "eccentricity"])
node_op.add_nodes(graph, labels=labels)

# Filter nodes by time
graph_filter = graph.filter(td.NodeAttr("t") == 0)
node_data = graph_filter.node_attrs()
print(node_data)
```

## Next Steps

- See the [Concepts](concepts.md) page for detailed explanations
- Check the [FAQ](faq.md) for common questions
- Browse the API documentation for complete reference

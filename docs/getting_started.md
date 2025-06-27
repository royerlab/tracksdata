# Getting Started

## Basic Concepts

TracksData is built around a graph-based representation of multi-object tracking data:

- **Nodes** represent objects at specific time points (detections)
- **Edges** represent connections between objects across time (tracks)
- **Attributes** store additional data like coordinates, features, or costs

## Quick Example

Here's a simple example of creating a graph and adding tracking data:

```python
import numpy as np
from tracksdata.graph import RustWorkXGraph
from tracksdata.nodes import RandomNodes
from tracksdata.edges import DistanceEdges
from tracksdata.solvers import ILPSolver

# Create a graph
graph = RustWorkXGraph()

# Generate random nodes for testing
node_generator = RandomNodes(
    n_time_points=5,
    n_nodes_per_tp=(10, 15),
    n_dim=2
)
node_generator.add_nodes(graph)

# Connect nearby nodes across time
edge_generator = DistanceEdges(
    distance_threshold=0.3,
    n_neighbors=3
)
edge_generator.add_edges(graph)

# Solve the tracking problem
solver = ILPSolver(
    edge_weight="weight",
    appearance_weight=5.0,
    disappearance_weight=5.0
)
solution = solver.solve(graph)

print(f"Solution has {solution.num_nodes} nodes and {solution.num_edges} edges")
```

## Working with Real Data

For real tracking applications, you'll typically:

1. **Create nodes from detections** using `RegionPropsNodes` for segmented images
2. **Add edges** using `DistanceEdges` or custom edge operators
3. **Solve tracking** using `ILPSolver` for optimal results or `NearestNeighborsSolver` for speed
4. **Analyze results** using the graph's filtering and querying capabilities

```python
from tracksdata.nodes import RegionPropsNodes
from tracksdata.attrs import NodeAttr

# Extract nodes from labeled images
node_op = RegionPropsNodes(extra_properties=["area", "eccentricity"])
node_op.add_nodes(graph, labels=labeled_images)

# Filter nodes by time
t0_nodes = graph.filter_nodes_by_attrs(NodeAttr("t") == 0)
node_data = graph.node_attrs(node_ids=t0_nodes)
print(node_data)
```

## Next Steps

- See the [Concepts](concepts.md) page for detailed explanations
- Check the [FAQ](faq.md) for common questions
- Browse the API documentation for complete reference

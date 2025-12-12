# Core Concepts

## Graph-Based Tracking

TracksData represents tracking data as a directed graph where:

- **Nodes** are detections/objects at specific time points
- **Edges** connect objects across consecutive time frames forward in time (t to t + delta_t)
- **Tracks** are paths through the graph representing object trajectories

## Graph Backends

TracksData supports multiple graph backends for different use cases:

### RustWorkXGraph
- **Use case**: In-memory graphs that fit in RAM
- **Performance**: Excellent for algorithms and analysis
- **Recommended**: For most tracking applications

### SQLGraph
- **Use case**: Large datasets that don't fit in memory
- **Performance**: Good for storage and querying
- **Features**: Persistent storage, complex queries, database indexes on frequently filtered attributes

SQLGraph lets you create indexes on node or edge attributes to keep repeated
filters fast:

```python
graph.ensure_node_attr_index(["t", "label"])  # composite index
graph.ensure_edge_attr_index("score", unique=True)
```

### GraphView
- **Use case**: Results subgraph either backends
- **Performance**: Low overhead, similar to RustWorkXGraph
- **Features**: Maintains connection to root graph, all operations are mirrored to the root graph

## Graph Operators

Graph operators are used to manipulate the graph:

- **Node operators**: Create or add attributes to nodes
- **Edge operators**: Create or add attributes to edges
- **Solver operators**: Solve the tracking problem

## Attribute System

TracksData uses a flexible attribute system:

### Node Attributes
- Store object properties (coordinates, features, measurements)
- Support various data types (floats, arrays, segmentation masks)

### Edge Attributes
- Store connection properties (distances, costs, confidences)
- Used by solvers for optimization

### Attribute Expressions

Attributes are used to filter nodes or edges, or to formulate the objective function for solvers.

```python
import tracksdata as td

# Simple attribute access
x_coords = td.NodeAttr("x")

# Mathematical expressions
distance_cost = td.EdgeAttr("distance") + 0.1 * td.EdgeAttr("angle_change")

# Comparison operations for filtering
recent_nodes = td.NodeAttr("t") >= 10
large_objects = td.NodeAttr("area") > 100
```

## Data Flow

A typical TracksData workflow:

1. **Create Graph**: Choose appropriate backend
2. **Add Nodes**: Use node operators to populate detections
3. **Add Edges**: Use edge operators to create potential connections
4. **Solve**: Apply solver to find optimal tracks
5. **Analyze**: Query and filter results for downstream analysis

This modular design allows mixing and matching components for different tracking scenarios.

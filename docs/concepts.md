# Core Concepts

## Graph-Based Tracking

TracksData represents tracking data as a directed graph where:

- **Nodes** are detections/objects at specific time points
- **Edges** connect objects across consecutive time frames
- **Tracks** are paths through the graph representing object trajectories

This representation naturally handles complex tracking scenarios like:

- Object appearances and disappearances
- Track merging and splitting (divisions)
- Temporary occlusions

## Graph Backends

TracksData supports multiple graph backends for different use cases:

### RustWorkXGraph
- **Use case**: In-memory graphs that fit in RAM
- **Performance**: Excellent for algorithms and analysis
- **Recommended**: For most tracking applications

### SQLGraph
- **Use case**: Large datasets that don't fit in memory
- **Performance**: Good for storage and querying
- **Features**: Persistent storage, complex queries

### GraphView
- **Use case**: Working with subsets of larger graphs
- **Performance**: Efficient filtering and analysis
- **Features**: Maintains connection to root graph

## Node Operators

Node operators create graph nodes from different data sources:

### RegionPropsNodes
- Extracts nodes from segmented images
- Computes region properties (area, centroid, etc.)
- Handles 2D and 3D images with time series

### RandomNodes
- Generates synthetic nodes for testing
- Useful for algorithm development and benchmarking

### Mask
- Represents individual object instances
- Stores compressed binary masks and bounding boxes
- Supports geometric operations (IoU, intersection)

## Edge Operators

Edge operators create connections between nodes:

### DistanceEdges
- Connects nearby objects based on spatial distance
- Uses KDTree for efficient neighbor finding
- Configurable distance thresholds and neighbor counts

### IoUEdgeAttr
- Computes Intersection over Union between object masks
- Useful for overlap-based linking

### GenericNodeFunctionEdgeAttrs
- Flexible framework for custom edge attributes
- Apply any function to node pairs

## Solvers

Solvers find optimal tracking solutions:

### ILPSolver
- **Method**: Integer Linear Programming
- **Quality**: Globally optimal solutions
- **Use case**: When accuracy is paramount
- **Features**: Handles appearances, disappearances, divisions

### NearestNeighborsSolver
- **Method**: Greedy nearest neighbor assignment
- **Quality**: Good heuristic solutions
- **Use case**: When speed is important
- **Features**: Fast, simple, works well for simple scenarios

## Attribute System

TracksData uses a flexible attribute system:

### Node Attributes
- Store object properties (coordinates, features, measurements)
- Support various data types (floats, arrays, custom objects)
- Automatically validated against graph schema

### Edge Attributes
- Store connection properties (distances, costs, confidences)
- Used by solvers for optimization
- Support mathematical expressions

### Attribute Expressions
```python
from tracksdata.attrs import NodeAttr, EdgeAttr

# Simple attribute access
x_coords = NodeAttr("x")

# Mathematical expressions
distance_cost = EdgeAttr("distance") + 0.1 * EdgeAttr("angle_change")

# Comparison operations for filtering
recent_nodes = NodeAttr("t") >= 10
large_objects = NodeAttr("area") > 100
```

## Data Flow

A typical TracksData workflow:

1. **Create Graph**: Choose appropriate backend
2. **Add Nodes**: Use node operators to populate detections
3. **Add Edges**: Use edge operators to create potential connections
4. **Solve**: Apply solver to find optimal tracks
5. **Analyze**: Query and filter results for downstream analysis

This modular design allows mixing and matching components for different tracking scenarios.

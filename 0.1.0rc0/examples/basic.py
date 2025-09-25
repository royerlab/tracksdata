"""
Basic multi-object tracking example using TracksData.

This example demonstrates the complete workflow for tracking objects across time:
1. Load segmented image data
2. Extract object features (nodes) from each frame
3. Create temporal connections (edges) between objects
4. Compute additional attributes to edges (e.g. IoU)
5. Solve the tracking optimization problem
6. Convert results to napari format
7. Visualize results


Requirements:
- Set CTC_DIR environment variable pointing to Cell Tracking Challenge data
- Example assumes Fluo-N2DL-HeLa dataset structure

Usage:
    python basic.py                    # Run with napari visualization
    python basic.py --profile          # Run with performance profiling
"""

import os
from pathlib import Path

import click
import napari
import numpy as np
from profilehooks import profile as profile_hook
from tifffile import imread

import tracksdata as td


def basic_tracking_example(show_napari_viewer: bool = True) -> None:
    """
    Perform basic multi-object tracking on segmented microscopy data.

    This function demonstrates the core TracksData workflow:
    - Node extraction from regionprops
    - Distance-based edge creation
    - IoU attribute computation
    - Nearest neighbors tracking solution

    Parameters
    ----------
    show_napari_viewer : bool
        Whether to display results in napari viewer
    """
    # Step 1: Load and prepare data

    # Load example data from Cell Tracking Challenge format
    data_dir = Path(os.environ["CTC_DIR"]) / "training/Fluo-N2DL-HeLa/01_GT/TRA"
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."

    # Load all timepoints as a 3D array: (time, height, width)
    labels = np.stack(
        [imread(p) for p in sorted(data_dir.glob("*.tif"))],
    )

    # Configure TracksData options (disable progress bars for cleaner output)
    td.options.set_options(show_progress=False)

    print("Starting tracking workflow...")

    # Step 2: Initialize graph and extract nodes
    graph = td.graph.InMemoryGraph()

    # Extract object features using region properties
    # This creates one node per object per timeframe
    nodes_operator = td.nodes.RegionPropsNodes()
    nodes_operator.add_nodes(graph, labels=labels)
    print(f"✓ Extracted {graph.num_nodes} nodes from {labels.shape[0]} timeframes")

    # Step 3: Create temporal edges between consecutive frames

    # Add distance-based edges between objects in consecutive timeframes
    # Only connects objects within distance_threshold and limits to n_neighbors
    dist_operator = td.edges.DistanceEdges(
        distance_threshold=30.0,
        n_neighbors=5,
    )
    dist_operator.add_edges(graph)
    print(f"✓ Created {graph.num_edges} potential temporal connections")

    # Step 4: Add IoU (Intersection over Union) attributes to edges

    # Compute IoU between connected objects to measure shape similarity
    # Higher IoU values indicate better matches for tracking
    iou_operator = td.edges.IoUEdgeAttr(output_key="iou")
    iou_operator.add_edge_attrs(graph)
    print("✓ Computed IoU attributes for edge weights")

    # Step 5: Solve tracking optimization problem

    # Create edge weights combining distance and IoU information
    # Lower distance + higher IoU = better connection (lower cost)
    dist_weight = 1 / dist_operator.distance_threshold

    # Use nearest neighbors solver for fast, greedy tracking
    # Each edge weight is defined as:
    # - IoU(e_ij) * exp(-distance(e_ij) / dist_threshold)
    # Where e_ij is the edge between nodes i and j.
    # Alternative: ILPSolver for globally optimal but slower solutions
    solver = td.solvers.NearestNeighborsSolver(
        edge_weight=-td.EdgeAttr("iou") * (td.EdgeAttr("distance") * dist_weight).exp(),
        max_children=2,  # Allow cell divisions (max 2 children per parent)
    )

    # Alternative ILP solver (uncomment for optimal tracking):
    # solver = td.solvers.ILPSolver(
    #     edge_weight=-td.EdgeAttr("iou") * (td.EdgeAttr("weight") * dist_weight).exp(),
    #     node_weight=0.0,           # Cost for keeping an object
    #     appearance_weight=1.0,    # Cost for object appearing
    #     disappearance_weight=1.0, # Cost for object disappearing
    #     division_weight=1.0,       # Cost for cell division
    # )

    solver.solve(graph)
    print("✓ Solved tracking assignments")

    # Step 6: Convert results for visualization

    # Convert tracking graph to napari-compatible format
    # Returns: tracked labels, tracks dataframe, and track graph
    print("Converting results to napari format...")
    tracks_df, track_graph, track_labels = td.functional.to_napari_format(graph, labels.shape, mask_key="mask")

    print(f"✓ Generated {len(tracks_df)} track points across {len(set(tracks_df['track_id']))} tracks")

    # Step 7: Visualize results (optional)

    if show_napari_viewer:
        print("Opening napari viewer...")
        viewer = napari.Viewer()

        # Add original segmented labels
        viewer.add_labels(track_labels, name="Tracked Labels")

        # Add tracking trajectories with lineage information
        viewer.add_tracks(tracks_df, graph=track_graph, name="Tracks")

        # Start interactive viewer
        napari.run()


@click.command()
@click.option("--profile", is_flag=True, help="Enable performance profiling (disables napari viewer)")
def main(profile: bool) -> None:
    """Run the basic tracking example with optional profiling."""
    if profile:
        # Run with performance profiling, no visualization
        profile_hook(basic_tracking_example, immediate=True, sort="time")(show_napari_viewer=False)
    else:
        # Normal run with visualization
        basic_tracking_example(show_napari_viewer=True)


if __name__ == "__main__":
    main()

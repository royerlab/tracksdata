import os
from pathlib import Path

import click
import napari
import numpy as np
from profilehooks import profile as profile_hook
from tifffile import imread

from tracksdata.attrs import EdgeAttr
from tracksdata.edges import DistanceEdges, IoUEdgeAttr
from tracksdata.functional._napari import to_napari_format
from tracksdata.graph import RustWorkXGraph, SQLGraph  # noqa: F401
from tracksdata.nodes import RegionPropsNodes
from tracksdata.options import set_options
from tracksdata.solvers import ILPSolver, NearestNeighborsSolver  # noqa: F401


def _minimal_example(show_napari_viewer: bool) -> None:
    # load from HeLa
    data_dir = Path(os.environ["CTC_DIR"]) / "training/Fluo-N2DL-HeLa/01_GT/TRA"
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."

    labels = np.stack(
        [imread(p) for p in sorted(data_dir.glob("*.tif"))],
    )

    set_options(show_progress=False)

    print("starting tracking ...")
    graph = RustWorkXGraph()

    nodes_operator = RegionPropsNodes()
    nodes_operator.add_nodes(graph, labels=labels)
    print(f"Number of nodes: {graph.num_nodes}")

    dist_operator = DistanceEdges(distance_threshold=30.0, n_neighbors=5)
    dist_operator.add_edges(graph)
    print(f"Number of edges: {graph.num_edges}")

    iou_operator = IoUEdgeAttr(output_key="iou")
    iou_operator.add_edge_attrs(graph)

    dist_weight = 1 / dist_operator.distance_threshold
    solver = NearestNeighborsSolver(
        edge_weight=-EdgeAttr("iou") + EdgeAttr("weight") * dist_weight,
        max_children=2,
    )
    # solver = ILPSolver(
    #     edge_weight=-AttrExpr("iou") + AttrExpr("weight") * dist_weight,
    #     node_weight=0.0,
    #     appearance_weight=10.0,
    #     disappearance_weight=10.0,
    #     division_weight=1.0,
    # )
    solver.solve(graph)

    print("Converting to napari format ...")
    labels, tracks_df, track_graph = to_napari_format(graph, labels.shape)

    print("Opening napari viewer ...")

    if show_napari_viewer:
        viewer = napari.Viewer()
        viewer.add_labels(labels)
        viewer.add_tracks(tracks_df, graph=track_graph)
        napari.run()


@click.command()
@click.option("--profile", is_flag=True, type=bool, default=False)
def main(profile: bool) -> None:
    if profile:
        profile_hook(_minimal_example, immediate=True, sort="time")(show_napari_viewer=False)
    else:
        _minimal_example(show_napari_viewer=True)


if __name__ == "__main__":
    main()

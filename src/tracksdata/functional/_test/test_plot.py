import numpy as np
import polars as pl
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from tracksdata.constants import DEFAULT_ATTR_KEYS  # noqa: E402
from tracksdata.functional import plot_lineage_tree  # noqa: E402
from tracksdata.graph import RustWorkXGraph  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    yield
    plt.close("all")


def _dividing_graph() -> RustWorkXGraph:
    """Build a graph with a single lineage: tracklet 1 divides into tracklets 2 and 3."""
    positions = np.asarray(
        [
            [0, 0, 0],  # t=0, tracklet 1
            [1, 0, 0],  # t=1, tracklet 1
            [2, 0, 0],  # t=2, tracklet 2
            [3, 0, 0],  # t=3, tracklet 2
            [2, 1, 1],  # t=2, tracklet 3
            [3, 1, 1],  # t=3, tracklet 3
        ]
    )
    tracklet_ids = np.asarray([1, 1, 2, 2, 3, 3])
    graph = RustWorkXGraph.from_array(
        positions,
        tracklet_ids=tracklet_ids,
        tracklet_id_graph={2: 1, 3: 1},
    )
    graph.add_node_attr_key("feature", pl.Float64)
    graph.update_node_attrs(
        node_ids=graph.node_ids(),
        attrs={"feature": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]},
    )
    return graph


def test_plot_lineage_tree_basic() -> None:
    """Test the default lineage tree layout and edge drawing."""
    graph = _dividing_graph()

    ax = plot_lineage_tree(graph)

    assert isinstance(ax, Axes)

    lines, scatter = ax.collections
    offsets = np.asarray(scatter.get_offsets())
    assert offsets.shape == (graph.num_nodes(), 2)
    assert len(lines.get_segments()) == graph.num_edges()

    # vertical orientation: time on the (inverted) y-axis
    assert ax.yaxis_inverted()
    assert ax.get_ylabel() == "time"
    np.testing.assert_array_equal(np.sort(np.unique(offsets[:, 1])), [0.0, 1.0, 2.0, 3.0])

    # the parent tracklet is centered between its two children
    nodes_df = graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.TRACKLET_ID])
    tracklet_ids = nodes_df[DEFAULT_ATTR_KEYS.TRACKLET_ID].to_numpy()
    tree_coords = {tid: set(offsets[tracklet_ids == tid, 0]) for tid in (1, 2, 3)}
    for tid in (1, 2, 3):
        assert len(tree_coords[tid]) == 1  # all nodes of a tracklet share the same coordinate
    (parent_x,) = tree_coords[1]
    (child_a_x,) = tree_coords[2]
    (child_b_x,) = tree_coords[3]
    assert child_a_x != child_b_x
    assert parent_x == pytest.approx((child_a_x + child_b_x) / 2)


def test_plot_lineage_tree_color_and_size() -> None:
    """Test binding attributes to marker colors and sizes."""
    graph = _dividing_graph()

    ax = plot_lineage_tree(
        graph,
        color="feature",
        cmap="magma",
        color_norm=(0.0, 10.0),
        size="feature",
        size_range=(10.0, 50.0),
    )

    scatter = ax.collections[-1]

    feature = graph.node_attrs(attr_keys=["feature"])["feature"].to_numpy()
    np.testing.assert_array_equal(np.asarray(scatter.get_array()), feature)
    assert scatter.get_cmap().name == "magma"
    assert scatter.norm.vmin == 0.0
    assert scatter.norm.vmax == 10.0

    sizes = np.asarray(scatter.get_sizes())
    expected = 10.0 + (feature - feature.min()) / (feature.max() - feature.min()) * 40.0
    np.testing.assert_allclose(sizes, expected)


def test_plot_lineage_tree_size_norm() -> None:
    """Test explicit size normalization limits with clipping."""
    graph = _dividing_graph()

    ax = plot_lineage_tree(graph, size="feature", size_norm=(0.0, 2.0), size_range=(10.0, 50.0))

    sizes = np.asarray(ax.collections[-1].get_sizes())
    feature = graph.node_attrs(attr_keys=["feature"])["feature"].to_numpy()
    expected = 10.0 + np.clip(feature / 2.0, 0.0, 1.0) * 40.0
    np.testing.assert_allclose(sizes, expected)

    # a matplotlib Normalize instance is accepted as well
    _, ax2 = plt.subplots()
    norm = matplotlib.colors.Normalize(0.0, 2.0, clip=True)
    plot_lineage_tree(graph, ax=ax2, size="feature", size_norm=norm, size_range=(10.0, 50.0))
    np.testing.assert_allclose(np.asarray(ax2.collections[-1].get_sizes()), expected)


def test_plot_lineage_tree_time_points_window() -> None:
    """A contiguous `time_points` window (a range) limits the displayed nodes and edges."""
    graph = _dividing_graph()

    ax = plot_lineage_tree(graph, time_points=range(1, 3))

    lines, scatter = ax.collections
    # nodes: t=1 (tracklet 1) and t=2 (tracklets 2 and 3)
    assert len(scatter.get_offsets()) == 3
    # edges: only the two division edges are fully within the range
    assert len(lines.get_segments()) == 2

    # evenly separated positions labeled with the actual time points
    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert labels == ["1", "2"]


def test_plot_lineage_tree_time_points() -> None:
    """Test selecting an arbitrary, non-contiguous subset of time points."""
    graph = _dividing_graph()

    ax = plot_lineage_tree(graph, time_points=[0, 3])

    lines, scatter = ax.collections
    # markers: t=0 (tracklet 1) and t=3 (tracklets 2 and 3)
    offsets = np.asarray(scatter.get_offsets())
    assert len(offsets) == 3
    # every time point in the displayed range is evenly separated, hidden ones
    # included, and only the displayed ones are labeled
    np.testing.assert_array_equal(np.sort(np.unique(offsets[:, 1])), [0.0, 3.0])
    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert labels == ["0", "3"]

    # edges still run through the hidden t=1 and t=2 nodes
    segments = np.asarray(lines.get_segments())
    assert len(segments) == graph.num_edges()
    np.testing.assert_array_equal(np.sort(np.unique(segments[:, :, 1])), [0.0, 1.0, 2.0, 3.0])
    # the division fork starts at the hidden t=1 node and ends at the two distinct t=2 nodes
    forks = segments[segments[:, 0, 1] == 1.0]
    assert len(forks) == 2
    np.testing.assert_array_equal(forks[:, 1, 1], [2.0, 2.0])
    assert forks[0, 1, 0] != forks[1, 1, 0]


def test_plot_lineage_tree_edge_colors() -> None:
    """Test styling marker borders via scatter_kwargs (edgecolors/linewidths)."""
    graph = _dividing_graph()

    ax = plot_lineage_tree(
        graph,
        color="feature",
        scatter_kwargs={"edgecolors": "red", "linewidths": 1.5},
    )

    scatter = ax.collections[-1]
    np.testing.assert_allclose(scatter.get_edgecolors()[0], [1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(scatter.get_linewidths(), [1.5])
    # face colors still come from the colormap, independent of the edge color
    np.testing.assert_array_equal(np.asarray(scatter.get_array()), graph.node_attrs(attr_keys=["feature"])["feature"])


def test_plot_lineage_tree_time_positions() -> None:
    """Test exact time positions given as a mapping and as a sequence."""
    graph = _dividing_graph()

    timestamps = {t: 100.0 + 10.0 * t for t in range(4)}
    ax = plot_lineage_tree(graph, time_positions=timestamps)
    offsets = np.asarray(ax.collections[-1].get_offsets())
    np.testing.assert_array_equal(
        np.sort(np.unique(offsets[:, 1])),
        [100.0, 110.0, 120.0, 130.0],
    )

    ax = plot_lineage_tree(graph, time_positions=np.asarray([0.0, 1.0, 2.0, 10.0]))
    offsets = np.asarray(ax.collections[-1].get_offsets())
    np.testing.assert_array_equal(np.sort(np.unique(offsets[:, 1])), [0.0, 1.0, 2.0, 10.0])

    with pytest.raises(ValueError, match="missing positions"):
        plot_lineage_tree(graph, time_positions={0: 0.0})

    with pytest.raises(ValueError, match="cannot be indexed"):
        plot_lineage_tree(graph, time_positions=np.asarray([0.0, 1.0]))


def test_plot_lineage_tree_horizontal() -> None:
    """Test horizontal orientation with time on the x-axis."""
    graph = _dividing_graph()

    ax = plot_lineage_tree(graph, orientation="horizontal")

    offsets = np.asarray(ax.collections[-1].get_offsets())
    np.testing.assert_array_equal(np.sort(np.unique(offsets[:, 0])), [0.0, 1.0, 2.0, 3.0])
    assert ax.get_xlabel() == "time"
    assert not ax.yaxis_inverted()

    with pytest.raises(ValueError, match="`orientation` must be"):
        plot_lineage_tree(graph, orientation="diagonal")


def test_plot_lineage_tree_assigns_tracklet_ids() -> None:
    """Test that tracklet ids are assigned when the key is missing."""
    positions = np.asarray([[0, 0, 0], [1, 5, 5]])
    graph = RustWorkXGraph.from_array(positions)

    assert "my_tracklet_id" not in graph.node_attr_keys()

    ax = plot_lineage_tree(graph, tracklet_id_key="my_tracklet_id")

    assert "my_tracklet_id" in graph.node_attr_keys()
    assert len(ax.collections[-1].get_offsets()) == 2


def test_plot_lineage_tree_existing_axes_and_kwargs() -> None:
    """Test plotting into an existing axes with custom artist kwargs."""
    graph = _dividing_graph()

    _, ax = plt.subplots()
    returned_ax = plot_lineage_tree(
        graph,
        ax=ax,
        scatter_kwargs={"alpha": 0.5},
        line_kwargs={"color": "red"},
    )

    assert returned_ax is ax
    assert ax.collections[-1].get_alpha() == 0.5


def test_plot_lineage_tree_errors() -> None:
    """Test error handling for empty selections and missing attributes."""
    graph = _dividing_graph()

    with pytest.raises(ValueError, match="No nodes to plot"):
        plot_lineage_tree(graph, time_points=range(10, 21))

    with pytest.raises(ValueError, match="not found in graph"):
        plot_lineage_tree(graph, color="does_not_exist")


def test_plot_lineage_tree_color_callable_scalar() -> None:
    """A color callable returning numbers is colormap-mapped and colorbar-compatible."""
    graph = _dividing_graph()

    ax = plot_lineage_tree(
        graph,
        color=lambda row: 2.0 * row["feature"],
        cmap="magma",
        attrs=["feature"],
    )

    scatter = ax.collections[-1]
    feature = graph.node_attrs(attr_keys=["feature"])["feature"].to_numpy()
    # numeric callable output feeds scatter's color array -> colorbar mapping exists
    assert scatter.get_array() is not None
    np.testing.assert_array_equal(np.asarray(scatter.get_array()), 2.0 * feature)
    assert scatter.get_cmap().name == "magma"


def test_plot_lineage_tree_color_callable_categorical() -> None:
    """A color callable returning literal colors is used verbatim (no colorbar)."""
    graph = _dividing_graph()

    ax = plot_lineage_tree(
        graph,
        color=lambda row: "red" if row["feature"] < 3.0 else "blue",
        attrs=["feature"],
    )

    scatter = ax.collections[-1]
    # literal colors -> no scalar mapping
    assert scatter.get_array() is None
    feature = graph.node_attrs(attr_keys=["feature"])["feature"].to_numpy()
    facecolors = scatter.get_facecolors()
    red = np.array([1.0, 0.0, 0.0, 1.0])
    blue = np.array([0.0, 0.0, 1.0, 1.0])
    expected = np.where((feature < 3.0)[:, None], red, blue)
    np.testing.assert_allclose(facecolors, expected)


def test_plot_lineage_tree_size_callable() -> None:
    """A size callable returns marker sizes directly."""
    graph = _dividing_graph()

    ax = plot_lineage_tree(graph, size=lambda row: 5.0 + row["feature"], attrs=["feature"])

    sizes = np.asarray(ax.collections[-1].get_sizes())
    feature = graph.node_attrs(attr_keys=["feature"])["feature"].to_numpy()
    np.testing.assert_allclose(sizes, 5.0 + feature)


def test_plot_lineage_tree_size_constant() -> None:
    """A numeric size sets a constant marker size."""
    graph = _dividing_graph()

    ax = plot_lineage_tree(graph, size=42.0)

    sizes = np.asarray(ax.collections[-1].get_sizes())
    np.testing.assert_allclose(sizes, np.full(graph.num_nodes(), 42.0))


def test_plot_lineage_tree_marker_callable_groups() -> None:
    """A marker callable groups nodes by glyph into one scatter call each."""
    graph = _dividing_graph()

    ax = plot_lineage_tree(
        graph,
        marker=lambda row: "s" if row["feature"] < 3.0 else "^",
        attrs=["feature"],
    )

    # one LineCollection for edges + one PathCollection per distinct glyph
    scatters = ax.collections[1:]
    assert len(scatters) == 2
    total = sum(len(s.get_offsets()) for s in scatters)
    assert total == graph.num_nodes()


def test_plot_lineage_tree_marker_single_glyph() -> None:
    """A marker string applies a single glyph to every node in one scatter call."""
    graph = _dividing_graph()

    ax = plot_lineage_tree(graph, marker="s")

    _lines, scatter = ax.collections
    assert len(scatter.get_offsets()) == graph.num_nodes()


def test_plot_lineage_tree_marker_and_color_share_norm() -> None:
    """Marker groups share one normalization so colors stay consistent and colorbar-compatible."""
    graph = _dividing_graph()

    ax = plot_lineage_tree(
        graph,
        color="feature",
        color_norm=(0.0, 10.0),
        marker=lambda row: "s" if row["feature"] < 3.0 else "^",
        attrs=["feature"],
    )

    scatters = ax.collections[1:]
    assert len(scatters) == 2
    for scatter in scatters:
        assert scatter.norm.vmin == 0.0
        assert scatter.norm.vmax == 10.0
        assert scatter.get_cmap().name == "viridis"


def test_plot_lineage_tree_text() -> None:
    """Text labels are annotated per node, from an attribute name or a callable."""
    graph = _dividing_graph()

    ax = plot_lineage_tree(graph, text="feature")
    texts = {t.get_text() for t in ax.texts}
    assert len(ax.texts) == graph.num_nodes()
    assert "0.0" in texts

    _, ax2 = plt.subplots()
    plot_lineage_tree(graph, ax=ax2, text=lambda row: f"n{row['feature']:.0f}", attrs=["feature"])
    labels = {t.get_text() for t in ax2.texts}
    assert "n0" in labels and "n5" in labels


def test_plot_lineage_tree_callable_without_attrs_warns() -> None:
    """A callable without `attrs` warns and still plots by loading all attributes."""
    graph = _dividing_graph()

    with pytest.warns(UserWarning, match="loading all node attributes"):
        ax = plot_lineage_tree(graph, color=lambda row: row["feature"])

    assert len(ax.collections[-1].get_offsets()) == graph.num_nodes()


def test_plot_lineage_tree_color_literal() -> None:
    """A `color` string that is not an attribute name is a literal color for every node."""
    graph = _dividing_graph()

    ax = plot_lineage_tree(graph, color="tab:blue")

    scatter = ax.collections[-1]
    assert scatter.get_array() is None
    expected = np.tile(matplotlib.colors.to_rgba("tab:blue"), (graph.num_nodes(), 1))
    np.testing.assert_allclose(scatter.get_facecolors(), expected)


def test_plot_lineage_tree_edge_color_literal() -> None:
    """A literal `edge_color` outlines every node without touching the face colors."""
    graph = _dividing_graph()

    ax = plot_lineage_tree(graph, color="feature", edge_color="red")

    scatter = ax.collections[-1]
    expected = np.tile([1.0, 0.0, 0.0, 1.0], (graph.num_nodes(), 1))
    np.testing.assert_allclose(scatter.get_edgecolors(), expected)
    np.testing.assert_array_equal(np.asarray(scatter.get_array()), graph.node_attrs(attr_keys=["feature"])["feature"])


def test_plot_lineage_tree_edge_color_attribute_shares_cmap_and_norm() -> None:
    """A numeric `edge_color` is mapped through the same colormap and normalization as `color`."""
    graph = _dividing_graph()
    feature = graph.node_attrs(attr_keys=["feature"])["feature"].to_numpy()

    ax = plot_lineage_tree(graph, color="feature", edge_color="feature", cmap="magma")

    scatter = ax.collections[-1]
    expected = plt.get_cmap("magma")(matplotlib.colors.Normalize(0.0, 5.0)(feature))
    np.testing.assert_allclose(scatter.get_edgecolors(), expected)
    # colormapped face colors are only resolved when drawn
    ax.figure.canvas.draw()
    np.testing.assert_allclose(scatter.get_facecolors(), expected)

    # the shared normalization spans the union of face and edge value ranges
    _, ax2 = plt.subplots()
    plot_lineage_tree(
        graph,
        ax=ax2,
        color="feature",
        edge_color=lambda row: 2.0 * row["feature"],
        attrs=["feature"],
    )
    assert ax2.collections[-1].norm.vmin == 0.0
    assert ax2.collections[-1].norm.vmax == 10.0


def test_plot_lineage_tree_edge_color_callable_categorical() -> None:
    """An `edge_color` callable returning literal colors is used per node, also across marker groups."""
    graph = _dividing_graph()
    feature = graph.node_attrs(attr_keys=["feature"])["feature"].to_numpy()

    ax = plot_lineage_tree(
        graph,
        edge_color=lambda row: "red" if row["feature"] < 3.0 else "blue",
        marker=lambda row: "s" if row["feature"] < 3.0 else "^",
        attrs=["feature"],
    )

    red = np.array([1.0, 0.0, 0.0, 1.0])
    blue = np.array([0.0, 0.0, 1.0, 1.0])
    squares, triangles = ax.collections[1:]
    np.testing.assert_allclose(squares.get_edgecolors(), np.tile(red, (int((feature < 3.0).sum()), 1)))
    np.testing.assert_allclose(triangles.get_edgecolors(), np.tile(blue, (int((feature >= 3.0).sum()), 1)))


def test_plot_lineage_tree_edge_color_invalid() -> None:
    """An `edge_color` string that is neither an attribute nor a color is rejected."""
    graph = _dividing_graph()

    with pytest.raises(ValueError, match="not found in graph"):
        plot_lineage_tree(graph, edge_color="does_not_exist")

"""Matplotlib-based plotting utilities for lineage trees."""

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
import rustworkx as rx
from numpy.typing import ArrayLike

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize

__all__ = ["plot_lineage_tree"]


def _resolve_color_channel(
    spec: "str | Callable[[Mapping[str, Any]], Any] | None",
    nodes_df: pl.DataFrame,
    rows: list[Mapping[str, Any]],
) -> tuple[Any, bool]:
    """
    Resolve a `color`/`edge_color` specification into per-node color values.

    Parameters
    ----------
    spec : str | Callable | None
        An attribute name (loaded in `nodes_df`), a literal matplotlib color,
        a callable of the node attribute row, or None.
    nodes_df : pl.DataFrame
        Displayed nodes with the attributes referenced by string specs loaded.
    rows : list[Mapping[str, Any]]
        Per-node attribute rows, fed to callables (unused otherwise).

    Returns
    -------
    tuple[Any, bool]
        `(values, is_scalar)`. If the values form a 1-D numeric array,
        `values` is that array and `is_scalar` is True, so they are mapped
        through a colormap (and support a colorbar). Otherwise `values` is a
        list of literal colors (names, hex, or RGB(A) tuples), one per node,
        and `is_scalar` is False. `(None, False)` when `spec` is None.
    """
    if spec is None:
        return None, False
    if not callable(spec):
        if spec in nodes_df.columns:
            return nodes_df[spec].to_numpy(), True
        # validated upfront as a matplotlib color: same literal for every node
        return [spec] * len(nodes_df), False

    raw = [spec(row) for row in rows]
    try:
        arr = np.asarray(raw, dtype=float)
    except (ValueError, TypeError):
        return raw, False
    if arr.ndim == 1:
        return arr, True
    # (N, 3) or (N, 4): literal RGB(A) colors, not colormap-able scalars
    return raw, False


def _tracklet_tree_layout(tracklet_graph: rx.PyDiGraph) -> dict[int, float]:
    """
    Assign a tree-axis coordinate to each tracklet of a tracklet graph.

    Leaf tracklets receive consecutive integer coordinates and each parent
    tracklet is centered at the mean coordinate of its children, resulting
    in the classic dendrogram-like lineage tree layout.

    Parameters
    ----------
    tracklet_graph : rx.PyDiGraph
        Compressed tracklet graph as returned by
        [BaseGraph.tracklet_graph][tracksdata.graph.BaseGraph.tracklet_graph],
        where node values are tracklet ids and edges point from parent to child.

    Returns
    -------
    dict[int, float]
        Mapping of tracklet id to tree-axis coordinate.
    """
    positions: dict[int, float] = {}
    visited: set[int] = set()
    next_leaf = 0.0

    roots = sorted(
        (rx_id for rx_id in tracklet_graph.node_indices() if tracklet_graph.in_degree(rx_id) == 0),
        key=tracklet_graph.__getitem__,
    )

    for root in roots:
        # Iterative post-order DFS: children are positioned before their parents.
        # DFS visits a subtree contiguously, so its leaves take a consecutive block of
        # slots and its parents, being means of their children, land inside that block.
        # Sibling blocks are therefore disjoint and no edges cross.
        #
        # For 1 -> (2, 3) and 3 -> (4, 5), tracklets are positioned in the order
        # 2, 4, 5, 3, 1, giving 2: 0.0, 4: 1.0, 5: 2.0, 3: 1.5, 1: 0.75.
        stack: list[tuple[int, bool]] = [(root, False)]
        while stack:
            # `rx_id` is a rustworkx node index, `tracklet_graph[rx_id]` the tracklet id
            rx_id, expanded = stack.pop()
            if expanded:
                # second visit: every child has been positioned already
                children_pos = [
                    positions[tracklet_graph[child]]
                    for child in tracklet_graph.successor_indices(rx_id)
                    if tracklet_graph[child] in positions
                ]
                if children_pos:
                    positions[tracklet_graph[rx_id]] = float(np.mean(children_pos))
                else:
                    # leaf tracklet: take the next free slot
                    positions[tracklet_graph[rx_id]] = next_leaf
                    next_leaf += 1.0
            elif rx_id not in visited:
                # first visit: re-push self as expanded, then push children on top
                visited.add(rx_id)
                stack.append((rx_id, True))
                # reversed order, so LIFO pops children by ascending tracklet id
                for child in sorted(
                    tracklet_graph.successor_indices(rx_id),
                    key=tracklet_graph.__getitem__,
                    reverse=True,
                ):
                    # skip children already reached via another parent (merges)
                    if child not in visited:
                        stack.append((child, False))

    return positions


def _time_axis_positions(
    time_points: list[int],
    time_positions: "Mapping[int, float] | ArrayLike | None",
) -> dict[int, float]:
    """
    Map each time point to its coordinate along the time axis.

    Parameters
    ----------
    time_points : list[int]
        Sorted unique time points needing a coordinate: the displayed ones
        and the hidden ones in between that edges pass through.
    time_positions : Mapping[int, float] | ArrayLike | None
        Exact time-axis coordinates (e.g. timestamps). Either a mapping of
        time point to coordinate or a sequence indexed by time point.
        If None, time points are evenly separated in their sorted order.

    Returns
    -------
    dict[int, float]
        Mapping of time point to time-axis coordinate.
    """
    if time_positions is None:
        return {t: float(i) for i, t in enumerate(time_points)}

    if isinstance(time_positions, Mapping):
        missing = [t for t in time_points if t not in time_positions]
        if missing:
            raise ValueError(f"`time_positions` is missing positions for time points {missing}")
        return {t: float(time_positions[t]) for t in time_points}

    time_positions = np.asarray(time_positions)
    if time_positions.ndim != 1:
        raise ValueError(f"`time_positions` must be 1-dimensional, got {time_positions.ndim} dimensions.")
    if time_points[-1] >= len(time_positions):
        raise ValueError(
            f"`time_positions` of length {len(time_positions)} cannot be indexed "
            f"by the maximum time point {time_points[-1]}."
        )
    return {t: float(time_positions[t]) for t in time_points}


def plot_lineage_tree(
    graph: BaseGraph,
    *,
    ax: "Axes | None" = None,
    tracklet_id_key: str = DEFAULT_ATTR_KEYS.TRACKLET_ID,
    color: "str | Callable[[Mapping[str, Any]], Any] | None" = None,
    edge_color: "str | Callable[[Mapping[str, Any]], Any] | None" = None,
    cmap: "str | Colormap" = "viridis",
    color_norm: "Normalize | tuple[float, float] | None" = None,
    size: "str | Callable[[Mapping[str, Any]], float] | float" = 30.0,
    size_norm: "Normalize | tuple[float, float] | None" = None,
    size_range: tuple[float, float] = (10.0, 100.0),
    marker: "str | Callable[[Mapping[str, Any]], str] | None" = None,
    text: "str | Callable[[Mapping[str, Any]], Any] | None" = None,
    text_kwargs: dict[str, Any] | None = None,
    attrs: Sequence[str] | None = None,
    time_points: Sequence[int] | None = None,
    time_positions: "Mapping[int, float] | ArrayLike | None" = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    scatter_kwargs: dict[str, Any] | None = None,
    line_kwargs: dict[str, Any] | None = None,
) -> "Axes":
    """
    Plot a graph as a lineage tree with matplotlib.

    Nodes are drawn as points aligned in time and grouped by tracklet,
    with parent tracklets centered above their children. Edges are drawn
    as line segments, so divisions appear as forks in the tree. When only a
    subset of time points is shown, markers are limited to those time points
    while edges still run through the hidden nodes in between, so the tree
    keeps its shape and divisions stay at their true time.

    The `color`, `edge_color`, `size`, `marker`, and `text` aesthetics each
    accept either a fixed value or a callable, which is the main way to
    customize the markers:

    - As a string, `color`/`edge_color`/`size`/`text` name a numeric node
      attribute, and `marker` is a single matplotlib marker glyph applied to
      every node. A `color`/`edge_color` string that is not an attribute name
      is taken as a literal matplotlib color (e.g. `"tab:red"`, `"none"`)
      applied to every node.
    - As a callable, they receive each node's attribute row (a mapping of
      attribute key to value) and return that node's color, size, marker glyph,
      or text label. This allows categorical colors, per-node marker shapes,
      and colors derived from a computed quantity (e.g. `np.log1p(row["area"])`).

    A colorbar-compatible mapping is available whenever `color` produces numeric
    values (a numeric attribute name, or a callable returning numbers) together
    with `cmap`. If a callable returns literal colors (names, hex, or RGB(A)),
    those colors are used verbatim and no colorbar mapping exists. Numeric
    `edge_color` values are mapped through the same `cmap`/`color_norm` as
    `color`, so face and edge colors are directly comparable.

    Requires `matplotlib`, which is an optional dependency
    (`pip install "tracksdata[plot]"`).

    IMPORTANT: If `tracklet_id_key` is not an existing node attribute,
    tracklet ids are assigned on the fly, modifying the graph.
    To plot only solution nodes, pass the solution subgraph, e.g.
    `graph.filter(NodeAttr("solution") == True, EdgeAttr("solution") == True).subgraph()`.

    Parameters
    ----------
    graph : BaseGraph
        The graph to plot.
    ax : Axes | None, optional
        The matplotlib axes to plot into. If None, a new figure and axes
        are created.
    tracklet_id_key : str, optional
        The key of the tracklet id node attribute. If the key does not exist,
        [BaseGraph.assign_tracklet_ids][tracksdata.graph.BaseGraph.assign_tracklet_ids]
        is called first.
    color : str | Callable | None, optional
        Marker face color. A string names a numeric node attribute mapped
        through `cmap`/`color_norm` (a colorbar mapping is available); if it
        is not an attribute name, it is a literal matplotlib color applied to
        every node. A callable receives each node's attribute row and returns
        either a number (mapped through `cmap`, colorbar available) or a
        literal color (used as-is, no colorbar). If None, matplotlib's default
        color is used.
    edge_color : str | Callable | None, optional
        Marker edge (border) color, resolved exactly like `color`: an attribute
        name or numeric callable output is mapped through the shared
        `cmap`/`color_norm`, a literal color or a callable returning literal
        colors is used as-is. If None, edges take matplotlib's default (the
        face color). Set the border width with `scatter_kwargs={"linewidths": ...}`.
    cmap : str | Colormap, optional
        Colormap used when `color` yields numeric values.
    color_norm : Normalize | tuple[float, float] | None, optional
        Normalization for numeric colors, either a matplotlib `Normalize`
        instance or a `(vmin, vmax)` tuple. If None, the data range is used.
        A single shared normalization is applied across all marker groups.
    size : str | Callable | float, optional
        Marker size. A string names a numeric node attribute mapped into
        `size_range`. A callable receives each node's attribute row and returns
        the marker size in points**2 directly. A number sets a constant size
        in points**2 for every node.
    size_norm : Normalize | tuple[float, float] | None, optional
        Normalization of attribute values onto `size_range`, used when `size`
        is an attribute name: a matplotlib `Normalize` instance or a
        `(vmin, vmax)` tuple (values outside are clipped). If None, the data
        range is used.
    size_range : tuple[float, float], optional
        The marker sizes in points**2 assigned to the smallest and largest
        values when `size` is an attribute name.
    marker : str | Callable | None, optional
        Marker shape. A string is a single matplotlib marker glyph (e.g. "s")
        applied to every node. A callable receives each node's attribute row
        and returns the marker glyph for that node; nodes are grouped by glyph
        and drawn with one `Axes.scatter` call per group. If None, "o" is used.
    text : str | Callable | None, optional
        Per-node text label. A string names a node attribute whose value is
        annotated at each node. A callable receives each node's attribute row
        and returns the label. If None, no labels are drawn. Labels are drawn
        per node and can clutter large trees.
    text_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments forwarded to `Axes.annotate` for the text
        labels (e.g. `fontsize`, `color`, `xytext`).
    attrs : Sequence[str] | None, optional
        Extra node attribute keys to load so the `color`/`size`/`marker`/`text`
        callables can read them. If a callable is passed but `attrs` is None, a
        warning is emitted and all node attributes are loaded, which may be slow
        or memory-heavy (e.g. mask attributes). Ignored keys already loaded for
        other reasons are harmless.
    time_points : Sequence[int] | None, optional
        Time points at which markers are drawn, e.g. `range(10, 21)` for a
        contiguous window or `[0, 5, 10]` for a sparse subset. Edges are drawn
        for every node between the first and last displayed time point, hidden
        ones included. If None, all time points are displayed.
    time_positions : Mapping[int, float] | ArrayLike | None, optional
        Exact positions of the time points along the time axis
        (e.g. acquisition timestamps). Either a mapping of time point to
        position or a sequence indexed by time point, covering every time
        point within the displayed range. If None, every time point within
        the displayed range is evenly separated, hidden ones included, and
        the displayed ones are labeled with their values.
    orientation : {"vertical", "horizontal"}, optional
        If "vertical", time runs downward along the y-axis.
        If "horizontal", time runs rightward along the x-axis.
    scatter_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments forwarded to `Axes.scatter`,
        e.g. `linewidths` to set the marker border width or `alpha`.
        `c`, `edgecolors`, `s`, and `marker` are set from the aesthetics
        above and take precedence.
    line_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments forwarded to the edge
        `LineCollection` (e.g. `color`, `linewidth`).

    Returns
    -------
    Axes
        The matplotlib axes containing the lineage tree. When `color` yields
        numeric values, the last node `PathCollection` in `Axes.collections`
        is a colorbar-compatible mapping (all marker groups share the same
        normalization and colormap).

    Examples
    --------
    Continuous color and size from an attribute, with a colorbar:

    ```python
    from tracksdata.functional import plot_lineage_tree

    ax = plot_lineage_tree(graph, color="area", cmap="magma", size="area")
    ax.figure.colorbar(ax.collections[-1], ax=ax, label="area")
    ```

    Color by a computed quantity (still colorbar-compatible) and shape markers
    by a categorical attribute:

    ```python
    import numpy as np

    ax = plot_lineage_tree(
        graph,
        color=lambda row: np.log1p(row["area"]),
        marker=lambda row: "s" if row["is_dividing"] else "o",
        attrs=["area", "is_dividing"],
    )
    ```

    Outline dividing cells on top of a continuous face color:

    ```python
    ax = plot_lineage_tree(
        graph,
        color="area",
        edge_color=lambda row: "black" if row["is_dividing"] else "none",
        attrs=["is_dividing"],
        scatter_kwargs={"linewidths": 1.5},
    )
    ```

    Categorical colors and per-node text labels:

    ```python
    palette = {"A": "tab:red", "B": "tab:blue"}
    ax = plot_lineage_tree(
        graph,
        color=lambda row: palette[row["class"]],
        text=lambda row: row["class"],
        attrs=["class"],
    )
    ```

    Display only a time window with timestamps in seconds:

    ```python
    ax = plot_lineage_tree(
        graph,
        time_points=range(10, 21),
        time_positions={t: t * 30.0 for t in range(50)},
    )
    ```
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize, is_color_like
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for `plot_lineage_tree`. "
            "Install it with `pip install matplotlib` or `pip install 'tracksdata[plot]'`."
        ) from e

    if orientation not in ("vertical", "horizontal"):
        raise ValueError(f"`orientation` must be 'vertical' or 'horizontal', got '{orientation}'.")
    vertical = orientation == "vertical"

    node_attr_keys = graph.node_attr_keys()
    if tracklet_id_key not in node_attr_keys:
        graph.assign_tracklet_ids(tracklet_id_key)

    has_callable = any(callable(spec) for spec in (color, edge_color, size, marker, text))

    # a `color`/`edge_color` string is an attribute name when one exists,
    # otherwise it must be a literal matplotlib color
    color_attr_keys = []
    for spec in (color, edge_color):
        if not isinstance(spec, str):
            continue
        if spec in node_attr_keys:
            color_attr_keys.append(spec)
        elif not is_color_like(spec):
            raise ValueError(
                f"Color '{spec}' not found in graph attributes and is not a valid matplotlib color. "
                f"Expected a color or one of {node_attr_keys}"
            )

    # attribute names referenced directly (string aesthetics) plus any extra
    # keys the callables need. `marker` as a string is a matplotlib glyph, not
    # an attribute name, so it is not loaded.
    requested = color_attr_keys + [spec for spec in (size, text) if isinstance(spec, str)] + list(attrs or [])
    missing = [key for key in requested if key not in node_attr_keys]
    if missing:
        raise ValueError(f"Attributes {missing} not found in graph. Expected one of {node_attr_keys}")
    if has_callable and attrs is None:
        warnings.warn(
            "A `color`/`edge_color`/`size`/`marker`/`text` callable was given without `attrs`; "
            "loading all node attributes, which may be slow or memory-heavy "
            "(e.g. mask attributes). Pass `attrs=[...]` to load only the keys the callables need.",
            stacklevel=2,
        )
        requested = node_attr_keys
    attr_keys = list(dict.fromkeys([DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.T, tracklet_id_key, *requested]))

    all_nodes_df = graph.node_attrs(attr_keys=attr_keys)

    shown_time_points = all_nodes_df[DEFAULT_ATTR_KEYS.T].unique().sort()
    if time_points is not None:
        shown_time_points = shown_time_points.filter(shown_time_points.is_in(list(time_points)))
    shown_time_points = shown_time_points.to_list()
    if not shown_time_points:
        raise ValueError("No nodes to plot. The graph is empty or `time_points` excluded all nodes.")

    # tree-axis coordinate per tracklet, computed on the full graph so the
    # layout is independent of the displayed time range
    tracklet_positions = _tracklet_tree_layout(graph.tracklet_graph(tracklet_id_key=tracklet_id_key))

    # edges run through every node inside the displayed time span, hidden time
    # points included, so divisions appear at their true time
    span_df = all_nodes_df.filter(pl.col(DEFAULT_ATTR_KEYS.T).is_between(shown_time_points[0], shown_time_points[-1]))
    span_time_points = span_df[DEFAULT_ATTR_KEYS.T].unique().sort().to_list()
    time_axis_positions = _time_axis_positions(span_time_points, time_positions)

    tree_coords = span_df[tracklet_id_key].replace_strict(tracklet_positions, return_dtype=pl.Float64).to_numpy()
    time_coords = span_df[DEFAULT_ATTR_KEYS.T].replace_strict(time_axis_positions, return_dtype=pl.Float64).to_numpy()
    span_x, span_y = (tree_coords, time_coords) if vertical else (time_coords, tree_coords)

    # edge segments as (E, 2, 2) start/end points; the inner joins drop edges
    # with an endpoint outside the span
    node_id, source, target = DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET
    coords_df = pl.DataFrame({node_id: span_df[node_id], "x": span_x, "y": span_y})
    segments_df = (
        graph.edge_attrs(attr_keys=[])
        .join(coords_df.rename({node_id: source, "x": "x0", "y": "y0"}), on=source)
        .join(coords_df.rename({node_id: target, "x": "x1", "y": "y1"}), on=target)
    )
    segments = segments_df.select("x0", "y0", "x1", "y1").to_numpy().reshape(-1, 2, 2)

    # markers only at the displayed time points
    is_shown = span_df[DEFAULT_ATTR_KEYS.T].is_in(shown_time_points).to_numpy()
    nodes_df = span_df.filter(is_shown)
    x_coords, y_coords = span_x[is_shown], span_y[is_shown]

    if ax is None:
        _, ax = plt.subplots()

    line_kwargs = {"color": "0.6", "linewidth": 1.0, "zorder": 1, **(line_kwargs or {})}
    ax.add_collection(LineCollection(segments, **line_kwargs))

    # per-node attribute rows, only materialized when a callable needs them
    rows = list(nodes_df.iter_rows(named=True)) if has_callable else []

    # resolve the face color channel (scatter's `c`) and the edge color channel
    color_values, color_is_scalar = _resolve_color_channel(color, nodes_df, rows)
    edge_values, edge_is_scalar = _resolve_color_channel(edge_color, nodes_df, rows)

    # a single shared normalization so colors are consistent across marker
    # groups and between face and edge colors
    norm: Normalize | None = None
    scalar_channels = [
        values for values, is_scalar in ((color_values, color_is_scalar), (edge_values, edge_is_scalar)) if is_scalar
    ]
    if scalar_channels:
        if color_norm is None:
            stacked = np.concatenate(scalar_channels)
            norm = Normalize(vmin=float(np.nanmin(stacked)), vmax=float(np.nanmax(stacked)))
        elif isinstance(color_norm, tuple):
            norm = Normalize(*color_norm)
        else:
            norm = color_norm

    # scatter only colormaps `c`, so numeric edge colors are mapped here
    edge_cmap = plt.get_cmap(cmap) if edge_is_scalar else None

    # resolve the size channel to one size per node: attribute name -> linearly
    # mapped into `size_range`, callable -> raw sizes, number -> constant
    if callable(size):
        size_values = np.asarray([size(row) for row in rows], dtype=float)
    elif isinstance(size, str):
        if isinstance(size_norm, Normalize):
            size_scale = size_norm
        else:
            # autoscales to the data range when no limits are given
            size_scale = Normalize(*(size_norm or (None, None)), clip=True)
        fraction = size_scale(np.ma.masked_invalid(nodes_df[size].to_numpy().astype(float)))
        size_values = np.ma.filled(size_range[0] + fraction * (size_range[1] - size_range[0]), np.nan)
    else:
        size_values = np.full(len(nodes_df), float(size))

    # resolve the marker channel: callable -> per-node glyphs, string -> single
    # glyph, None -> "o"
    if callable(marker):
        marker_values = [marker(row) for row in rows]
    else:
        marker_values = [marker or "o"] * len(nodes_df)
    marker_arr = np.asarray(marker_values, dtype=object)

    scatter_kwargs = {"zorder": 2, **(scatter_kwargs or {})}

    # one scatter call per distinct glyph (scatter accepts a single marker)
    for glyph in dict.fromkeys(marker_values):
        idx = np.nonzero(marker_arr == glyph)[0]
        kwargs = dict(scatter_kwargs)
        if color_is_scalar:
            kwargs["c"] = color_values[idx]
            kwargs["cmap"] = cmap
            kwargs["norm"] = norm
        elif color_values is not None:
            kwargs["c"] = [color_values[i] for i in idx]
        if edge_is_scalar:
            kwargs["edgecolors"] = edge_cmap(norm(edge_values[idx]))
        elif edge_values is not None:
            kwargs["edgecolors"] = [edge_values[i] for i in idx]
        kwargs["s"] = size_values[idx]
        ax.scatter(x_coords[idx], y_coords[idx], marker=glyph, **kwargs)

    if text is not None:
        if callable(text):
            labels = [text(row) for row in rows]
        else:
            labels = nodes_df[text].to_list()
        annotate_kwargs = {
            "fontsize": 8,
            "xytext": (3.0, 0.0),
            "textcoords": "offset points",
            **(text_kwargs or {}),
        }
        for x, y, label in zip(x_coords, y_coords, labels, strict=True):
            ax.annotate(str(label), (x, y), **annotate_kwargs)

    time_axis, tree_axis = (ax.yaxis, ax.xaxis) if vertical else (ax.xaxis, ax.yaxis)
    time_axis.set_label_text("time")
    tree_axis.set_ticks([])
    # time runs downward in the vertical layout
    if vertical and not ax.yaxis_inverted():
        ax.invert_yaxis()

    if time_positions is None:
        # evenly separated positions: label the ticks with the time point values
        stride = max(1, len(shown_time_points) // 10)
        ticks = shown_time_points[::stride]
        time_axis.set_ticks([time_axis_positions[t] for t in ticks], labels=[str(t) for t in ticks])

    return ax

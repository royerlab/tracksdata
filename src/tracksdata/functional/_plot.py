"""Matplotlib-based plotting utilities for lineage trees."""

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import rustworkx as rx
from numpy.typing import ArrayLike

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize

__all__ = ["plot_lineage_tree"]


def _resolve_color_values(raw: list) -> tuple[Any, bool]:
    """
    Interpret per-node color-callable outputs as either scalars or literal colors.

    Parameters
    ----------
    raw : list
        One value per node, as returned by a `color` callable.

    Returns
    -------
    tuple[Any, bool]
        `(values, is_scalar)`. If the outputs form a 1-D numeric array,
        `values` is that array and `is_scalar` is True, so they are mapped
        through a colormap (and support a colorbar). Otherwise `values` is
        the original list of literal colors (names, hex, or RGB(A) tuples)
        and `is_scalar` is False.
    """
    try:
        arr = np.asarray(raw, dtype=float)
    except (ValueError, TypeError):
        return list(raw), False
    if arr.ndim == 1:
        return arr, True
    # (N, 3) or (N, 4): literal RGB(A) colors, not colormap-able scalars
    return list(raw), False


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
        # iterative post-order traversal: children are positioned before parents
        stack: list[tuple[int, bool]] = [(root, False)]
        while stack:
            rx_id, expanded = stack.pop()
            if expanded:
                children_pos = [
                    positions[tracklet_graph[child]]
                    for child in tracklet_graph.successor_indices(rx_id)
                    if tracklet_graph[child] in positions
                ]
                if children_pos:
                    positions[tracklet_graph[rx_id]] = float(np.mean(children_pos))
                else:
                    positions[tracklet_graph[rx_id]] = next_leaf
                    next_leaf += 1.0
            elif rx_id not in visited:
                visited.add(rx_id)
                stack.append((rx_id, True))
                for child in sorted(
                    tracklet_graph.successor_indices(rx_id),
                    key=tracklet_graph.__getitem__,
                    reverse=True,
                ):
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
        Sorted unique time points to be displayed.
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


def _map_to_size_range(
    values: np.ndarray,
    size_norm: tuple[float, float] | None,
    size_range: tuple[float, float],
) -> np.ndarray:
    """
    Linearly map attribute values to marker sizes within `size_range`.

    Parameters
    ----------
    values : np.ndarray
        Attribute values to map.
    size_norm : tuple[float, float] | None
        The (vmin, vmax) values mapped to the limits of `size_range`.
        If None, the minimum and maximum of `values` are used.
    size_range : tuple[float, float]
        The (smallest, largest) marker sizes in points**2.

    Returns
    -------
    np.ndarray
        Marker sizes, one per value.
    """
    values = np.asarray(values, dtype=float)
    if size_norm is None:
        vmin, vmax = np.nanmin(values), np.nanmax(values)
    else:
        vmin, vmax = size_norm

    smin, smax = size_range
    if vmax <= vmin:
        return np.full(values.shape, (smin + smax) / 2)

    fraction = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    return smin + fraction * (smax - smin)


def _bridged_edge_segments(
    successors: dict[int, list[int]],
    node_coords: dict[int, tuple[float, float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """
    Build edge segments between displayed nodes, bridging across hidden ones.

    Each displayed node is connected to its nearest displayed descendants by
    walking forward through the tracking graph and skipping over nodes that are
    not displayed. This keeps the lineage structure visible when only a subset
    of time points is shown. When all nodes are displayed it reduces to the
    direct edges of the graph.

    Parameters
    ----------
    successors : dict[int, list[int]]
        Forward adjacency of the full (sub)graph, mapping each source node id
        to the list of its target node ids.
    node_coords : dict[int, tuple[float, float]]
        Plot coordinates of the displayed nodes, keyed by node id.

    Returns
    -------
    list[tuple[tuple[float, float], tuple[float, float]]]
        Line segments connecting the coordinates of displayed nodes.
    """
    segments = []
    for source in node_coords:
        # walk forward to the nearest displayed descendants, skipping hidden nodes
        stack = list(successors.get(source, ()))
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node in node_coords:
                segments.append((node_coords[source], node_coords[node]))
            else:
                stack.extend(successors.get(node, ()))
    return segments


def plot_lineage_tree(
    graph: BaseGraph,
    *,
    ax: "Axes | None" = None,
    tracklet_id_key: str = DEFAULT_ATTR_KEYS.TRACKLET_ID,
    color: "str | Callable[[Mapping[str, Any]], Any] | None" = None,
    cmap: "str | Colormap" = "viridis",
    color_norm: "Normalize | tuple[float, float] | None" = None,
    size: "str | Callable[[Mapping[str, Any]], float] | float | None" = None,
    size_norm: tuple[float, float] | None = None,
    size_range: tuple[float, float] = (10.0, 100.0),
    node_size: float = 30.0,
    marker: "str | Callable[[Mapping[str, Any]], str] | None" = None,
    text: "str | Callable[[Mapping[str, Any]], Any] | None" = None,
    text_kwargs: dict[str, Any] | None = None,
    attrs: Sequence[str] | None = None,
    time_range: tuple[int, int] | None = None,
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
    subset of time points is shown, each node is connected to its nearest
    displayed descendants, bridging over the hidden time points so the lineage
    stays connected.

    The `color`, `size`, `marker`, and `text` aesthetics each accept either a
    fixed value or a callable, which is the main way to customize the markers:

    - As a string, `color`/`size`/`text` name a numeric node attribute, and
      `marker` is a single matplotlib marker glyph applied to every node.
    - As a callable, they receive each node's attribute row (a mapping of
      attribute key to value) and return that node's color, size, marker glyph,
      or text label. This allows categorical colors, per-node marker shapes,
      and colors derived from a computed quantity (e.g. `np.log1p(row["area"])`).

    A colorbar-compatible mapping is available whenever `color` produces numeric
    values (a numeric attribute name, or a callable returning numbers) together
    with `cmap`. If a callable returns literal colors (names, hex, or RGB(A)),
    those colors are used verbatim and no colorbar mapping exists.

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
        Marker color. A string names a numeric node attribute mapped through
        `cmap`/`color_norm` (a colorbar mapping is available). A callable
        receives each node's attribute row and returns either a number (mapped
        through `cmap`, colorbar available) or a literal color (used as-is, no
        colorbar). If None, matplotlib's default color is used.
    cmap : str | Colormap, optional
        Colormap used when `color` yields numeric values.
    color_norm : Normalize | tuple[float, float] | None, optional
        Normalization for numeric colors, either a matplotlib `Normalize`
        instance or a `(vmin, vmax)` tuple. If None, the data range is used.
        A single shared normalization is applied across all marker groups.
    size : str | Callable | float | None, optional
        Marker size. A string names a numeric node attribute mapped into
        `size_range`. A callable receives each node's attribute row and returns
        the marker size in points**2 directly. A number sets a constant size.
        If None, `node_size` is used.
    size_norm : tuple[float, float] | None, optional
        The `(vmin, vmax)` attribute values mapped to the limits of
        `size_range`, used when `size` is an attribute name. If None, the data
        range is used.
    size_range : tuple[float, float], optional
        The marker sizes in points**2 assigned to the smallest and largest
        values when `size` is an attribute name.
    node_size : float, optional
        Marker size in points**2 used when `size` is None.
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
    time_range : tuple[int, int] | None, optional
        Inclusive `(start, end)` range of time points to display.
        If None, all time points are displayed. Mutually exclusive with
        `time_points`.
    time_points : Sequence[int] | None, optional
        Explicit subset of time points to display, which need not be
        contiguous (e.g. `[0, 5, 10]`). Edges bridge over the hidden time
        points, connecting each displayed node to its nearest displayed
        descendants. Mutually exclusive with `time_range`.
    time_positions : Mapping[int, float] | ArrayLike | None, optional
        Exact positions of the time points along the time axis
        (e.g. acquisition timestamps). Either a mapping of time point to
        position or a sequence indexed by time point. If None, the displayed
        time points are evenly separated and labeled with their values.
    orientation : {"vertical", "horizontal"}, optional
        If "vertical", time runs downward along the y-axis.
        If "horizontal", time runs rightward along the x-axis.
    scatter_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments forwarded to `Axes.scatter`,
        e.g. `edgecolors` and `linewidths` to style the marker borders.
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
        time_range=(10, 20),
        time_positions={t: t * 30.0 for t in range(50)},
    )
    ```
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for `plot_lineage_tree`. "
            "Install it with `pip install matplotlib` or `pip install 'tracksdata[plot]'`."
        ) from e

    if orientation not in ("vertical", "horizontal"):
        raise ValueError(f"`orientation` must be 'vertical' or 'horizontal', got '{orientation}'.")

    if tracklet_id_key not in graph.node_attr_keys():
        graph.assign_tracklet_ids(tracklet_id_key)

    has_callable = any(callable(spec) for spec in (color, size, marker, text))

    attr_keys = [DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.T, tracklet_id_key]
    if has_callable and attrs is None:
        warnings.warn(
            "A `color`/`size`/`marker`/`text` callable was given without `attrs`; "
            "loading all node attributes, which may be slow or memory-heavy "
            "(e.g. mask attributes). Pass `attrs=[...]` to load only the keys the callables need.",
            stacklevel=2,
        )
        for key in graph.node_attr_keys():
            if key not in attr_keys:
                attr_keys.append(key)
    else:
        # attribute names referenced directly (string aesthetics) plus any
        # extra keys the callables need. `marker` as a string is a matplotlib
        # glyph, not an attribute name, so it is not loaded.
        requested = [spec for spec in (color, size, text) if isinstance(spec, str)]
        if attrs is not None:
            requested.extend(attrs)
        for key in requested:
            if key in attr_keys:
                continue
            if key not in graph.node_attr_keys():
                raise ValueError(f"Attribute '{key}' not found in graph. Expected one of {graph.node_attr_keys()}")
            attr_keys.append(key)

    if time_range is not None and time_points is not None:
        raise ValueError("`time_range` and `time_points` are mutually exclusive, provide at most one.")

    nodes_df = graph.node_attrs(attr_keys=attr_keys)

    if time_range is not None:
        start, end = time_range
        nodes_df = nodes_df.filter((nodes_df[DEFAULT_ATTR_KEYS.T] >= start) & (nodes_df[DEFAULT_ATTR_KEYS.T] <= end))
    elif time_points is not None:
        nodes_df = nodes_df.filter(nodes_df[DEFAULT_ATTR_KEYS.T].is_in(list(time_points)))

    if len(nodes_df) == 0:
        raise ValueError("No nodes to plot. The graph is empty or `time_range`/`time_points` excluded all nodes.")

    # tree-axis coordinate per tracklet, computed on the full graph so the
    # layout is independent of the displayed time range
    tracklet_positions = _tracklet_tree_layout(graph.tracklet_graph(tracklet_id_key=tracklet_id_key))

    time_points = nodes_df[DEFAULT_ATTR_KEYS.T].unique().sort().to_list()
    time_axis_positions = _time_axis_positions(time_points, time_positions)

    tree_coords = np.asarray([tracklet_positions[tid] for tid in nodes_df[tracklet_id_key]])
    time_coords = np.asarray([time_axis_positions[t] for t in nodes_df[DEFAULT_ATTR_KEYS.T]])

    if orientation == "vertical":
        x_coords, y_coords = tree_coords, time_coords
    else:
        x_coords, y_coords = time_coords, tree_coords

    node_coords = {
        node_id: (x, y) for node_id, x, y in zip(nodes_df[DEFAULT_ATTR_KEYS.NODE_ID], x_coords, y_coords, strict=True)
    }

    edges_df = graph.edge_attrs(attr_keys=[])
    successors: dict[int, list[int]] = {}
    for source, target in zip(
        edges_df[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list(),
        edges_df[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list(),
        strict=True,
    ):
        successors.setdefault(source, []).append(target)

    segments = _bridged_edge_segments(successors, node_coords)

    if ax is None:
        _, ax = plt.subplots()

    line_kwargs = {"color": "0.6", "linewidth": 1.0, "zorder": 1, **(line_kwargs or {})}
    ax.add_collection(LineCollection(segments, **line_kwargs))

    # per-node attribute rows, only materialized when a callable needs them
    rows = list(nodes_df.iter_rows(named=True)) if has_callable else []

    # resolve the color channel to values passed to scatter's `c`
    color_values: Any = None
    color_is_scalar = False
    if color is not None:
        if callable(color):
            color_values, color_is_scalar = _resolve_color_values([color(row) for row in rows])
        else:
            color_values = nodes_df[color].to_numpy()
            color_is_scalar = True

    # a single shared normalization so colors are consistent across marker groups
    norm: Normalize | None = None
    if color_is_scalar:
        if color_norm is None:
            norm = Normalize(vmin=float(np.nanmin(color_values)), vmax=float(np.nanmax(color_values)))
        elif isinstance(color_norm, tuple):
            norm = Normalize(*color_norm)
        else:
            norm = color_norm

    # resolve the size channel: attribute name -> mapped range, callable -> raw
    # sizes, number -> constant, None -> node_size default
    if size is None:
        size_values: Any = None
    elif callable(size):
        size_values = np.asarray([size(row) for row in rows], dtype=float)
    elif isinstance(size, str):
        size_values = _map_to_size_range(nodes_df[size].to_numpy(), size_norm, size_range)
    else:
        size_values = float(size)

    # resolve the marker channel: callable -> per-node glyphs (grouped), string
    # -> single glyph, None -> "o"
    if callable(marker):
        marker_values = [marker(row) for row in rows]
    else:
        marker_values = None
        single_marker = marker if isinstance(marker, str) else "o"

    scatter_kwargs = {"zorder": 2, **(scatter_kwargs or {})}

    def _scatter_group(idx: np.ndarray, marker_glyph: str) -> Any:
        kwargs = dict(scatter_kwargs)
        if color_is_scalar:
            kwargs["c"] = color_values[idx]
            kwargs["cmap"] = cmap
            kwargs["norm"] = norm
        elif color_values is not None:
            kwargs["c"] = [color_values[i] for i in idx]
        if size_values is None:
            kwargs.setdefault("s", node_size)
        elif np.isscalar(size_values):
            kwargs.setdefault("s", size_values)
        else:
            kwargs["s"] = size_values[idx]
        return ax.scatter(x_coords[idx], y_coords[idx], marker=marker_glyph, **kwargs)

    if marker_values is None:
        _scatter_group(np.arange(len(nodes_df)), single_marker)
    else:
        marker_arr = np.asarray(marker_values, dtype=object)
        # one scatter call per distinct glyph (scatter accepts a single marker)
        for glyph in dict.fromkeys(marker_values):
            idx = np.nonzero(marker_arr == glyph)[0]
            _scatter_group(idx, glyph)

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

    if orientation == "vertical":
        time_axis, tree_axis = ax.yaxis, ax.xaxis
        ax.set_ylabel("time")
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
    else:
        time_axis, tree_axis = ax.xaxis, ax.yaxis
        ax.set_xlabel("time")

    tree_axis.set_ticks([])

    if time_positions is None:
        # evenly separated positions: label the ticks with the time point values
        stride = max(1, len(time_points) // 10)
        ticks = time_points[::stride]
        time_axis.set_ticks([time_axis_positions[t] for t in ticks], labels=[str(t) for t in ticks])

    return ax

import logging
import time
from collections.abc import Callable
from pathlib import Path

import polars as pl
from tabulate import tabulate

from tracksdata.attrs import EdgeAttr, NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges import DistanceEdges
from tracksdata.graph import IndexedRXGraph, RustWorkXGraph, SQLGraph
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes import RandomNodes
from tracksdata.options import get_options, set_options
from tracksdata.solvers import NearestNeighborsSolver
from tracksdata.utils._logging import LOG


class SQLGraphWithMemory(SQLGraph):
    def __init__(self):
        super().__init__(drivername="sqlite", database=":memory:", overwrite=True)


class SQLGraphDisk(SQLGraph):
    def __init__(self):
        import datetime

        path = f"/tmp/_benchmarks_tracksdata_db_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        super().__init__(drivername="sqlite", database=path, overwrite=True)


def _run_benchmark(
    backend: type[BaseGraph],
    pipeline: list[tuple[str, Callable[[BaseGraph], None]]],
) -> pl.DataFrame:
    data = []
    total_time = 0

    start = time.perf_counter()
    graph = backend()
    end = time.perf_counter()

    data.append(
        {
            "operation": "init",
            "time": end - start,
        }
    )
    total_time += end - start

    for name, func in pipeline:
        start = time.perf_counter()
        output = func(graph)
        # replace graph with output if it is not None
        # it's used for the subgraph operation
        if output is not None:
            graph = output
        end = time.perf_counter()
        data.append(
            {
                "operation": name,
                "time": end - start,
            }
        )
        total_time += end - start

    data.append(
        {
            "operation": "total",
            "time": total_time,
        }
    )

    df = pl.DataFrame(data)
    df = df.with_columns(
        pl.col("operation").cast(pl.Categorical),
        pl.col("time").cast(pl.Float64),
    )
    return df


def _format_markdown_table(df: pl.DataFrame, output_file: Path | None = None) -> str:
    # Create time string column
    df = df.with_columns(time_str=pl.format("{} ± {}", pl.col("time_avg").round(3), pl.col("time_std").round(3)))

    # Pivot the table to get n_nodes as columns
    pivoted = df.pivot(values="time_str", index=["backend", "operation"], on="n_nodes", aggregate_function="first")

    # Format the results for markdown table
    prev_backend = pivoted["backend"][0]
    table_data = []
    for row in pivoted.iter_rows(named=True):
        if row["backend"] != prev_backend:
            table_data.append([""] * (len(df["n_nodes"].unique()) + 2))  # Empty row for separation
        table_data.append([row["backend"], row["operation"], *[row[str(n)] for n in sorted(df["n_nodes"].unique())]])
        prev_backend = row["backend"]

    # Print markdown table
    node_counts = sorted(df["n_nodes"].unique())
    headers = ["Backend", "Operation"] + [f"{n:,} nodes" for n in node_counts]

    print(f"\n| {get_options().n_workers} worker(s)")
    print("|:-----------")

    mk_table = tabulate(table_data, headers=headers, tablefmt="pipe", numalign="right")
    print(mk_table)

    if output_file is not None:
        with open(output_file, "w") as f:
            f.write(mk_table)
            f.write("\n")

    return mk_table


def _build_pipeline(
    n_time_points: int, n_nodes_per_tp: int
) -> list[tuple[str, Callable[[BaseGraph], None | BaseGraph]]]:
    return [
        (
            "random_nodes",
            RandomNodes(
                n_time_points=n_time_points,
                n_nodes_per_tp=(n_nodes_per_tp * 0.95, n_nodes_per_tp * 1.05),
                n_dim=3,
            ).add_nodes,
        ),
        ("distance_edges", DistanceEdges(distance_threshold=10, n_neighbors=5).add_edges),
        (
            "nearest_neighbors_solver",
            NearestNeighborsSolver(
                edge_weight=-EdgeAttr(DEFAULT_ATTR_KEYS.EDGE_DIST),
                max_children=2,
                return_solution=False,
            ).solve,
        ),
        (
            "subgraph",
            lambda graph: graph.filter(
                NodeAttr(DEFAULT_ATTR_KEYS.SOLUTION) == True,
                EdgeAttr(DEFAULT_ATTR_KEYS.SOLUTION) == True,
            ).subgraph(),
        ),
        ("assing_tracks", lambda graph: graph.assign_track_ids()),
    ]


def main() -> None:
    LOG.setLevel(logging.CRITICAL)

    n_repeats = 3
    n_time_points = 50
    first_round = True
    set_options(show_progress=False)

    for n_workers in [4, 1]:
        data = []
        set_options(n_workers=n_workers)
        for _ in range(n_repeats):
            for n_nodes in [1_000, 10_000, 100_000]:
                n_nodes_per_tp = int(n_nodes / n_time_points)
                pipeline = _build_pipeline(n_time_points, n_nodes_per_tp)
                for backend in [RustWorkXGraph, IndexedRXGraph, SQLGraphWithMemory, SQLGraphDisk]:
                    # SQLGraphWithMemory does not support multiprocessing
                    if backend == SQLGraphWithMemory and n_workers > 1:
                        continue

                    df = _run_benchmark(backend, pipeline)

                    if first_round:
                        # first round is re-executed because it required compiling numba code
                        df = _run_benchmark(backend, pipeline)
                        first_round = False

                    df = df.with_columns(
                        backend=pl.lit(backend.__name__),
                        n_nodes=pl.lit(n_nodes),
                    )
                    data.append(df)

        df = pl.concat(data)
        df = df.group_by(["backend", "n_nodes", "operation"], maintain_order=True).agg(
            pl.col("time").std().alias("time_std"),
            pl.col("time").mean().alias("time_avg"),
        )

        file_path = Path(__file__)
        _format_markdown_table(
            df,
            output_file=file_path.parent / f"outputs/{file_path.stem}.md",
        )


if __name__ == "__main__":
    main()

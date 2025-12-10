from os import environ

from asv_runner.benchmarks.mark import skip_benchmark

import tracksdata as td

# With subclassing, the asv calls "time_point" function as a benchmark.
# So we hook into it to skip that.


class SQLGraphWithMemory(td.graph.SQLGraph):
    def __init__(self):
        super().__init__(drivername="sqlite", database=":memory:", overwrite=True)

    @skip_benchmark
    def time_points(self):
        return super().time_points()


class SQLGraphDisk(td.graph.SQLGraph):
    def __init__(self):
        import datetime

        path = f"/tmp/_benchmarks_tracksdata_db_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}+{id(self)}.db"
        super().__init__(drivername="sqlite", database=path, overwrite=True)

    @skip_benchmark
    def time_points(self):
        return super().time_points()


BACKENDS = {
    "RustWorkXGraph": td.graph.RustWorkXGraph,
    "IndexedRXGraph": td.graph.IndexedRXGraph,
    "SQLGraphWithMemory": SQLGraphWithMemory,
    "SQLGraphDisk": SQLGraphDisk,
}

IS_CI = environ.get("CI", "false").lower() == "true"

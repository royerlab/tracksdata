import tracksdata as td

if __name__ == "__main__":
    from common import BACKENDS  # For local testing
else:
    from benchmarks.common import BACKENDS

NODE_COUNT = 1_000_000

class SQLIndexingBenchmark:
    def setup(self) -> None:
        graph : td.graph.SQLGraph = BACKENDS["SQLGraphDisk"]()
        graph.add_node_attr_key("attr1", 0)
        graph.bulk_add_nodes([{td.DEFAULT_ATTR_KEYS.T: i, "attr1": i%100} for i in range(NODE_COUNT)])
        if hasattr(graph, "ensure_node_attr_index"):
            graph.ensure_node_attr_index("attr1")
        self.graph = graph
 
    def time_sql_index_search(self) -> None:
        self.graph.filter(td.NodeAttr("attr1") == 0).subgraph()

if __name__ == "__main__":
    import cProfile

    sib = SQLIndexingBenchmark()
    sib.setup()
    with cProfile.Profile() as pr:
        sib.time_sql_index_search()
    pr.dump_stats("sql_indexing_result.pstat")
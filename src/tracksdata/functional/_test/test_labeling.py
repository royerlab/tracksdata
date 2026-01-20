import polars as pl

import tracksdata as td


def test_ancestral_connected_edges():
    r"""
    input_graph (fully connected graph, except within frames)

    0    1   2
    3    4   5
         6   7
    8        9

    reference graph

    0 ------- 1
    2 -- 3 -- 4
           \- 5
    6 -- 7 -- 8
    """
    input_graph = td.graph.IndexedRXGraph()
    input_graph.bulk_add_nodes(
        [{"t": t} for t in [0, 1, 2, 0, 1, 2, 1, 2, 0, 2]],
        indices=range(10),
    )
    edge_maps: dict[int, tuple[int, int]] = {}
    node_attrs = input_graph.node_attrs()
    for row_1 in node_attrs.rows(named=True):
        for row_2 in node_attrs.rows(named=True):
            if row_1[td.DEFAULT_ATTR_KEYS.T] < row_2[td.DEFAULT_ATTR_KEYS.T]:
                src, tgt = row_1[td.DEFAULT_ATTR_KEYS.NODE_ID], row_2[td.DEFAULT_ATTR_KEYS.NODE_ID]
                edge_maps[input_graph.add_edge(src, tgt, {})] = (src, tgt)

    ref_graph = td.graph.IndexedRXGraph()
    ref_graph.bulk_add_nodes(
        [{"t": t} for t in [0, 2, 0, 1, 2, 2, 0, 1, 2]],
        indices=range(9),
    )
    ref_graph.add_edge(0, 1, {})
    ref_graph.add_edge(2, 3, {})
    ref_graph.add_edge(3, 4, {})
    ref_graph.add_edge(3, 5, {})
    ref_graph.add_edge(6, 7, {})
    ref_graph.add_edge(7, 8, {})

    # manual matching
    input_graph.add_node_attr_key(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID, pl.Int64)
    input_graph.update_node_attrs(
        attrs={
            td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID: [0, 1, 2, 3, 4, 5, 6, 8],
        },
        node_ids=[0, 2, 3, 4, 5, 7, 8, 9],  # respective reference node ids
    )

    edge_ids = set(
        td.functional.ancestral_connected_edges(
            input_graph=input_graph,
            reference_graph=ref_graph,
            match=False,
        )
    )
    found_edges = {edge_maps[e] for e in edge_ids}

    expected_edges = {
        (0, 2),
        # (0, 1),  # they don't exist in the input graph
        # (1, 2),
        (3, 4),
        (3, 5),
        (4, 5),
        (3, 7),
        (4, 7),
        (8, 9),
    }

    assert found_edges == expected_edges

"""
Module to define default and often global values used through `tracksdata`.
"""


class DefaultAttrKeys:
    """
    This class defines the standard attribute names for nodes and edges in graphs to ensure
    consistency across different graph implementations and operators.

    Using these constants instead of hardcoded strings helps prevent typos.

    Attributes
    ----------
    NODE_ID : str
        Default key for node identifiers ("node_id").
    T : str
        Default key for time information ("t").
    MASK : str
        Default key for node masks ("mask").
    SOLUTION : str
        Default key for solution information ("solution").
    TRACK_ID : str
        Default key for track identifiers ("track_id").
    EDGE_ID : str
        Default key for edge identifiers ("edge_id").
    EDGE_WEIGHT : str
        Default key for edge weights ("weight").
    EDGE_SOURCE : str
        Default key for edge source node identifier ("source_id").
    EDGE_TARGET : str
        Default key for edge target node identifier ("target_id").
    MATCHED_NODE_ID : str
        Default key to identify respective node in another graph used for matching.
    MATCH_SCORE : str
        Default key between a node and its respective node in another graph used for matching.
    MATCHED_EDGE_MASK : str
        Default key for boolean mask indicating if edge exists in the matching graph.

    Examples
    --------
    ```python
    from tracksdata.constants import DEFAULT_ATTR_KEYS

    print(DEFAULT_ATTR_KEYS.NODE_ID)  # Output: node_id
    print(DEFAULT_ATTR_KEYS.EDGE_WEIGHT)  # Output: weight
    ```
    """

    NODE_ID = "node_id"
    T = "t"
    MASK = "mask"
    SOLUTION = "solution"
    TRACK_ID = "track_id"

    EDGE_ID = "edge_id"
    EDGE_WEIGHT = "weight"
    EDGE_SOURCE = "source_id"
    EDGE_TARGET = "target_id"

    MATCHED_NODE_ID = "match_node_id"
    MATCH_SCORE = "match_score"
    MATCHED_EDGE_MASK = "matched_edge_mask"


DEFAULT_ATTR_KEYS = DefaultAttrKeys()

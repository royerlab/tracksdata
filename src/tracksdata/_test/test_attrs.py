import math
import operator
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from tracksdata.attrs import (
    Attr,
    AttrComparison,
    EdgeAttr,
    NodeAttr,
    attr_comps_to_strs,
    polars_reduce_attr_comps,
    split_attr_comps,
)


def test_attr_expr_init_with_string() -> None:
    expr = Attr("test")
    assert isinstance(expr.expr, pl.Expr)
    assert expr.expr.meta.root_names() == ["test"]


def test_attr_expr_init_with_scalar() -> None:
    expr = Attr(1.0)
    assert isinstance(expr.expr, pl.Expr)
    # Literal expressions don't have root names
    assert expr.expr.meta.root_names() == []


def test_attr_expr_init_with_attr_expr() -> None:
    expr1 = Attr("test")
    expr2 = Attr(expr1).sqrt()
    assert isinstance(expr2.expr, pl.Expr)
    assert expr2.expr.meta.root_names() == ["test"]


def test_attr_expr_init_with_polars_expr() -> None:
    pl_expr = pl.col("test")
    expr = Attr(pl_expr)
    assert isinstance(expr.expr, pl.Expr)
    assert expr.expr.meta.root_names() == ["test"]


def test_attr_expr_evaluate() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    expr = Attr("a") + Attr("b")
    result = expr.evaluate(df)
    assert isinstance(result, pl.Series)
    assert result.to_list() == [5, 7, 9]


def test_attr_expr_column_names() -> None:
    expr = Attr("test")
    assert expr.columns == ["test"]


@pytest.mark.parametrize(
    "op,func",
    [
        (operator.neg, lambda x: -x),
        (operator.pos, lambda x: +x),
        (operator.abs, abs),
        (operator.invert, lambda x: ~x),
    ],
)
def test_attr_expr_unary_operators(op: Callable, func: Callable) -> None:
    df = pl.DataFrame({"a": [-1, 2, -3]})
    expr = op(Attr("a"))
    result = expr.evaluate(df)
    expected = pl.Series([func(x) for x in df["a"]])
    assert result.to_list() == expected.to_list()


@pytest.mark.parametrize(
    "op,func",
    [
        (operator.add, lambda x, y: x + y),
        (operator.sub, lambda x, y: x - y),
        (operator.mul, lambda x, y: x * y),
        (operator.truediv, lambda x, y: x / y),
        (operator.floordiv, lambda x, y: x // y),
        (operator.mod, lambda x, y: x % y),
        (operator.pow, lambda x, y: x**y),
        (operator.eq, lambda x, y: x == y),
        (operator.ne, lambda x, y: x != y),
        (operator.lt, lambda x, y: x < y),
        (operator.le, lambda x, y: x <= y),
        (operator.gt, lambda x, y: x > y),
        (operator.ge, lambda x, y: x >= y),
    ],
)
def test_attr_expr_binary_operators(op: Callable, func: Callable) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    expr = op(Attr("a"), Attr("b"))
    result = expr.evaluate(df)
    expected = pl.Series([func(x, y) for x, y in zip(df["a"], df["b"], strict=False)])
    assert result.to_list() == expected.to_list()


def test_attr_expr_alias() -> None:
    expr = Attr("test").alias("new_name")
    assert isinstance(expr, Attr)
    # Note: alias doesn't change root names
    assert expr.columns == ["test"]


def test_attr_expr_method_delegation() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    expr = Attr("a").log(2)
    result = expr.evaluate(df)
    expected = df.select(pl.col("a").log(2)).to_series()
    assert result.to_list() == expected.to_list()


def test_attr_expr_struct_field_method_delegation() -> None:
    df = pl.DataFrame({"s": [{"x": 1}, {"x": 2}, {"x": 3}]}, schema={"s": pl.Struct({"x": pl.Int64})})
    expr = NodeAttr("s").struct.field("x")
    result = expr.evaluate(df)
    assert isinstance(expr, NodeAttr)
    assert result.to_list() == [1, 2, 3]


def test_attr_comparison_struct_field() -> None:
    df = pl.DataFrame({"s": [{"x": 1}, {"x": 2}, {"x": 1}]}, schema={"s": pl.Struct({"x": pl.Int64})})
    comp = NodeAttr("s").struct.field("x") == 1
    result = comp.to_attr().evaluate(df)
    assert comp.column == "s"
    assert comp.attr.field_path == ("x",)
    assert result.to_list() == [True, False, True]


def test_attr_expr_complex_expression() -> None:
    df = pl.DataFrame({"iou": [0.5, 0.7, 0.9], "distance": [10, 20, 30]})
    expr = (1 - Attr("iou")) * Attr("distance")
    result = expr.evaluate(df)
    expected = [(1 - iou) * dist for iou, dist in zip(df["iou"], df["distance"], strict=False)]
    assert result.to_list() == expected


def test_attr_expr_with_infinity() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    expr = (Attr("a") == 1) * math.inf - math.inf * (Attr("b") > 4) + Attr("c")

    result = expr.evaluate(df)
    assert result.to_list() == [7, 8, 9]
    assert expr.expr_columns == ["c"]

    assert len(expr.inf_exprs) == 1
    assert expr.inf_exprs[0].expr_columns == ["a"]
    assert expr.inf_exprs[0].evaluate(df).to_list() == [True, False, False]

    assert len(expr.neg_inf_exprs) == 1
    assert expr.neg_inf_exprs[0].expr_columns == ["b"]
    assert expr.neg_inf_exprs[0].evaluate(df).to_list() == [False, True, True]


def test_attr_expr_multiple_positive_infinity() -> None:
    """Test expression with multiple positive infinity terms."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    expr = Attr("a") * math.inf + math.inf * Attr("b") + Attr("c")

    result = expr.evaluate(df)
    assert result.to_list() == [7, 8, 9]  # Only finite term remains
    assert expr.expr_columns == ["c"]

    assert len(expr.inf_exprs) == 2
    assert len(expr.neg_inf_exprs) == 0

    # Check both expressions are tracked
    inf_columns = set()
    for inf_expr in expr.inf_exprs:
        inf_columns.update(inf_expr.columns)
    assert inf_columns == {"a", "b"}


def test_attr_expr_multiple_negative_infinity() -> None:
    """Test expression with multiple negative infinity terms."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    expr = Attr("a") * (-math.inf) - math.inf * Attr("b") + Attr("c")

    result = expr.evaluate(df)
    assert result.to_list() == [7, 8, 9]  # Only finite term remains
    assert expr.expr_columns == ["c"]

    assert len(expr.inf_exprs) == 0
    assert len(expr.neg_inf_exprs) == 2

    # Check both expressions are tracked
    neg_inf_columns = set()
    for neg_inf_expr in expr.neg_inf_exprs:
        neg_inf_columns.update(neg_inf_expr.columns)
    assert neg_inf_columns == {"a", "b"}


def test_attr_expr_only_infinity_terms() -> None:
    """Test expression with only infinity terms (no finite terms)."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    expr = Attr("a") * math.inf - math.inf * Attr("b")

    result = expr.evaluate(df)
    assert result.to_list() == [0]  # All infinity terms become literal zero
    assert expr.expr_columns == []  # No finite columns

    assert len(expr.inf_exprs) == 1
    assert len(expr.neg_inf_exprs) == 1
    assert expr.inf_exprs[0].columns == ["a"]
    assert expr.neg_inf_exprs[0].columns == ["b"]


def test_attr_expr_complex_infinity_expressions() -> None:
    """Test infinity with more complex expressions (not just boolean comparisons)."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    expr = (Attr("a") + Attr("b")) * math.inf - math.inf * (Attr("c") / 2) + Attr("a")

    result = expr.evaluate(df)
    assert result.to_list() == [1, 2, 3]  # Only AttrExpr("a") remains
    assert set(expr.columns) == {"a", "b", "c"}
    assert expr.expr_columns == ["a"]

    assert len(expr.inf_exprs) == 1
    assert len(expr.neg_inf_exprs) == 1

    # Test that complex expressions are properly tracked
    inf_expr_result = expr.inf_exprs[0].evaluate(df)
    assert inf_expr_result.to_list() == [5, 7, 9]  # a + b
    assert set(expr.inf_exprs[0].columns) == {"a", "b"}

    neg_inf_expr_result = expr.neg_inf_exprs[0].evaluate(df)
    assert neg_inf_expr_result.to_list() == [3.5, 4.0, 4.5]  # c / 2
    assert expr.neg_inf_exprs[0].columns == ["c"]


def test_attr_expr_nested_infinity_operations() -> None:
    """Test nested operations involving infinity."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Create intermediate expressions with infinity
    inf_expr = Attr("a") * math.inf
    finite_expr = Attr("b") + 1

    # Combine them
    combined = inf_expr + finite_expr

    result = combined.evaluate(df)
    assert result.to_list() == [5, 6, 7]  # Only (b + 1) remains
    assert combined.expr_columns == ["b"]
    assert set(combined.columns) == {"a", "b"}

    assert len(combined.inf_exprs) == 1
    assert len(combined.neg_inf_exprs) == 0
    assert combined.inf_exprs[0].columns == ["a"]


def test_attr_expr_infinity_column_name_tracking() -> None:
    """Test that inf_columns and neg_inf_columns properties work correctly."""
    df = pl.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})

    # Simpler test: separate infinity operations that are easier to track
    expr = (
        Attr("x") * math.inf  # positive infinity
        + Attr("y") * math.inf  # positive infinity
        - math.inf * Attr("z")  # negative infinity
        + Attr("x")  # finite term
    )

    result = expr.evaluate(df)
    assert result.to_list() == [1, 2]  # Only finite AttrExpr("x") remains

    # Test the convenience properties
    assert set(expr.inf_columns) == {"x", "y"}  # x and y in positive infinity
    assert set(expr.neg_inf_columns) == {"z"}  # z in negative infinity

    assert expr.has_inf() is True

    # Check individual expressions
    assert len(expr.inf_exprs) == 2  # x and y
    assert len(expr.neg_inf_exprs) == 1  # z


def test_attr_expr_no_infinity_terms() -> None:
    """Test that expressions without infinity work normally."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    expr = Attr("a") * 2 + Attr("b") - 1

    result = expr.evaluate(df)
    assert result.to_list() == [5, 8, 11]  # 2*a + b - 1 = 2*[1,2,3] + [4,5,6] - 1 = [2,4,6] + [4,5,6] - 1 = [5,8,11]

    # No infinity terms should be tracked
    assert len(expr.inf_exprs) == 0
    assert len(expr.neg_inf_exprs) == 0
    assert expr.inf_columns == []
    assert expr.neg_inf_columns == []
    assert expr.has_inf() is False


def test_attr_expr_scalar_operations() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    expr = Attr("a") * 2
    result = expr.evaluate(df)
    assert result.to_list() == [2, 4, 6]

    expr = 2 * Attr("a")  # Test reverse operation
    result = expr.evaluate(df)
    assert result.to_list() == [2, 4, 6]


def test_attr_expr_boolean_operations() -> None:
    df = pl.DataFrame({"a": [True, False, True], "b": [False, True, True]})

    expr = Attr("a") & Attr("b")  # and
    result = expr.evaluate(df)
    assert result.to_list() == [False, False, True]

    expr = Attr("a") | Attr("b")  # or
    result = expr.evaluate(df)
    assert result.to_list() == [True, True, True]


def test_duplicated_columns() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    expr = (Attr("a") == 1) * 10 - 5 * (Attr("a") > 2)
    result = expr.evaluate(df)
    assert result.to_list() == [10, 0, -5]
    assert expr.columns == ["a"]


def test_attr_reverse_comparison() -> None:
    """Test basic initialization of AttrComparison."""
    attr = Attr("test_column")
    comp = 5 == attr  # reversed on purpose

    assert comp.attr == attr
    assert comp.column == "test_column"
    assert comp.op == operator.eq
    assert comp.other == 5


def test_attr_numpy_comparison() -> None:
    """Test basic initialization of AttrComparison."""
    attr = Attr("test_column")
    comp = attr == np.asarray(5)

    assert comp.attr == attr
    assert comp.column == "test_column"
    assert comp.op == operator.eq
    assert comp.other == 5


def test_attr_comparison_repr() -> None:
    """Test string representation of AttrComparison."""
    attr = Attr("test_column")
    comp = attr > 10

    assert repr(comp) == "Attr(test_column) > 10"


def test_attr_comparison_to_attr() -> None:
    """Test converting AttrComparison back to Attr."""
    df = pl.DataFrame({"test_column": [1, 2, 3, 4, 5]})
    attr = Attr("test_column")
    comp = attr > 3

    converted_attr = comp.to_attr()
    result = converted_attr.evaluate(df)

    assert result.to_list() == [False, False, False, True, True]


def test_attr_comparison_getattr_delegation() -> None:
    """Test that AttrComparison delegates attribute access to its Attr representation."""
    attr = Attr("test_column")
    comp = attr == 5

    # Test that we can access Attr methods through AttrComparison
    assert comp.columns == ["test_column"]
    assert comp.expr_columns == ["test_column"]


def test_attr_comparison_operator_delegation() -> None:
    """Test that AttrComparison delegates operators to its Attr representation."""
    df = pl.DataFrame({"test_column": [1, 2, 3, 4, 5]})
    attr = Attr("test_column")
    comp = attr > 3

    # Test that we can use operators on AttrComparison
    result_attr = comp + 10
    result = result_attr.evaluate(df)

    # Should be (test_column > 3) + 10
    expected = [(x > 3) + 10 for x in df["test_column"]]
    assert result.to_list() == expected


def test_attr_comparison_init_with_infinity_attr() -> None:
    """Test that AttrComparison raises error when attr has infinity."""
    # Create an attr with infinity
    attr_with_inf = Attr("test") * math.inf

    with pytest.raises(ValueError, match="Comparison operators are not supported for expressions with infinity"):
        AttrComparison(attr_with_inf, operator.eq, 5)


def test_attr_comparison_init_with_attr_other() -> None:
    """Test that AttrComparison raises error when comparing two Attr objects."""
    attr1 = Attr("col1")
    attr2 = Attr("col2")

    with pytest.raises(ValueError, match="Does not support comparison between expressions"):
        AttrComparison(attr1, operator.eq, attr2)


def test_attr_comparison_init_with_empty_columns() -> None:
    """Test that AttrComparison raises error for empty expressions."""
    # Create an attr with no columns (literal)
    attr_no_cols = Attr(5)

    with pytest.raises(ValueError, match="Comparison operators are not supported for empty expressions"):
        AttrComparison(attr_no_cols, operator.eq, 10)


def test_attr_comparison_init_with_multiple_columns() -> None:
    """Test that AttrComparison raises error for multiple columns."""
    # Create an attr with multiple columns
    attr_multi_cols = Attr("col1") + Attr("col2")

    with pytest.raises(ValueError, match="Comparison operators are not supported for multiple columns"):
        AttrComparison(attr_multi_cols, operator.eq, 10)


def test_attr_comparison_comparison_operators() -> None:
    """Test all comparison operators with AttrComparison."""
    df = pl.DataFrame({"test_column": [1, 2, 3, 4, 5]})
    attr = Attr("test_column")

    test_cases = [
        (operator.eq, 3, [False, False, True, False, False]),
        (operator.ne, 3, [True, True, False, True, True]),
        (operator.lt, 3, [True, True, False, False, False]),
        (operator.le, 3, [True, True, True, False, False]),
        (operator.gt, 3, [False, False, False, True, True]),
        (operator.ge, 3, [False, False, True, True, True]),
    ]

    for op, other, expected in test_cases:
        comp = op(attr, other)
        result = comp.to_attr().evaluate(df)
        assert result.to_list() == expected


def test_attr_comparison_binary_operators() -> None:
    """Test binary operators with AttrComparison."""
    df = pl.DataFrame({"test_column": [1, 2, 3, 4, 5]})
    attr = Attr("test_column")
    comp = AttrComparison(attr, operator.gt, 3)

    # Test addition
    result = comp + 10
    result_series = result.evaluate(df)
    expected = [(x > 3) + 10 for x in df["test_column"]]
    assert result_series.to_list() == expected

    # Test multiplication
    result = comp * 2
    result_series = result.evaluate(df)
    expected = [(x > 3) * 2 for x in df["test_column"]]
    assert result_series.to_list() == expected


def test_attr_comparison_reverse_operators() -> None:
    """Test reverse operators with AttrComparison."""
    df = pl.DataFrame({"test_column": [1, 2, 3, 4, 5]})
    attr = Attr("test_column")
    comp = AttrComparison(attr, operator.gt, 3)

    # Test reverse addition
    result = 10 + comp
    result_series = result.evaluate(df)
    expected = [10 + (x > 3) for x in df["test_column"]]
    assert result_series.to_list() == expected

    # Test reverse multiplication
    result = 2 * comp
    result_series = result.evaluate(df)
    expected = [2 * (x > 3) for x in df["test_column"]]
    assert result_series.to_list() == expected


def test_split_attr_comps() -> None:
    """Test splitting attribute comparisons into node and edge comparisons."""
    node_attr1 = NodeAttr("node_col1")
    node_attr2 = NodeAttr("node_col2")
    edge_attr1 = EdgeAttr("edge_col1")
    edge_attr2 = EdgeAttr("edge_col2")

    node_comp1 = node_attr1 == 1
    node_comp2 = node_attr2 > 5
    edge_comp1 = edge_attr1 < 10
    edge_comp2 = edge_attr2 != 0

    all_comps = [node_comp1, edge_comp1, node_comp2, edge_comp2]
    node_comps, edge_comps = split_attr_comps(all_comps)

    assert len(node_comps) == 2
    assert len(edge_comps) == 2
    assert node_comps[0].column == "node_col1"
    assert node_comps[1].column == "node_col2"
    assert edge_comps[0].column == "edge_col1"
    assert edge_comps[1].column == "edge_col2"


def test_split_attr_comps_empty() -> None:
    """Test splitting empty list of attribute comparisons."""
    node_comps, edge_comps = split_attr_comps([])
    assert node_comps == []
    assert edge_comps == []


def test_split_attr_comps_only_node() -> None:
    """Test splitting only node attribute comparisons."""
    node_attr = NodeAttr("node_col")
    node_comp = AttrComparison(node_attr, operator.eq, 1)

    node_comps, edge_comps = split_attr_comps([node_comp])
    assert len(node_comps) == 1
    assert len(edge_comps) == 0
    assert node_comps[0].column == "node_col"


def test_split_attr_comps_only_edge() -> None:
    """Test splitting only edge attribute comparisons."""
    edge_attr = EdgeAttr("edge_col")
    edge_comp = AttrComparison(edge_attr, operator.gt, 5)

    node_comps, edge_comps = split_attr_comps([edge_comp])
    assert len(node_comps) == 0
    assert len(edge_comps) == 1
    assert edge_comps[0].column == "edge_col"


def test_split_attr_comps_invalid_type() -> None:
    """Test splitting with invalid attribute type."""
    # Create a comparison with regular Attr instead of NodeAttr or EdgeAttr
    regular_attr = Attr("regular_col")
    regular_comp = regular_attr == 1

    with pytest.raises(ValueError, match="Expected comparisons of 'NodeAttr' or 'EdgeAttr' objects"):
        split_attr_comps([regular_comp])


def test_attr_comps_to_strs() -> None:
    """Test converting attribute comparisons to strings."""
    node_attr = NodeAttr("node_col")
    edge_attr = EdgeAttr("edge_col")

    node_comp = node_attr == 1
    edge_comp = edge_attr > 5

    result = attr_comps_to_strs([node_comp, edge_comp])
    assert result == ["node_col", "edge_col"]


def test_attr_comps_to_strs_empty() -> None:
    """Test converting empty list of attribute comparisons to strings."""
    result = attr_comps_to_strs([])
    assert result == []


def test_polars_reduce_attr_comps() -> None:
    """Test reducing attribute comparisons to a single polars expression."""
    df = pl.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [10, 20, 30, 40, 50], "col3": [True, False, True, False, True]})

    attr1 = Attr("col1")
    attr2 = Attr("col2")
    attr3 = Attr("col3")

    comp1 = attr1 > 2
    comp2 = attr2 < 35
    comp3 = attr3 == True

    result_expr = polars_reduce_attr_comps(df, [comp1, comp2, comp3], operator.and_)
    result = df.select(result_expr).to_series()

    # Expected: (col1 > 2) & (col2 < 35) & (col3 == True)
    expected = [
        False,  # 1 > 2 = False, so False
        False,  # 2 > 2 = False, so False
        True,  # 3 > 2 = True, 30 < 35 = True, True == True = True, so True
        False,  # 4 > 2 = True, 40 < 35 = False, so False
        False,  # 5 > 2 = True, 50 < 35 = False, so False
    ]
    assert result.to_list() == expected


def test_polars_reduce_attr_comps_empty() -> None:
    """Test reducing empty list of attribute comparisons raises ValueError."""
    df = pl.DataFrame({"col1": [1, 2, 3]})

    with pytest.raises(ValueError, match="No attribute comparisons provided"):
        polars_reduce_attr_comps(df, [], operator.and_)


def test_polars_reduce_attr_comps_single() -> None:
    """Test reducing single attribute comparison."""
    df = pl.DataFrame({"col1": [1, 2, 3, 4, 5]})

    attr = Attr("col1")
    comp = attr > 3

    result_expr = polars_reduce_attr_comps(df, [comp], operator.and_)
    result = df.select(result_expr).to_series()

    expected = [False, False, False, True, True]
    assert result.to_list() == expected


def test_attr_comparison_complex_operations() -> None:
    """Test complex operations involving AttrComparison."""
    df = pl.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [10, 20, 30, 40, 50]})

    attr1 = Attr("col1")
    attr2 = Attr("col2")

    comp1 = attr1 > 2
    comp2 = attr2 < 35

    # Test combining comparisons with arithmetic
    result = comp1 * 10 + comp2 * 5
    result_series = result.evaluate(df)

    expected = [(x > 2) * 10 + (y < 35) * 5 for x, y in zip(df["col1"], df["col2"], strict=False)]
    assert result_series.to_list() == expected


def test_attr_comparison_method_delegation() -> None:
    """Test that AttrComparison properly delegates method calls."""
    attr = Attr("test_column")
    comp = attr > 3

    # Test that we can call methods on the comparison
    result = comp.alias("new_name")
    assert isinstance(result, Attr)
    assert result.columns == ["test_column"]


def test_attr_is_in_creates_membership_expression() -> None:
    df = pl.DataFrame({"col": [1, 2, 3, 4]})
    comp = Attr("col").is_in([1, 3, 4])

    assert isinstance(comp, AttrComparison)
    evaluated = comp.to_attr().evaluate(df)
    assert evaluated.to_list() == [True, False, True, True]


def test_attr_is_in_errors() -> None:
    with pytest.raises(
        ValueError, match=r"Cannot use 'is_in' method with non-membership values. Found '1' of type <class 'int'>."
    ):
        Attr("col").is_in(1)


def test_attr_is_in_accepts_numpy_arrays() -> None:
    df = pl.DataFrame({"col": [5, 6, 7]})
    comp = Attr("col").is_in(np.array([6, 8], dtype=np.int64))

    evaluated = comp.to_attr().evaluate(df)
    assert evaluated.to_list() == [False, True, False]

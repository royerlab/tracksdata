"""
Module to compose attribute expressions for attribute filtering or value evaluation.

Attributes are used to query content of nodes and edges through their names as columns in a data frame.

Users will mostly interact with [NodeAttr][tracksdata.attrs.NodeAttr] and [EdgeAttr][tracksdata.attrs.EdgeAttr]
which are thin wrappers around [Attr][tracksdata.attrs.Attr] to distinguish between node and edge attributes
in ambiguous cases.

They can be used to filter elements in the graph as:
```python
graph.filter(NodeAttr("t") == 1).subgraph()
```

Boolean combinations of comparisons can be expressed with `|` (or), `^` (xor),
`&` (and) and `~` (not). Comparisons passed as multiple positional arguments to
`filter()` are still implicitly AND-ed together.
```python
graph.filter((NodeAttr("t") == 1) | (NodeAttr("t") == 2)).subgraph()
graph.filter(~(NodeAttr("t") == 0)).subgraph()
```

Or to create complex expression when solving the tracking problem:
```python
NearestNeighborsSolver(-Attr("iou") * (-Attr("distance") / 30.0).exp())
```
"""

import functools
import math
import operator
from collections.abc import Callable, Sequence
from typing import Any, TypeGuard, Union, overload

import numpy as np
import polars as pl
from polars import DataFrame, Expr, Series

Scalar = int | float | str | bool | complex | np.number
ExprInput = Union[str, Scalar, "Attr", Expr, "AttrComparison"]
MembershipExprInput = Sequence[Scalar]

# Logical operators supported by AttrFilter compounds.
_FILTER_LOGICAL_OPS = ("and", "or", "xor", "not")
FilterInput = Union["AttrComparison", "AttrFilter"]


__all__ = [
    "AttrComparison",
    "AttrFilter",
    "EdgeAttr",
    "NodeAttr",
    "attr_comps_to_strs",
    "polars_reduce_attr_comps",
    "split_attr_comps",
]


def _is_in_op(lhs: Any, values: MembershipExprInput) -> Any:
    """
    Backend-aware membership operator that works for Polars expressions, SQLAlchemy columns, and Python scalars.
    """
    if isinstance(lhs, pl.Expr):
        return lhs.is_in(values)
    if hasattr(lhs, "in_"):
        return lhs.in_(values)
    return lhs in values


_OPS_MATH_SYMBOLS: dict[Callable, str] = {
    operator.add: "+",
    operator.sub: "-",
    operator.mul: "*",
    operator.truediv: "/",
    operator.floordiv: "//",
    operator.mod: "%",
    operator.pow: "**",
    operator.and_: "&",
    operator.or_: "|",
    operator.xor: "^",
    operator.eq: "==",
    operator.ne: "!=",
    operator.lt: "<",
    operator.le: "<=",
    operator.gt: ">",
    operator.ge: ">=",
    _is_in_op: "in",
}


def _is_membership_expr_input(x: Any) -> TypeGuard[MembershipExprInput]:
    if isinstance(x, Attr | AttrComparison | pl.Expr):
        return False
    if isinstance(x, Scalar):
        return False
    if isinstance(x, np.ndarray):
        return getattr(x, "ndim", 1) >= 1
    return isinstance(x, Sequence)


class AttrComparison:
    """
    Class to store a comparison between an [Attr][tracksdata.attrs.Attr] and a value
    (a sequence of values for `is_in`).
    It's mainly used for filtering.
    Complex expression are transformed back to [Attr][tracksdata.attrs.Attr] objects
    which can be used to evaluate the expression on a DataFrame.

    Parameters
    ----------
    attr : Attr
        The attribute to compare.
    op : Callable
        The operator to use for the comparison.
    other : ExprInput | MembershipExprInput
        The value to compare the attribute to.
    """

    def __init__(self, attr: "Attr", op: Callable, other: ExprInput | MembershipExprInput) -> None:
        is_membership_expr = _is_membership_expr_input(other)
        if is_membership_expr and op != _is_in_op:
            raise ValueError(
                f"Membership values can only be used with the 'is_in' method. Found '{_OPS_MATH_SYMBOLS[op]}'."
            )
        elif not is_membership_expr and op == _is_in_op:
            raise ValueError(
                f"Cannot use 'is_in' method with non-membership values. Found '{other}' of type {type(other)}."
            )

        if attr.has_inf():
            raise ValueError("Comparison operators are not supported for expressions with infinity.")

        if isinstance(other, Attr):
            raise ValueError(f"Does not support comparison between expressions. Found {other} and {attr}.")

        columns = attr.expr_columns

        if len(columns) == 0:
            raise ValueError("Comparison operators are not supported for empty expressions.")

        elif len(columns) > 1:
            raise ValueError(f"Comparison operators are not supported for multiple columns. Found {columns}.")

        self.attr = attr
        self.column = columns[0]
        self.op = op

        # casting numpy scalars to python scalars
        # numpy scalars are problematic for sqlalchemy
        if is_membership_expr:
            if isinstance(other, np.ndarray):
                other = other.tolist()
            else:
                other = list(other)
        elif isinstance(other, np.ndarray):
            other = other.item()
        self.other = other

    def __repr__(self) -> str:
        return f"{type(self.attr).__name__}({self.column}) {_OPS_MATH_SYMBOLS[self.op]} {self.other}"

    def to_attr(self) -> "Attr":
        """
        Transform the comparison back to an [Attr][tracksdata.attrs.Attr] object.
        This is useful for evaluating the expression on a DataFrame.
        """
        return Attr(self.op(pl.col(self.column), self.other))

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.to_attr(), attr)

    def _delegate_operator(self, other: ExprInput, op: Callable[[Expr, Expr], Expr], reverse: bool = False) -> "Attr":
        return self.to_attr()._delegate_operator(other, op, reverse)

    # Arithmetic operators (auto-generated by `_setup_ops`, return Attr)
    def __add__(self, other: ExprInput) -> "Attr": ...
    def __sub__(self, other: ExprInput) -> "Attr": ...
    def __mul__(self, other: ExprInput) -> "Attr": ...
    def __truediv__(self, other: ExprInput) -> "Attr": ...
    def __floordiv__(self, other: ExprInput) -> "Attr": ...
    def __mod__(self, other: ExprInput) -> "Attr": ...
    def __pow__(self, other: ExprInput) -> "Attr": ...

    # Reverse arithmetic operators
    def __radd__(self, other: Scalar) -> "Attr": ...
    def __rsub__(self, other: Scalar) -> "Attr": ...
    def __rmul__(self, other: Scalar) -> "Attr": ...
    def __rtruediv__(self, other: Scalar) -> "Attr": ...
    def __rfloordiv__(self, other: Scalar) -> "Attr": ...
    def __rmod__(self, other: Scalar) -> "Attr": ...
    def __rpow__(self, other: Scalar) -> "Attr": ...

    # Logical operators combine comparisons into an AttrFilter compound.
    # `AttrFilter` is defined later in the module; the references below resolve
    # at call time, so the forward reference is fine.
    def _logical_op(self, op_name: str, other: Any, reverse: bool = False) -> "AttrFilter":
        if not isinstance(other, AttrComparison | AttrFilter):
            symbol = _FILTER_OP_SYMBOLS[op_name]
            raise TypeError(
                f"Cannot apply '{symbol}' between an AttrComparison and {type(other).__name__}. "
                "Boolean operators on comparisons combine them into a filter; both operands "
                "must be an AttrComparison or AttrFilter."
            )
        operands = [other, self] if reverse else [self, other]
        return AttrFilter(op_name, operands)

    def __and__(self, other: FilterInput) -> "AttrFilter":
        return self._logical_op("and", other)

    def __rand__(self, other: FilterInput) -> "AttrFilter":
        return self._logical_op("and", other, reverse=True)

    def __or__(self, other: FilterInput) -> "AttrFilter":
        return self._logical_op("or", other)

    def __ror__(self, other: FilterInput) -> "AttrFilter":
        return self._logical_op("or", other, reverse=True)

    def __xor__(self, other: FilterInput) -> "AttrFilter":
        return self._logical_op("xor", other)

    def __rxor__(self, other: FilterInput) -> "AttrFilter":
        return self._logical_op("xor", other, reverse=True)

    def __invert__(self) -> "AttrFilter":
        return AttrFilter("not", [self])

    # Comparison operators (always return Attr)
    def __eq__(self, other: ExprInput) -> "Attr": ...
    def __req__(self, other: ExprInput) -> "Attr": ...
    def __ne__(self, other: ExprInput) -> "Attr": ...
    def __rne__(self, other: ExprInput) -> "Attr": ...
    def __lt__(self, other: ExprInput) -> "Attr": ...
    def __rlt__(self, other: ExprInput) -> "Attr": ...
    def __le__(self, other: ExprInput) -> "Attr": ...
    def __rle__(self, other: ExprInput) -> "Attr": ...
    def __gt__(self, other: ExprInput) -> "Attr": ...
    def __rgt__(self, other: ExprInput) -> "Attr": ...
    def __ge__(self, other: ExprInput) -> "Attr": ...
    def __rge__(self, other: ExprInput) -> "Attr": ...


class Attr:
    """
    A class to compose an attribute expression for attribute filtering or value evaluation.

    Parameters
    ----------
    value : ExprInput
        The value to compose the attribute expression from.

    Examples
    --------
    ```python
    Attr("t") == 1  # filter for time point 1
    Attr("iou").log()  # log the iou
    Attr(1.0)  # constant value
    Attr((1 - Attr("iou")) * Attr("distance"))  # complex expression
    ```
    """

    expr: Expr

    def __init__(self, value: ExprInput) -> None:
        self._inf_exprs = []  # expressions multiplied by +inf
        self._neg_inf_exprs = []  # expressions multiplied by -inf

        if isinstance(value, str):
            self.expr = pl.col(value)
        elif isinstance(value, Attr):
            self.expr = value.expr
            # Copy infinity tracking from the other AttrExpr
            self._inf_exprs = value.inf_exprs
            self._neg_inf_exprs = value.neg_inf_exprs
        elif isinstance(value, AttrComparison):
            attr = value.to_attr()
            self.expr = attr.expr
            self._inf_exprs = attr.inf_exprs
            self._neg_inf_exprs = attr.neg_inf_exprs
        elif isinstance(value, Expr):
            self.expr = value
        else:
            self.expr = pl.lit(value)

    def _wrap(self, expr: ExprInput) -> Union["Attr", Any]:
        if isinstance(expr, Expr):
            result = Attr(expr)
            # Propagate infinity tracking
            result._inf_exprs = self._inf_exprs.copy()
            result._neg_inf_exprs = self._neg_inf_exprs.copy()
            return result
        return expr

    def _delegate_operator(self, other: ExprInput, op: Callable[[Expr, Expr], Expr], reverse: bool = False) -> "Attr":
        """
        Delegate the operator to the expression.

        Parameters
        ----------
        other : ExprInput
            The other expression to delegate the operator to.
        op : Callable[[Expr, Expr], Expr]
            The operator to delegate.
        reverse : bool, optional
            Whether the operator is reversed.

        Returns
        -------
        Attr
            The result of the operator.
        """
        # Special handling for multiplication with infinity
        if op == operator.mul:
            # Check if we're multiplying with infinity scalar
            # In both reverse and non-reverse cases, 'other' is the infinity value
            # and 'self' is the AttrExpr we want to track
            if isinstance(other, int | float) and math.isinf(other):
                result = Attr(pl.lit(0))  # Clean expression is zero (infinity term removed)

                # Copy existing infinity tracking
                result._inf_exprs = self._inf_exprs.copy()
                result._neg_inf_exprs = self._neg_inf_exprs.copy()

                # Add the expression to appropriate infinity list
                if other > 0:
                    result._inf_exprs.append(self)
                else:
                    result._neg_inf_exprs.append(self)

                return result

        # Regular operation - no infinity involved
        left = Attr(other).expr if reverse else self.expr
        right = self.expr if reverse else Attr(other).expr
        result = Attr(op(left, right))

        # Combine infinity tracking from both operands
        if isinstance(other, Attr):
            result._inf_exprs = self._inf_exprs + other._inf_exprs
            result._neg_inf_exprs = self._neg_inf_exprs + other._neg_inf_exprs

            # Special handling for subtraction: flip signs of the second operand's infinity terms
            if op == operator.sub and not reverse:
                # self - other: other's positive infinity becomes negative, negative becomes positive
                result._inf_exprs = self._inf_exprs + other._neg_inf_exprs
                result._neg_inf_exprs = self._neg_inf_exprs + other._inf_exprs
            elif op == operator.sub and reverse:
                # other - self: self's positive infinity becomes negative, negative becomes positive
                result._inf_exprs = other._inf_exprs + self._neg_inf_exprs
                result._neg_inf_exprs = other._neg_inf_exprs + self._inf_exprs
        else:
            result._inf_exprs = self._inf_exprs.copy()
            result._neg_inf_exprs = self._neg_inf_exprs.copy()

        return result

    def _delegate_comparison_operator(
        self,
        other: ExprInput,
        op: Callable,
        reverse: bool = False,
    ) -> "AttrComparison | Attr":
        """
        Simplified version of `_delegate_operator` for comparison operators.
        [AttrComparison][tracksdata.attrs.AttrComparison] has a limited scope and
        it's mainly used for filtering.
        If creating an [AttrComparison][tracksdata.attrs.AttrComparison] object is
        not possible, it will return an [Attr][tracksdata.attrs.Attr] object.

        Parameters
        ----------
        other : ExprInput
            The other expression to delegate the operator to.
        op : Callable
            The operator to delegate.
        reverse : bool, optional
            Whether the operator is reversed.

        Returns
        -------
        AttrComparison | Attr
            The result of the operator.
        """
        if reverse:
            lhs = Attr(other)
            rhs = self
        else:
            lhs = self
            rhs = other

        if isinstance(other, Attr):
            return self._delegate_operator(other, op, reverse=False)

        return AttrComparison(lhs, op, rhs)

    def alias(self, name: str) -> "Attr":
        result = Attr(self.expr.alias(name))
        result._inf_exprs = self._inf_exprs.copy()
        result._neg_inf_exprs = self._neg_inf_exprs.copy()
        return result

    def evaluate(self, df: DataFrame) -> Series:
        """
        Evaluate the expression on a DataFrame returning a numeric result.

        Parameters
        ----------
        df : DataFrame
            The DataFrame to evaluate the expression on.

        Returns
        -------
        Series
            The evaluated expression.
        """
        return df.select(self.expr).to_series()

    @property
    def columns(self) -> list[str]:
        return list(dict.fromkeys(self.expr_columns + self.inf_columns + self.neg_inf_columns))

    @property
    def inf_exprs(self) -> list["Attr"]:
        """Get the expressions multiplied by positive infinity."""
        return self._inf_exprs.copy()

    @property
    def neg_inf_exprs(self) -> list["Attr"]:
        """Get the expressions multiplied by negative infinity."""
        return self._neg_inf_exprs.copy()

    @property
    def expr_columns(self) -> list[str]:
        """Get the names of columns in the expression."""
        return list(dict.fromkeys(self.expr.meta.root_names()))

    @property
    def inf_columns(self) -> list[str]:
        """Get the names of columns multiplied by positive infinity."""
        columns = []
        for attr_expr in self._inf_exprs:
            columns.extend(attr_expr.columns)
        return list(dict.fromkeys(columns))

    @property
    def neg_inf_columns(self) -> list[str]:
        """Get the names of columns multiplied by negative infinity."""
        columns = []
        for attr_expr in self._neg_inf_exprs:
            columns.extend(attr_expr.columns)
        return list(dict.fromkeys(columns))

    def has_inf(self) -> bool:
        """
        Check if any column in the expression is multiplied by infinity or negative infinity.

        Returns
        -------
        bool
            True if any column is multiplied by infinity, False otherwise.
        """
        return self.has_pos_inf() or self.has_neg_inf()

    def has_pos_inf(self) -> bool:
        """
        Check if any column in the expression is multiplied by positive infinity.
        """
        return len(self._inf_exprs) > 0

    def has_neg_inf(self) -> bool:
        """
        Check if any column in the expression is multiplied by negative infinity.
        """
        return len(self._neg_inf_exprs) > 0

    def is_in(self, values: MembershipExprInput) -> "AttrComparison":
        """
        Create a membership comparison between the attribute and a collection of literals.

        Parameters
        ----------
        values : Iterable[Scalar] | Sequence[Scalar] | np.ndarray | Series
            Values the attribute should belong to.

        Returns
        -------
        AttrComparison
            A comparison suitable for filtering across all graph backends.
        """
        return AttrComparison(self, _is_in_op, values)

    def __invert__(self) -> "Attr":
        return Attr(~self.expr)

    def __neg__(self) -> "Attr":
        return Attr(-self.expr)

    def __pos__(self) -> "Attr":
        return Attr(+self.expr)

    def __abs__(self) -> "Attr":
        return Attr(abs(self.expr))

    def __getattr__(self, attr: str) -> Any:
        # Don't delegate our internal attributes to the expr
        if attr.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

        # To auto generate operator methods such as `.log()``
        expr_attr = getattr(self.expr, attr)
        if callable(expr_attr):

            @functools.wraps(expr_attr)
            def _wrapped(*args, **kwargs):
                return self._wrap(expr_attr(*args, **kwargs))

            return _wrapped
        return expr_attr

    def __repr__(self) -> str:
        return f"Attr({self.expr})"

    # Binary operators
    def __add__(self, other: ExprInput) -> "Attr": ...
    def __sub__(self, other: ExprInput) -> "Attr": ...
    def __mul__(self, other: ExprInput) -> "Attr": ...
    def __truediv__(self, other: ExprInput) -> "Attr": ...
    def __floordiv__(self, other: ExprInput) -> "Attr": ...
    def __mod__(self, other: ExprInput) -> "Attr": ...
    def __pow__(self, other: ExprInput) -> "Attr": ...
    def __and__(self, other: ExprInput) -> "Attr": ...
    def __or__(self, other: ExprInput) -> "Attr": ...
    def __xor__(self, other: ExprInput) -> "Attr": ...

    # Reverse operators
    def __radd__(self, other: Scalar) -> "Attr": ...
    def __rsub__(self, other: Scalar) -> "Attr": ...
    def __rmul__(self, other: Scalar) -> "Attr": ...
    def __rtruediv__(self, other: Scalar) -> "Attr": ...
    def __rfloordiv__(self, other: Scalar) -> "Attr": ...
    def __rmod__(self, other: Scalar) -> "Attr": ...
    def __rpow__(self, other: Scalar) -> "Attr": ...
    def __rand__(self, other: Scalar) -> "Attr": ...
    def __ror__(self, other: Scalar) -> "Attr": ...
    def __rxor__(self, other: Scalar) -> "Attr": ...

    # Comparison operators with overloads
    @overload
    def __eq__(self, other: "Attr") -> "Attr": ...
    @overload
    def __eq__(self, other: Scalar) -> "AttrComparison": ...
    def __eq__(self, other: ExprInput) -> "Attr | AttrComparison": ...

    @overload
    def __req__(self, other: "Attr") -> "Attr": ...
    @overload
    def __req__(self, other: Scalar) -> "AttrComparison": ...
    def __req__(self, other: ExprInput) -> "Attr | AttrComparison": ...

    @overload
    def __ne__(self, other: "Attr") -> "Attr": ...
    @overload
    def __ne__(self, other: Scalar) -> "AttrComparison": ...
    def __ne__(self, other: ExprInput) -> "Attr | AttrComparison": ...

    @overload
    def __rne__(self, other: "Attr") -> "Attr": ...
    @overload
    def __rne__(self, other: Scalar) -> "AttrComparison": ...
    def __rne__(self, other: ExprInput) -> "Attr | AttrComparison": ...

    @overload
    def __lt__(self, other: "Attr") -> "Attr": ...
    @overload
    def __lt__(self, other: Scalar) -> "AttrComparison": ...
    def __lt__(self, other: ExprInput) -> "Attr | AttrComparison": ...

    @overload
    def __rlt__(self, other: "Attr") -> "Attr": ...
    @overload
    def __rlt__(self, other: Scalar) -> "AttrComparison": ...
    def __rlt__(self, other: ExprInput) -> "Attr | AttrComparison": ...

    @overload
    def __le__(self, other: "Attr") -> "Attr": ...
    @overload
    def __le__(self, other: Scalar) -> "AttrComparison": ...
    def __le__(self, other: ExprInput) -> "Attr | AttrComparison": ...

    @overload
    def __rle__(self, other: "Attr") -> "Attr": ...
    @overload
    def __rle__(self, other: Scalar) -> "AttrComparison": ...
    def __rle__(self, other: ExprInput) -> "Attr | AttrComparison": ...

    @overload
    def __gt__(self, other: "Attr") -> "Attr": ...
    @overload
    def __gt__(self, other: Scalar) -> "AttrComparison": ...
    def __gt__(self, other: ExprInput) -> "Attr | AttrComparison": ...

    @overload
    def __rgt__(self, other: "Attr") -> "Attr": ...
    @overload
    def __rgt__(self, other: Scalar) -> "AttrComparison": ...
    def __rgt__(self, other: ExprInput) -> "Attr | AttrComparison": ...

    @overload
    def __ge__(self, other: "Attr") -> "Attr": ...
    @overload
    def __ge__(self, other: Scalar) -> "AttrComparison": ...
    def __ge__(self, other: ExprInput) -> "Attr | AttrComparison": ...

    @overload
    def __rge__(self, other: "Attr") -> "Attr": ...
    @overload
    def __rge__(self, other: Scalar) -> "AttrComparison": ...
    def __rge__(self, other: ExprInput) -> "Attr | AttrComparison": ...


# Auto-generate operator methods using functools.partialmethod
def _add_operator(
    cls: type[Attr] | type[AttrComparison],
    name: str,
    op: Callable,
    reverse: bool = False,
) -> None:
    method = functools.partialmethod(cls._delegate_operator, op=op, reverse=reverse)
    setattr(cls, name, method)


def _add_comparison_operator(
    name: str,
    op: Callable,
    reverse: bool = False,
) -> None:
    method = functools.partialmethod(Attr._delegate_comparison_operator, op=op, reverse=reverse)
    setattr(Attr, name, method)


def _setup_ops() -> None:
    """
    Setup the operator methods for the AttrExpr class.
    """
    # Arithmetic operators: generated for both Attr and AttrComparison.
    bin_ops = {
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "truediv": operator.truediv,
        "floordiv": operator.floordiv,
        "mod": operator.mod,
        "pow": operator.pow,
    }

    # Logical operators: generated only for Attr (bitwise on the polars expr).
    # AttrComparison defines its own `& | ^ ~` in the class body to build
    # AttrFilter compounds, so they are intentionally excluded here.
    logical_ops = {
        "and": operator.and_,
        "or": operator.or_,
        "xor": operator.xor,
    }

    comp_ops = {
        "eq": operator.eq,
        "ne": operator.ne,
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
    }

    for op_name, op_func in (bin_ops | logical_ops).items():
        _add_operator(Attr, f"__{op_name}__", op_func, reverse=False)
        _add_operator(Attr, f"__r{op_name}__", op_func, reverse=True)

    for op_name, op_func in bin_ops.items():
        _add_operator(AttrComparison, f"__{op_name}__", op_func, reverse=False)
        _add_operator(AttrComparison, f"__r{op_name}__", op_func, reverse=True)

    for op_name, op_func in comp_ops.items():
        _add_comparison_operator(f"__{op_name}__", op_func, reverse=False)
        _add_comparison_operator(f"__r{op_name}__", op_func, reverse=True)

        # attrr_comparision uses normal delegate_operator
        _add_operator(AttrComparison, f"__{op_name}__", op_func, reverse=False)
        _add_operator(AttrComparison, f"__r{op_name}__", op_func, reverse=True)


_setup_ops()


class NodeAttr(Attr):
    """
    Wrapper of [Attr][tracksdata.attrs.Attr] to represent a node attribute.

    See Also
    --------
    [Attr][tracksdata.attrs.Attr]:
        The base class for all attributes.
    """


class EdgeAttr(Attr):
    """
    Wrapper of [Attr][tracksdata.attrs.Attr] to represent an edge attribute.

    See Also
    --------
    [Attr][tracksdata.attrs.Attr]:
        The base class for all attributes.
    """


_FILTER_OP_SYMBOLS = {"and": "&", "or": "|", "xor": "^", "not": "~"}


class AttrFilter:
    """
    A compound boolean combination of [AttrComparison][tracksdata.attrs.AttrComparison]
    (or nested `AttrFilter`) operands, used to express OR / XOR / AND / NOT
    relationships when filtering nodes or edges in a graph.

    Use Python's bitwise operators on `AttrComparison` (or `AttrFilter`)
    instances to build compounds:

    ```python
    graph.filter((NodeAttr("t") == 1) | (NodeAttr("t") == 2))
    graph.filter(~(NodeAttr("t") == 0))
    graph.filter((EdgeAttr("w") > 0.5) ^ (EdgeAttr("w") < -0.5))
    ```

    All leaves of a single `AttrFilter` must reference attributes of the same
    kind (either all [NodeAttr][tracksdata.attrs.NodeAttr] or all
    [EdgeAttr][tracksdata.attrs.EdgeAttr]). Mixing node and edge attributes
    inside one compound is not supported because it would require joining the
    node and edge tables in a way that conflicts with the existing AND-based
    filter semantics. Top-level node/edge filters can still be combined via
    positional arguments to `graph.filter()` (implicit AND).

    Parameters
    ----------
    op : str
        Logical operator, one of `"and"`, `"or"`, `"xor"`, `"not"`.
    operands : Sequence[AttrComparison | AttrFilter]
        Operands. `"not"` requires exactly one operand; the others require at
        least two.
    """

    def __init__(self, op: str, operands: Sequence[FilterInput]) -> None:
        if op not in _FILTER_LOGICAL_OPS:
            raise ValueError(f"Unknown logical operator '{op}'. Expected one of {_FILTER_LOGICAL_OPS}.")
        operands = list(operands)
        for o in operands:
            if not isinstance(o, AttrComparison | AttrFilter):
                raise TypeError(f"AttrFilter operands must be AttrComparison or AttrFilter, got {type(o).__name__}.")
        if op == "not":
            if len(operands) != 1:
                raise ValueError("'not' filter requires exactly one operand.")
        else:
            if len(operands) < 2:
                raise ValueError(f"'{op}' filter requires at least two operands.")
        self.op = op
        self.operands = operands

    def __and__(self, other: FilterInput) -> "AttrFilter":
        return AttrFilter("and", [self, other])

    def __rand__(self, other: FilterInput) -> "AttrFilter":
        return AttrFilter("and", [other, self])

    def __or__(self, other: FilterInput) -> "AttrFilter":
        return AttrFilter("or", [self, other])

    def __ror__(self, other: FilterInput) -> "AttrFilter":
        return AttrFilter("or", [other, self])

    def __xor__(self, other: FilterInput) -> "AttrFilter":
        return AttrFilter("xor", [self, other])

    def __rxor__(self, other: FilterInput) -> "AttrFilter":
        return AttrFilter("xor", [other, self])

    def __invert__(self) -> "AttrFilter":
        return AttrFilter("not", [self])

    def to_attr(self) -> "Attr":
        """Translate the compound filter to an `Attr` holding the polars boolean expression.

        Mirrors [AttrComparison.to_attr][tracksdata.attrs.AttrComparison.to_attr]
        and folds children polymorphically — both operand types expose `to_attr`,
        so no parallel `(AttrComparison | AttrFilter)` walker is needed for
        evaluation or column extraction.
        """
        if self.op == "not":
            return Attr(~self.operands[0].to_attr().expr)
        child_exprs = [o.to_attr().expr for o in self.operands]
        if self.op == "and":
            return Attr(functools.reduce(operator.and_, child_exprs))
        if self.op == "or":
            return Attr(functools.reduce(operator.or_, child_exprs))
        # xor
        return Attr(functools.reduce(operator.xor, child_exprs))

    def leaves(self) -> list["AttrComparison"]:
        """Flatten the filter tree to its leaf comparisons."""
        out: list[AttrComparison] = []
        for o in self.operands:
            if isinstance(o, AttrFilter):
                out.extend(o.leaves())
            else:
                out.append(o)
        return out

    @property
    def columns(self) -> list[str]:
        return self.to_attr().expr_columns

    def __repr__(self) -> str:
        if self.op == "not":
            return f"~{self.operands[0]!r}"
        sep = f" {_FILTER_OP_SYMBOLS[self.op]} "
        return "(" + sep.join(repr(o) for o in self.operands) + ")"


def _filter_attr_kind(f: FilterInput) -> type[Attr]:
    """Return the leaf-attribute kind (NodeAttr / EdgeAttr) of a filter.

    Raises ValueError if the filter mixes node and edge attributes.
    """
    if isinstance(f, AttrComparison):
        if isinstance(f.attr, NodeAttr):
            return NodeAttr
        if isinstance(f.attr, EdgeAttr):
            return EdgeAttr
        raise ValueError(f"Expected comparisons of 'NodeAttr' or 'EdgeAttr' objects, got {type(f.attr)}")

    kinds = {_filter_attr_kind(o) for o in f.operands}
    if len(kinds) > 1:
        raise ValueError(
            "A single AttrFilter compound cannot mix NodeAttr and EdgeAttr comparisons. "
            "Combine node and edge filters via separate positional arguments to graph.filter()."
        )
    return kinds.pop()


def split_attr_comps(
    attr_comps: Sequence[FilterInput],
) -> tuple[list[FilterInput], list[FilterInput]]:
    """
    Split a list of attribute comparisons (or compound filters) into node and
    edge groups based on the kind of their leaf comparisons.

    Parameters
    ----------
    attr_comps : Sequence[AttrComparison | AttrFilter]
        The attribute comparisons or compound filters to split.

    Returns
    -------
    tuple[list[AttrComparison | AttrFilter], list[AttrComparison | AttrFilter]]
        A tuple of lists of node and edge filters.
    """
    node_attr_comps: list[FilterInput] = []
    edge_attr_comps: list[FilterInput] = []

    for attr_comp in attr_comps:
        kind = _filter_attr_kind(attr_comp)
        if kind is NodeAttr:
            node_attr_comps.append(attr_comp)
        else:
            edge_attr_comps.append(attr_comp)

    return node_attr_comps, edge_attr_comps


def attr_comps_to_strs(attr_comps: Sequence[FilterInput]) -> list[str]:
    """
    Convert a list of attribute comparisons (or compound filters) to a list of
    column names involved in them.

    Parameters
    ----------
    attr_comps : Sequence[AttrComparison | AttrFilter]
        The filters to extract column names from.

    Returns
    -------
    list[str]
        The column names referenced by the filters, deduplicated while
        preserving order.
    """
    # Both `AttrComparison` (via `__getattr__` → `to_attr().columns`) and
    # `AttrFilter` (via its `columns` property) expose `.columns`.
    return list(dict.fromkeys(c for ac in attr_comps for c in ac.columns))


def polars_reduce_attr_comps(
    df: pl.DataFrame,
    attr_comps: Sequence[FilterInput],
    reduce_op: Callable[[Expr, Expr], Expr],
) -> pl.Expr:
    """
    Reduce a list of attribute comparisons (or compound filters) into a single
    polars expression, combined with `reduce_op` at the top level (AND-ed by
    default in callers).

    Parameters
    ----------
    df : pl.DataFrame
        Unused; kept for backward-compatible signature. Each filter already
        produces a fully-formed polars expression via `to_attr`.
    attr_comps : Sequence[AttrComparison | AttrFilter]
        The filters to reduce.
    reduce_op : Callable[[Expr, Expr], Expr]
        The operation to reduce the top-level filters with.

    Returns
    -------
    pl.Expr
        The reduced polars expression.
    """
    if not attr_comps:
        raise ValueError("No attribute comparisons provided.")
    del df  # unused; preserved in the signature to avoid a breaking change
    return pl.reduce(reduce_op, [f.to_attr().expr for f in attr_comps])

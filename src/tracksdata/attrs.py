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
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, TypeGuard, Union, overload

import numpy as np
import polars as pl
from polars import DataFrame, Expr, Series

Scalar = int | float | str | bool | complex | np.number
ExprInput = Union[str, Scalar, "Attr", Expr, "AttrComparison"]
MembershipExprInput = Sequence[Scalar] | np.ndarray

# Logical operators supported by AttrFilter compounds (op name → bitwise symbol).
_FILTER_OP_SYMBOLS = {"and": "&", "or": "|", "xor": "^", "not": "~"}


__all__ = [
    "AttrComparison",
    "AttrFilter",
    "EdgeAttr",
    "Filter",
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


class Filter(ABC):
    """
    Common interface for anything usable as a filter argument to `graph.filter()`.

    Implemented by [AttrComparison][tracksdata.attrs.AttrComparison] (leaf
    comparisons like `NodeAttr("t") == 1`) and
    [AttrFilter][tracksdata.attrs.AttrFilter] (compound boolean trees built
    with `& | ^ ~`). Helpers in this module accept any `Filter` and treat
    them interchangeably via `to_attr()` and `columns`.

    Backend code that needs to walk a compound tree (e.g. SQL pushdown in
    `_to_sql_clause`, Python evaluation in `_eval_filter`) still uses
    `isinstance(_, AttrComparison)` to distinguish leaves from compounds —
    that structural dispatch is intentional and not part of this interface.
    """

    @abstractmethod
    def to_attr(self) -> "Attr":
        """Materialize this filter as an `Attr` holding a polars boolean expression."""

    @property
    @abstractmethod
    def columns(self) -> list[str]:
        """Column names referenced by this filter, in order, deduplicated."""

    def _combine(self, op: str, other: Any, reverse: bool = False) -> "AttrFilter":
        if not isinstance(other, Filter):
            symbol = _FILTER_OP_SYMBOLS[op]
            raise TypeError(
                f"Cannot apply '{symbol}' between {type(self).__name__} and {type(other).__name__}. "
                "Boolean operators on filters combine them into a compound filter; both operands "
                "must be a Filter (AttrComparison or AttrFilter)."
            )
        operands = [other, self] if reverse else [self, other]
        return AttrFilter(op, operands)

    def __and__(self, other: "Filter") -> "AttrFilter":
        return self._combine("and", other)

    def __rand__(self, other: "Filter") -> "AttrFilter":
        return self._combine("and", other, reverse=True)

    def __or__(self, other: "Filter") -> "AttrFilter":
        return self._combine("or", other)

    def __ror__(self, other: "Filter") -> "AttrFilter":
        return self._combine("or", other, reverse=True)

    def __xor__(self, other: "Filter") -> "AttrFilter":
        return self._combine("xor", other)

    def __rxor__(self, other: "Filter") -> "AttrFilter":
        return self._combine("xor", other, reverse=True)

    def __invert__(self) -> "AttrFilter":
        return AttrFilter("not", [self])


class AttrComparison(Filter):
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
        # Prefer the explicitly tracked root_column so struct-field comparisons
        # (e.g. `NodeAttr("m").struct.field("x") == 1`) record the parent storage
        # column ("m"), letting backends remap to their physical layout via field_path.
        self.column = attr.root_column if attr.root_column is not None else columns[0]
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
        if self.attr.field_path:
            column = ".".join([str(self.column), *self.attr.field_path])
        else:
            column = str(self.column)
        return f"{type(self.attr).__name__}({column}) {_OPS_MATH_SYMBOLS[self.op]} {self.other}"

    def to_attr(self) -> "Attr":
        """
        Transform the comparison back to an [Attr][tracksdata.attrs.Attr] object.
        This is useful for evaluating the expression on a DataFrame.
        """
        return Attr(self.op(self.attr.expr, self.other))

    @property
    def columns(self) -> list[str]:
        return [self.column]

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

    # Boolean operators (`& | ^ ~`) are inherited from `Filter` and combine
    # comparisons into an `AttrFilter` compound.

    # Comparison operators (always return Attr).
    # No `__r{op}__` stubs: Python's data model uses `__eq__`/`__ne__` for
    # reflected `==`/`!=` and the OPPOSITE op for reflected `<>≤≥`
    # (e.g. `1 < attr` → `attr.__gt__(1)`), so `__rlt__` etc. are never called.
    def __eq__(self, other: ExprInput) -> "Attr": ...
    def __ne__(self, other: ExprInput) -> "Attr": ...
    def __lt__(self, other: ExprInput) -> "Attr": ...
    def __le__(self, other: ExprInput) -> "Attr": ...
    def __gt__(self, other: ExprInput) -> "Attr": ...
    def __ge__(self, other: ExprInput) -> "Attr": ...


class _StructNamespace:
    """Wrapper around polars struct namespace that preserves Attr semantics.

    Polars' own ``Expr.struct.field(name)`` only updates the underlying expression;
    it loses the parent column identity, which backends need to map a filter back
    to its physical storage (e.g. SQL flat columns, dict lookups in rustworkx).
    This wrapper proxies the namespace while threading ``root_column`` and
    ``field_path`` through ``.field(...)`` calls.
    """

    def __init__(self, attr: "Attr") -> None:
        self._attr = attr
        self._namespace = attr.expr.struct

    def field(self, name: str) -> "Attr":
        # preserve_field_path keeps the existing root/path before appending the new field.
        out = self._attr._wrap(self._namespace.field(name), preserve_field_path=True)
        # _namespace.field() always returns a polars Expr, so _wrap always yields an Attr here.
        out._append_field_path(name)
        return out

    def __getattr__(self, name: str) -> Any:
        namespace_attr = getattr(self._namespace, name)
        if callable(namespace_attr):

            @functools.wraps(namespace_attr)
            def _wrapped(*args, **kwargs):
                return self._attr._wrap(namespace_attr(*args, **kwargs))

            return _wrapped
        return namespace_attr


class _StructNamespace:
    """Wrapper around polars struct namespace that preserves Attr semantics.

    Polars' own ``Expr.struct.field(name)`` only updates the underlying expression;
    it loses the parent column identity, which backends need to map a filter back
    to its physical storage (e.g. SQL flat columns, dict lookups in rustworkx).
    This wrapper proxies the namespace while threading ``root_column`` and
    ``field_path`` through ``.field(...)`` calls.
    """

    def __init__(self, attr: "Attr") -> None:
        self._attr = attr
        self._namespace = attr.expr.struct

    def field(self, name: str) -> "Attr":
        # preserve_field_path keeps the existing root/path before appending the new field.
        out = self._attr._wrap(self._namespace.field(name), preserve_field_path=True)
        # _namespace.field() always returns a polars Expr, so _wrap always yields an Attr here.
        out._append_field_path(name)
        return out

    def __getattr__(self, name: str) -> Any:
        namespace_attr = getattr(self._namespace, name)
        if callable(namespace_attr):

            @functools.wraps(namespace_attr)
            def _wrapped(*args, **kwargs):
                return self._attr._wrap(namespace_attr(*args, **kwargs))

            return _wrapped
        return namespace_attr


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
        # Path-tracking for backend filters:
        # - root_column: top-level column used to store the value.
        # - field_path: nested struct path from that root column.
        self._root_column: str | None = None
        self._field_path: tuple[str, ...] = ()

        if isinstance(value, str):
            self.expr = pl.col(value)
            self._root_column = value
        elif isinstance(value, Attr):
            self.expr = value.expr
            # Copy infinity tracking from the other AttrExpr
            self._inf_exprs = value.inf_exprs
            self._neg_inf_exprs = value.neg_inf_exprs
            self._root_column = value.root_column
            self._field_path = value.field_path
        elif isinstance(value, AttrComparison):
            attr = value.to_attr()
            self.expr = attr.expr
            self._inf_exprs = attr.inf_exprs
            self._neg_inf_exprs = attr.neg_inf_exprs
            self._root_column = attr.root_column
            self._field_path = attr.field_path
        elif isinstance(value, Expr):
            self.expr = value
        else:
            self.expr = pl.lit(value)

    def _wrap(self, expr: ExprInput, *, preserve_field_path: bool = False) -> Union["Attr", Any]:
        if isinstance(expr, Expr):
            result = type(self)(expr)
            # Propagate infinity tracking
            result._inf_exprs = self._inf_exprs.copy()
            result._neg_inf_exprs = self._neg_inf_exprs.copy()
            if preserve_field_path:
                result._root_column = self._root_column
                result._field_path = self._field_path
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

        Returns
        -------
        AttrComparison | Attr
            The result of the operator.
        """
        if isinstance(other, Attr):
            return self._delegate_operator(other, op, reverse=False)

        return AttrComparison(self, op, other)

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
    def root_column(self) -> str | None:
        """
        Top-level column name from which this expression originates.

        Examples
        --------
        `Attr("t").root_column == "t"`
        `NodeAttr("measurements").struct.field("score").root_column == "measurements"`
        """
        return self._root_column

    @property
    def field_path(self) -> tuple[str, ...]:
        """
        Nested struct-field path relative to [root_column][tracksdata.attrs.Attr.root_column].

        Empty tuple means no nested access.

        Examples
        --------
        `Attr("t").field_path == ()`
        `NodeAttr("measurements").struct.field("score").field_path == ("score",)`
        `NodeAttr("meta").struct.field("det").struct.field("conf").field_path == ("det", "conf")`
        """
        return self._field_path

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

        if attr == "struct":
            return _StructNamespace(self)

        # To auto generate operator methods such as `.log()``
        expr_attr = getattr(self.expr, attr)
        if callable(expr_attr):

            @functools.wraps(expr_attr)
            def _wrapped(*args, **kwargs):
                return self._wrap(expr_attr(*args, **kwargs))

            return _wrapped
        return expr_attr

    def _append_field_path(self, field_name: str) -> None:
        if self._root_column is None:
            self._field_path = ()
        else:
            self._field_path = (*self._field_path, field_name)

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
    def __ne__(self, other: "Attr") -> "Attr": ...
    @overload
    def __ne__(self, other: Scalar) -> "AttrComparison": ...
    def __ne__(self, other: ExprInput) -> "Attr | AttrComparison": ...

    @overload
    def __lt__(self, other: "Attr") -> "Attr": ...
    @overload
    def __lt__(self, other: Scalar) -> "AttrComparison": ...
    def __lt__(self, other: ExprInput) -> "Attr | AttrComparison": ...

    @overload
    def __le__(self, other: "Attr") -> "Attr": ...
    @overload
    def __le__(self, other: Scalar) -> "AttrComparison": ...
    def __le__(self, other: ExprInput) -> "Attr | AttrComparison": ...

    @overload
    def __gt__(self, other: "Attr") -> "Attr": ...
    @overload
    def __gt__(self, other: Scalar) -> "AttrComparison": ...
    def __gt__(self, other: ExprInput) -> "Attr | AttrComparison": ...

    @overload
    def __ge__(self, other: "Attr") -> "Attr": ...
    @overload
    def __ge__(self, other: Scalar) -> "AttrComparison": ...
    def __ge__(self, other: ExprInput) -> "Attr | AttrComparison": ...


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
) -> None:
    method = functools.partialmethod(Attr._delegate_comparison_operator, op=op)
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
    # AttrComparison inherits `& | ^ ~` from `Filter` (they build AttrFilter
    # compounds), so they are intentionally excluded here.
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

    # No reverse comparison operators (`__rlt__` etc.) — Python uses the
    # opposite op for reflected `<>≤≥` and `__eq__`/`__ne__` symmetrically.
    for op_name, op_func in comp_ops.items():
        _add_comparison_operator(f"__{op_name}__", op_func)
        # AttrComparison uses normal delegate_operator
        _add_operator(AttrComparison, f"__{op_name}__", op_func, reverse=False)


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


class AttrFilter(Filter):
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
    operands : Sequence[Filter]
        Operands. `"not"` requires exactly one operand; the others require at
        least two.
    """

    def __init__(self, op: str, operands: Sequence[Filter]) -> None:
        if op not in _FILTER_OP_SYMBOLS:
            raise ValueError(f"Unknown logical operator '{op}'. Expected one of {tuple(_FILTER_OP_SYMBOLS)}.")
        operands = list(operands)
        for o in operands:
            if not isinstance(o, Filter):
                raise TypeError(
                    f"AttrFilter operands must be Filter (AttrComparison or AttrFilter), got {type(o).__name__}."
                )
        if op == "not":
            if len(operands) != 1:
                raise ValueError("'not' filter requires exactly one operand.")
        else:
            if len(operands) < 2:
                raise ValueError(f"'{op}' filter requires at least two operands.")
        self.op = op
        self.operands = operands

    # Boolean operators (`& | ^ ~`) are inherited from `Filter`.

    def to_attr(self) -> "Attr":
        """Translate the compound filter to an `Attr` holding the polars boolean expression.

        Mirrors [AttrComparison.to_attr][tracksdata.attrs.AttrComparison.to_attr]
        and folds children polymorphically — both operand types expose `to_attr`,
        so no parallel `(Filter)` walker is needed for
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
                assert isinstance(o, AttrComparison)
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


def _filter_attr_kind(f: Filter) -> type[Attr]:
    """Return the leaf-attribute kind (NodeAttr / EdgeAttr) of a filter.

    Raises ValueError if the filter mixes node and edge attributes.
    """
    if isinstance(f, AttrComparison):
        if isinstance(f.attr, NodeAttr):
            return NodeAttr
        if isinstance(f.attr, EdgeAttr):
            return EdgeAttr
        raise ValueError(f"Expected comparisons of 'NodeAttr' or 'EdgeAttr' objects, got {type(f.attr)}")

    assert isinstance(f, AttrFilter)
    kinds = {_filter_attr_kind(o) for o in f.operands}
    if len(kinds) > 1:
        raise ValueError(
            "A single AttrFilter compound cannot mix NodeAttr and EdgeAttr comparisons. "
            "Combine node and edge filters via separate positional arguments to graph.filter()."
        )
    return kinds.pop()


def split_attr_comps(
    attr_comps: Sequence[Filter],
) -> tuple[list[Filter], list[Filter]]:
    """
    Split a list of attribute comparisons (or compound filters) into node and
    edge groups based on the kind of their leaf comparisons.

    Parameters
    ----------
    attr_comps : Sequence[Filter]
        The attribute comparisons or compound filters to split.

    Returns
    -------
    tuple[list[Filter], list[Filter]]
        A tuple of lists of node and edge filters.
    """
    node_attr_comps: list[Filter] = []
    edge_attr_comps: list[Filter] = []

    for attr_comp in attr_comps:
        kind = _filter_attr_kind(attr_comp)
        if kind is NodeAttr:
            node_attr_comps.append(attr_comp)
        else:
            edge_attr_comps.append(attr_comp)

    return node_attr_comps, edge_attr_comps


def attr_comps_to_strs(attr_comps: Sequence[Filter]) -> list[str]:
    """
    Convert a list of attribute comparisons (or compound filters) to a list of
    column names involved in them.

    Parameters
    ----------
    attr_comps : Sequence[Filter]
        The filters to extract column names from.

    Returns
    -------
    list[str]
        The column names referenced by the filters, deduplicated while
        preserving order.
    """
    # Both subclasses of `Filter` expose `.columns`.
    return list(dict.fromkeys(c for ac in attr_comps for c in ac.columns))


def polars_reduce_attr_comps(
    attr_comps: Sequence[Filter],
    reduce_op: Callable[[Expr, Expr], Expr],
) -> pl.Expr:
    """
    Reduce a list of attribute comparisons (or compound filters) into a single
    polars expression, combined with `reduce_op` at the top level (AND-ed by
    default in callers).

    Parameters
    ----------
    attr_comps : Sequence[Filter]
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
    # `f.to_attr().expr` lets each filter render its own expression. For
    # `AttrComparison` over a struct field, that expression already drills into
    # the struct (e.g. `pl.col("m").struct.field("x")`) rather than reading the
    # bare column. For compound `AttrFilter`s, it returns the combined boolean
    # expression.
    return pl.reduce(reduce_op, [f.to_attr().expr for f in attr_comps])

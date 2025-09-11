import numpy as np
import polars as pl
from ilpy import (
    Constraints,
    Objective,
    Preference,
    Solution,
    Solver,
    SolverStatus,
    Variable,
    VariableType,
)

from tracksdata.attrs import Attr, EdgeAttr, ExprInput, NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.graph._graph_view import GraphView
from tracksdata.solvers._base_solver import BaseSolver
from tracksdata.utils._logging import LOG


class ILPSolver(BaseSolver):
    """
    Optimal tracking solver using Integer Linear Programming (ILP).

    ILPSolver formulates the multi-object tracking problem as an Integer Linear
    Programming optimization problem. It finds globally optimal solutions by
    minimizing a cost function while satisfying flow conservation constraints.
    The solver supports appearance, disappearance, and division events, making
    it suitable for complex biological tracking scenarios.

    The optimization problem includes:

    - Node selection variables (whether a detection is part of a track)
    - Edge selection variables (connections between detections)
    - Appearance variables (track starts)
    - Disappearance variables (track ends)
    - Division variables (cell divisions or track splits)

    Parameters
    ----------
    edge_weight : str | ExprInput, default DEFAULT_ATTR_KEYS.EDGE_WEIGHT
        Edge attribute or expression to use as edge costs in the optimization.
        Lower values indicate preferred connections.
    node_weight : str | ExprInput, default 0.0
        Node attribute or expression to use as node costs.
    appearance_weight : str | ExprInput, default 0.0
        Cost for track appearances (new tracks starting).
    disappearance_weight : str | ExprInput, default 0.0
        Cost for track disappearances (tracks ending).
    division_weight : str | ExprInput, default 0.0
        Cost for track divisions (one track splitting into two).
    output_key : str, default DEFAULT_ATTR_KEYS.SOLUTION
        Attribute key to store the solution (True/False for selected items).
    num_threads : int, default 1
        Number of threads to use for solving.
    reset : bool, default True
        Whether to reset previous solutions before solving.
    return_solution : bool, default True
        Whether to return a subgraph containing only the solution.
    gap : float, default 0.0
        Optimality gap tolerance (0.0 for exact solutions).
    timeout : float | None
        Time limit for solving in seconds. If None, no timeout is set.

    Attributes
    ----------
    edge_weight_expr : EdgeAttr
        Compiled edge weight expression.
    node_weight_expr : NodeAttr
        Compiled node weight expression.
    appearance_weight_expr : NodeAttr
        Compiled appearance weight expression.
    disappearance_weight_expr : NodeAttr
        Compiled disappearance weight expression.
    division_weight_expr : NodeAttr
        Compiled division weight expression.

    See Also
    --------
    [NearestNeighborsSolver][tracksdata.solvers.NearestNeighborsSolver]:
        Greedy nearest neighbors tracking solver.

    [NodeAttr][tracksdata.attrs.NodeAttr]:
        For creating node attribute expressions.

    [EdgeAttr][tracksdata.attrs.EdgeAttr]:
        For creating edge attribute expressions.

    Examples
    --------
    Basic tracking with distance-based costs:

    ```python
    from tracksdata.solvers import ILPSolver

    solver = ILPSolver(edge_weight="distance")
    solution = solver.solve(graph)
    ```

    Tracking with appearance and disappearance costs:

    ```python
    solver = ILPSolver(edge_weight="distance", appearance_weight=10.0, disappearance_weight=10.0)
    solution = solver.solve(graph)
    ```

    Using attribute expressions for complex costs:

    ```python
    from tracksdata.attrs import EdgeAttr

    solver = ILPSolver(edge_weight=EdgeAttr("distance") + 0.1 * EdgeAttr("angle_change"), division_weight=5.0)
    solution = solver.solve(graph)
    ```

    Notes
    -----
    The solver uses the [ilpy](https://github.com/funkelab/ilpy) library which provides interfaces
    to commercial solvers (Gurobi) and open-source solvers (SCIP).
    For best performance, install Gurobi if available, otherwise SCIP will be used as fallback.

    The ILP formulation ensures:

    - Flow conservation: incoming flow equals outgoing flow for each node
    - Track consistency: each detection belongs to at most one track
    - Optimal solution: globally minimizes the total cost function
    """

    def __init__(
        self,
        *,
        edge_weight: str | ExprInput = DEFAULT_ATTR_KEYS.EDGE_DIST,
        node_weight: str | ExprInput = 0.0,
        appearance_weight: str | ExprInput = 0.0,
        disappearance_weight: str | ExprInput = 0.0,
        division_weight: str | ExprInput = 0.0,
        output_key: str = DEFAULT_ATTR_KEYS.SOLUTION,
        num_threads: int = 1,
        reset: bool = True,
        return_solution: bool = True,
        gap: float = 0.0,
        timeout: float | None = None,
    ):
        super().__init__(output_key=output_key, reset=reset, return_solution=return_solution)
        self.edge_weight_expr = EdgeAttr(edge_weight)
        self.node_weight_expr = NodeAttr(node_weight)
        self.appearance_weight_expr = NodeAttr(appearance_weight)
        self.disappearance_weight_expr = NodeAttr(disappearance_weight)
        self.division_weight_expr = NodeAttr(division_weight)
        self.num_threads = num_threads
        self.gap = gap
        self.timeout = timeout
        self.reset_model()

    def reset_model(self) -> None:
        self._objective = Objective()
        self._count = 0
        self._constraints = Constraints()
        self._node_vars = {}
        self._appear_vars = {}
        self._disappear_vars = {}
        self._division_vars = {}
        self._edge_vars = {}

    def _evaluate_expr(
        self,
        expr: Attr,
        df: pl.DataFrame,
    ) -> list[float]:
        if len(expr.expr_columns) == 0:
            return [expr.evaluate(df).item()] * len(df)
        else:
            return expr.evaluate(df).to_list()

    def _evaluate_inf_expr(
        self,
        inf_expr: list[Attr],
        df: pl.DataFrame,
        node_key: str,
    ) -> list[int]:
        """
        Evaluate a list of infinity expressions and return the node ids that satisfy the expressions.

        Parameters
        ----------
        inf_expr : list[AttrExpr]
            The list of infinity expressions to evaluate.
        df : pl.DataFrame
            The dataframe to evaluate the expressions on.
        node_key : str
            The key of the node column to filter on.

        Returns
        -------
        list[int]
            The node ids that satisfy the expressions.
        """
        if len(inf_expr) == 0:
            return []
        mask = False
        for expr in inf_expr:
            mask = mask | expr.evaluate(df)
        return df.select(node_key).filter(mask).to_series().to_list()

    def _add_objective_and_variables(
        self,
        nodes_df: pl.DataFrame,
        edges_df: pl.DataFrame,
    ) -> None:
        node_ids = nodes_df[DEFAULT_ATTR_KEYS.NODE_ID].to_list()

        num_new_variables = len(node_ids) * 4 + len(edges_df)

        self._objective.resize(self._count + num_new_variables)

        for name, variables, expr in zip(
            ["node", "appear", "disappear", "division"],
            [self._node_vars, self._appear_vars, self._disappear_vars, self._division_vars],
            [
                self.node_weight_expr,
                self.appearance_weight_expr,
                self.disappearance_weight_expr,
                self.division_weight_expr,
            ],
            strict=True,
        ):
            weights = self._evaluate_expr(expr, nodes_df)

            for node_id, weight in zip(node_ids, weights, strict=True):
                variables[node_id] = Variable(f"{name}_{node_id}", index=self._count)
                self._objective.set_coefficient(self._count, weight)
                self._count += 1

            # positive inf, pin to 0
            for node_id in self._evaluate_inf_expr(expr.inf_exprs, nodes_df, DEFAULT_ATTR_KEYS.NODE_ID):
                self._constraints.add(variables[node_id] == 0)

            # negative inf, pin to 1
            for node_id in self._evaluate_inf_expr(expr.neg_inf_exprs, nodes_df, DEFAULT_ATTR_KEYS.NODE_ID):
                self._constraints.add(variables[node_id] == 1)

        # edge weights
        weights = self._evaluate_expr(self.edge_weight_expr, edges_df)
        edge_ids = edges_df[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()

        for edge_id, weight in zip(edge_ids, weights, strict=False):
            self._edge_vars[edge_id] = Variable(f"edge_{edge_id}", index=self._count)
            self._objective.set_coefficient(self._count, weight)
            self._count += 1

        # positive inf, pin to 0
        for edge_id in self._evaluate_inf_expr(self.edge_weight_expr.inf_exprs, edges_df, DEFAULT_ATTR_KEYS.EDGE_ID):
            self._constraints.add(self._edge_vars[edge_id] == 0)

        # negative inf, pin to 1
        for edge_id in self._evaluate_inf_expr(
            self.edge_weight_expr.neg_inf_exprs, edges_df, DEFAULT_ATTR_KEYS.EDGE_ID
        ):
            self._constraints.add(self._edge_vars[edge_id] == 1)

    def _add_continuous_flow_constraints(
        self,
        node_ids: list[int],
        edges_df: pl.DataFrame,
    ) -> None:
        # fewer columns are faster to group by
        edges_df = edges_df.select(
            [
                DEFAULT_ATTR_KEYS.EDGE_TARGET,
                DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                DEFAULT_ATTR_KEYS.EDGE_ID,
            ]
        )

        unseen_in_nodes = set(node_ids)
        unseen_out_nodes = unseen_in_nodes.copy()

        # incoming flow
        for (target_id,), group in edges_df.group_by(DEFAULT_ATTR_KEYS.EDGE_TARGET):
            edge_ids = group[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()
            self._constraints.add(
                self._appear_vars[target_id] + sum(self._edge_vars[edge_id] for edge_id in edge_ids)
                == self._node_vars[target_id]
            )
            unseen_in_nodes.remove(target_id)

        for node_id in unseen_in_nodes:
            self._constraints.add(self._appear_vars[node_id] == self._node_vars[node_id])

        # outgoing flow
        for (source_id,), group in edges_df.group_by(DEFAULT_ATTR_KEYS.EDGE_SOURCE):
            edge_ids = group[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()
            self._constraints.add(
                self._disappear_vars[source_id] + sum(self._edge_vars[edge_id] for edge_id in edge_ids)
                == self._node_vars[source_id] + self._division_vars[source_id]
            )
            unseen_out_nodes.remove(source_id)

        for node_id in unseen_out_nodes:
            self._constraints.add(
                self._disappear_vars[node_id] == self._node_vars[node_id] + self._division_vars[node_id]
            )

        # existing division from nodes
        for node_id, node_var in self._node_vars.items():
            self._constraints.add(node_var >= self._division_vars[node_id])

    def _add_overlap_constraints(
        self,
        overlaps: list[list[int, 2]],
    ) -> None:
        # this could be improved to avoid pairwise constraints by using
        # mutual exclusivity set constraints for a easier ILP formulation
        for source_id, target_id in overlaps:
            self._constraints.add(self._node_vars[source_id] + self._node_vars[target_id] <= 1)

    def _solve(self) -> Solution:
        if self._count == 0:
            raise ValueError(r"Empty ILPSolver model, there is nothing to solve.")

        elif len(self._edge_vars) == 0:
            raise ValueError(r"No edges found in the graph, there is nothing to solve.")

        solution = None
        for preference in [Preference.Gurobi, Preference.Scip]:
            try:
                solver = Solver(
                    num_variables=self._count,
                    default_variable_type=VariableType.Binary,
                    preference=preference,
                )
                solver.set_num_threads(self.num_threads)
                solver.set_objective(self._objective)
                solver.set_constraints(self._constraints)
                solver.set_optimality_gap(self.gap)
                if self.timeout is not None:
                    solver.set_timeout(self.timeout)
                solution = solver.solve()
                break
            except Exception as e:
                LOG.warning(f"Solver failed with {preference.name}, trying Scip.\nGot error:\n{e}")
                continue

        if solution is None:
            raise RuntimeError("Failed to solve the ILP problem with any solver.")

        if not np.asarray(solution, dtype=bool).any():
            LOG.warning("Trivial solution found with all variables set to 0!")
            return solution

        if solution.status != SolverStatus.OPTIMAL:
            LOG.warning(f"Solver did not converge to an optimal solution, returned status {solution.status}.")

        return solution

    def solve(
        self,
        graph: BaseGraph,
    ) -> GraphView | None:
        nodes_df = graph.node_attrs(
            attr_keys=[
                DEFAULT_ATTR_KEYS.NODE_ID,
                *self.node_weight_expr.columns,
                *self.appearance_weight_expr.columns,
                *self.disappearance_weight_expr.columns,
                *self.division_weight_expr.columns,
            ],
        )
        edges_df = graph.edge_attrs(
            attr_keys=self.edge_weight_expr.columns,
        )

        self._add_objective_and_variables(nodes_df, edges_df)
        self._add_continuous_flow_constraints(nodes_df[DEFAULT_ATTR_KEYS.NODE_ID].to_list(), edges_df)

        if graph.has_overlaps():
            self._add_overlap_constraints(graph.overlaps())

        solution = self._solve()

        selected_nodes = [node_id for node_id, var in self._node_vars.items() if solution[var.index] > 0.5]

        if self.output_key not in graph.node_attr_keys:
            graph.add_node_attr_key(self.output_key, False)
        elif self.reset:
            graph.update_node_attrs(attrs={self.output_key: False})

        graph.update_node_attrs(
            node_ids=selected_nodes,
            attrs={self.output_key: True},
        )

        selected_edges = [edge_id for edge_id, var in self._edge_vars.items() if solution[var.index] > 0.5]

        if self.output_key not in graph.edge_attr_keys:
            graph.add_edge_attr_key(self.output_key, False)
        elif self.reset:
            graph.update_edge_attrs(attrs={self.output_key: False})

        graph.update_edge_attrs(
            edge_ids=selected_edges,
            attrs={self.output_key: True},
        )

        if self.return_solution:
            return graph.filter(
                NodeAttr(self.output_key) == True,
                EdgeAttr(self.output_key) == True,
            ).subgraph()

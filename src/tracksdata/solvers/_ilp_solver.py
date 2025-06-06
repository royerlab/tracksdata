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

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.expr import AttrExpr, ExprInput
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.solvers._base_solver import BaseSolver
from tracksdata.utils._logging import LOG


class ILPSolver(BaseSolver):
    """
    Solver tracking problem with integer linear programming.
    """

    def __init__(
        self,
        *,
        edge_weight: str | AttrExpr = DEFAULT_ATTR_KEYS.EDGE_WEIGHT,
        node_weight: str | ExprInput = 0.0,
        appearance_weight: str | ExprInput = 0.0,
        disappearance_weight: str | ExprInput = 0.0,
        division_weight: str | ExprInput = 0.0,
        output_key: str = DEFAULT_ATTR_KEYS.SOLUTION,
        num_threads: int = 1,
    ):
        self.edge_weight_expr = AttrExpr(edge_weight)
        self.node_weight_expr = AttrExpr(node_weight)
        self.appearance_weight_expr = AttrExpr(appearance_weight)
        self.disappearance_weight_expr = AttrExpr(disappearance_weight)
        self.division_weight_expr = AttrExpr(division_weight)
        self.output_key = output_key
        self.num_threads = num_threads
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
        expr: AttrExpr,
        df: pl.DataFrame,
    ) -> list[float]:
        if len(expr.column_names()) == 0:
            return [expr.evaluate(df).item()] * len(df)
        else:
            return expr.evaluate(df).to_list()

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
            strict=False,
        ):
            weights = self._evaluate_expr(expr, nodes_df)

            # TODO: setup some kind of filter to update weights, for example:
            # - starting and ending frames
            # - closeness to the image boundaries

            for node_id, weight in zip(node_ids, weights, strict=True):
                variables[node_id] = Variable(f"{name}_{node_id}", index=self._count)
                self._objective.set_coefficient(self._count, weight)
                self._count += 1

        weights = self._evaluate_expr(self.edge_weight_expr, edges_df)

        for edge_id, weight in zip(edges_df[DEFAULT_ATTR_KEYS.EDGE_ID].to_list(), weights, strict=False):
            self._edge_vars[edge_id] = Variable(f"edge_{edge_id}", index=self._count)
            self._objective.set_coefficient(self._count, weight)
            self._count += 1

    def _add_constraints(
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

    def _solve(self) -> Solution:
        if self._count == 0:
            raise ValueError("Empty ILPSolver model, there is nothing to solve.")

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
                solution = solver.solve()
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
    ) -> None:
        nodes_df = graph.node_features(
            feature_keys=[
                DEFAULT_ATTR_KEYS.NODE_ID,
                *self.node_weight_expr.column_names(),
                *self.appearance_weight_expr.column_names(),
                *self.disappearance_weight_expr.column_names(),
                *self.division_weight_expr.column_names(),
            ],
        )
        edges_df = graph.edge_features(
            feature_keys=self.edge_weight_expr.column_names(),
        )

        self._add_objective_and_variables(nodes_df, edges_df)
        self._add_constraints(nodes_df[DEFAULT_ATTR_KEYS.NODE_ID].to_list(), edges_df)

        solution = self._solve()

        selected_nodes = [node_id for node_id, var in self._node_vars.items() if solution[var.index] > 0.5]

        if self.output_key not in graph.node_features_keys:
            graph.add_node_feature_key(self.output_key, False)

        graph.update_node_features(
            node_ids=selected_nodes,
            attributes={self.output_key: True},
        )

        selected_edges = [edge_id for edge_id, var in self._edge_vars.items() if solution[var.index] > 0.5]

        if self.output_key not in graph.edge_features_keys:
            graph.add_edge_feature_key(self.output_key, False)

        graph.update_edge_features(
            edge_ids=selected_edges,
            attributes={self.output_key: True},
        )

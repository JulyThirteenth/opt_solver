import numpy as np
from copy import copy
from typing import Callable, Tuple

from opt_uncons_solver import practical_lbfgs_optimizer


class PHR_ALM_Solver:
    """Powell-Hestenes-Rockafellar Augmented Lagrangian Method (PHR-ALM) solver.

    This class implements the PHR-ALM algorithm for constrained optimization
    problems with equality and inequality constraints.

    Attributes:
        dim_opti_vars (int): Dimension of optimization variables
        dim_eq_cons (int): Dimension of equality constraints
        dim_ieq_cons (int): Dimension of inequality constraints
        rho (float): Penalty parameter growth factor
        tau (float): Constraint improvement threshold
        beta (float): Initial penalty parameter
        outer_max_iter (int): Maximum outer iterations
        inner_max_iter (int): Maximum inner iterations
        inner_optimizer (str): Inner unconstrained optimizer name
        lambda_eq (np.ndarray): Equality constraint multipliers
        lambda_ieq (np.ndarray): Inequality constraint multipliers
        mu (float): Current penalty parameter
        cost_value_and_grad (Callable): Objective value and gradient function
        eq_cons_value_and_jac (Callable): Equality constraint value and Jacobian function
        ieq_cons_value_and_jac (Callable): Inequality constraint value and Jacobian function
        aug_lagrangian_value_and_grad (Callable): Optional specialized augmented
            Lagrangian value and gradient function
    """

    def __init__(
        self,
        dim_opti_vars: int,
        dim_eq_cons: int = 0,
        dim_ieq_cons: int = 0,
        rho: float = 1.5,
        tau: float = 0.25,
        beta: float = 1e3,
        outer_max_iter: int = 60,
        inner_max_iter: int = 1000,
    ):
        """Initializes the PHR-ALM solver.

        Args:
            dim_opti_vars: Dimension of optimization variables
            dim_eq_cons: Dimension of equality constraints
            dim_ieq_cons: Dimension of inequality constraints
            rho: Penalty parameter growth factor (ρ > 1)
            tau: Increase penalty if violation does not drop below this
                fraction of the previous violation
            beta: Initial penalty parameter (β > 0)
            outer_max_iter: Maximum outer iterations
            inner_max_iter: Maximum inner iterations
        """
        # Validate input parameters
        if dim_opti_vars <= 0:
            raise ValueError(f"dim_opti_vars must be positive, got {dim_opti_vars}")
        if dim_eq_cons < 0:
            raise ValueError(f"dim_eq_cons must be non-negative, got {dim_eq_cons}")
        if dim_ieq_cons < 0:
            raise ValueError(f"dim_ieq_cons must be non-negative, got {dim_ieq_cons}")
        if rho <= 1.0:
            raise ValueError(f"rho must be > 1, got {rho}")
        if not 0.0 < tau < 1.0:
            raise ValueError(
                "penalty_update_threshold must be in (0, 1), " f"got {tau}"
            )
        if beta <= 0.0:
            raise ValueError(f"beta must be positive, got {beta}")
        if outer_max_iter <= 0:
            raise ValueError(f"outer_max_iter must be positive, got {outer_max_iter}")
        if inner_max_iter <= 0:
            raise ValueError(f"inner_max_iter must be positive, got {inner_max_iter}")

        # Store problem dimensions
        self.dim_opti_vars = dim_opti_vars
        self.dim_eq_cons = dim_eq_cons
        self.dim_ieq_cons = dim_ieq_cons

        # Store algorithm parameters
        self.rho = rho
        self.tau = tau
        self.beta = beta
        self.outer_max_iter = outer_max_iter
        self.inner_max_iter = inner_max_iter

        # Initialize multipliers and penalty parameter
        self.lambda_eq = np.zeros(dim_eq_cons)
        self.lambda_ieq = np.zeros(dim_ieq_cons)
        self.mu = beta

        # Initialize function pointers
        self.cost_value_and_grad = None
        self.eq_cons_value_and_jac = None
        self.ieq_cons_value_and_jac = None
        self.aug_lagrangian_value_and_grad = None

        # Initialize history
        self.inner_iter_history = []
        self.cons_violation_history = []
        self.alm_violation_history = []
        self.mu_history = []

    def set_cost(
        self, value_and_grad: Callable[[np.ndarray], Tuple[float, np.ndarray]]
    ) -> None:
        """Sets the objective value-and-gradient function.

        Args:
            value_and_grad: Function returning objective value and gradient
        """
        self.cost_value_and_grad = value_and_grad

    def set_eq_cons(
        self,
        value_and_jac: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Sets the equality constraint value-and-Jacobian function.

        Args:
            value_and_jac: Function returning h(x) and ∇h(x)
        """
        if self.dim_eq_cons == 0:
            raise RuntimeError("dim_eq_cons is 0 - cannot set equality constraints")
        self.eq_cons_value_and_jac = value_and_jac

    def set_ieq_cons(
        self,
        value_and_jac: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Sets the inequality constraint value-and-Jacobian function.

        Args:
            value_and_jac: Function returning g(x) and ∇g(x)
        """
        if self.dim_ieq_cons == 0:
            raise RuntimeError("dim_ieq_cons is 0 - cannot set inequality constraints")
        self.ieq_cons_value_and_jac = value_and_jac

    def set_augmented_lagrangian(
        self,
        value_and_grad: Callable[
            [np.ndarray, np.ndarray, np.ndarray, float], Tuple[float, np.ndarray]
        ],
    ) -> None:
        """Sets an optional specialized augmented Lagrangian value-gradient function.

        Args:
            value_and_grad: Function returning augmented Lagrangian value and
                gradient for the current x, lambda_eq, lambda_ieq, and mu.
        """
        self.aug_lagrangian_value_and_grad = value_and_grad

    def _validate_functions(self) -> None:
        """Validates that all required functions are set."""
        if self.cost_value_and_grad is None:
            raise RuntimeError("Cost value-and-gradient function not set")
        if self.dim_eq_cons > 0 and self.eq_cons_value_and_jac is None:
            raise RuntimeError(
                "Equality constraint value-and-Jacobian function not set"
            )
        if self.dim_ieq_cons > 0 and self.ieq_cons_value_and_jac is None:
            raise RuntimeError(
                "Inequality constraint value-and-Jacobian function not set"
            )

    def _augmented_lagrangian_value_and_grad(
        self, x: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Computes augmented Lagrangian value and gradient together."""
        if self.aug_lagrangian_value_and_grad is not None:
            aug_val, grad = self.aug_lagrangian_value_and_grad(
                x, self.lambda_eq, self.lambda_ieq, self.mu
            )
            return float(np.asarray(aug_val)), np.asarray(grad).flatten()

        assert self.cost_value_and_grad is not None, "cost_value_and_grad must be set"
        aug_val, grad = self.cost_value_and_grad(x)
        aug_val = float(np.asarray(aug_val))
        grad = np.asarray(grad).flatten()

        if self.dim_eq_cons > 0:
            assert (
                self.eq_cons_value_and_jac is not None
            ), "eq_cons_value_and_jac must be set when dim_eq_cons > 0"
            c_eq, grad_eq = self.eq_cons_value_and_jac(x)
            c_eq = np.asarray(c_eq).flatten()
            grad_eq = np.asarray(grad_eq)
            lambda_scaled_eq = self.lambda_eq / self.mu
            scaled_eq = c_eq + lambda_scaled_eq
            aug_val += (self.mu / 2) * (
                np.sum(scaled_eq**2) - np.sum(lambda_scaled_eq**2)
            )
            grad += self.mu * np.dot(scaled_eq, grad_eq)

        if self.dim_ieq_cons > 0:
            assert (
                self.ieq_cons_value_and_jac is not None
            ), "ieq_cons_value_and_jac must be set when dim_ieq_cons > 0"
            c_ieq, grad_ieq = self.ieq_cons_value_and_jac(x)
            c_ieq = np.asarray(c_ieq).flatten()
            grad_ieq = np.asarray(grad_ieq)
            lambda_scaled_ieq = self.lambda_ieq / self.mu
            scaled_ieq = c_ieq + lambda_scaled_ieq
            phi = np.maximum(0, scaled_ieq)
            aug_val += (self.mu / 2) * (np.sum(phi**2) - np.sum(lambda_scaled_ieq**2))
            grad += self.mu * np.dot(phi, grad_ieq)

        return float(aug_val), np.asarray(grad).flatten()

    def _solve_inner_problem(self, x0: np.ndarray) -> np.ndarray:
        """Solves the inner unconstrained optimization problem.

        Args:
            x0: Initial point for the inner problem

        Returns:
            Optimized solution for the inner problem
        """
        result = practical_lbfgs_optimizer(
            x_init=x0,
            value_and_grad=self._augmented_lagrangian_value_and_grad,
            max_iterations=self.inner_max_iter,
            verbose=False,
            store_history=False,
        )
        inner_iters_val = result.get("iterations", 0)
        inner_iters = (
            int(inner_iters_val) if isinstance(inner_iters_val, (int, float)) else 0
        )
        self.inner_iter_history.append(inner_iters)
        x_opt = np.asarray(result.get("x_opt", x0))
        return x_opt

    def _evaluate_constraint_values(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates equality and inequality constraint values once."""
        if self.dim_eq_cons > 0:
            assert (
                self.eq_cons_value_and_jac is not None
            ), "eq_cons_value_and_jac must be set when dim_eq_cons > 0"
            c_eq, _ = self.eq_cons_value_and_jac(x)
            c_eq = np.asarray(c_eq).flatten()
        else:
            c_eq = np.zeros(0)

        if self.dim_ieq_cons > 0:
            assert (
                self.ieq_cons_value_and_jac is not None
            ), "ieq_cons_value_and_jac must be set when dim_ieq_cons > 0"
            c_ieq, _ = self.ieq_cons_value_and_jac(x)
            c_ieq = np.asarray(c_ieq).flatten()
        else:
            c_ieq = np.zeros(0)

        return c_eq, c_ieq

    def _constraint_violation_from_values(
        self, c_eq: np.ndarray, c_ieq: np.ndarray
    ) -> float:
        """Computes total constraint violation from cached constraint values."""
        violation = 0.0

        if self.dim_eq_cons > 0:
            violation += np.linalg.norm(c_eq, 2)

        if self.dim_ieq_cons > 0:
            violation += np.linalg.norm(np.maximum(0, c_ieq), 2)

        return float(violation)

    def _alm_violation_from_values(self, c_eq: np.ndarray, c_ieq: np.ndarray) -> float:
        """Computes ALM violation from cached constraint values."""
        violation = 0.0

        if self.dim_eq_cons > 0:
            violation += np.linalg.norm(c_eq, 2)

        if self.dim_ieq_cons > 0:
            violation += np.linalg.norm(
                np.maximum(c_ieq, -self.lambda_ieq / self.mu), 2
            )

        return float(violation)

    def solve(self, x0: np.ndarray, tol: float = 1e-6, verbose: bool = False) -> dict:
        """Solves the constrained optimization problem using PHR-ALM.

        Args:
            x0: Initial point in parameter space
            tol: Tolerance for constraint violation
            verbose: Whether to print progress information

        Returns:
            Dictionary with solver information containing:
                'status': Convergence status
                'opti_vars': Optimized solution
        """
        # Validate input
        if not isinstance(x0, np.ndarray) or x0.size != self.dim_opti_vars:
            raise ValueError(
                f"x0 must be numpy array of size {self.dim_opti_vars}, "
                f"got {x0.shape if isinstance(x0, np.ndarray) else type(x0)}"
            )
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}")

        self._validate_functions()

        # Initialize variables
        x = copy(x0)
        convergence_status = "max_outer_iterations"
        c_eq, c_ieq = self._evaluate_constraint_values(x)
        prev_alm_violation = self._alm_violation_from_values(c_eq, c_ieq)
        outer_iterations = 0

        # Outer loop of ALM
        for outer_iter in range(self.outer_max_iter):
            outer_iterations = outer_iter + 1
            # Solve inner unconstrained problem
            x = self._solve_inner_problem(x)

            # Compute constraint values once, then reuse them below.
            c_eq, c_ieq = self._evaluate_constraint_values(x)
            cons_violation = self._constraint_violation_from_values(c_eq, c_ieq)
            alm_violation = self._alm_violation_from_values(c_eq, c_ieq)
            self.cons_violation_history.append(cons_violation)
            self.alm_violation_history.append(alm_violation)
            self.mu_history.append(self.mu)

            # Check convergence
            if cons_violation < tol:
                convergence_status = "converged"
                if verbose:
                    print(f"Converged at outer iteration {outer_iter}")
                break

            # Update multipliers
            if self.dim_eq_cons > 0:
                self.lambda_eq += self.mu * c_eq

            if self.dim_ieq_cons > 0:
                self.lambda_ieq = np.maximum(0, self.lambda_ieq + self.mu * c_ieq)

            # Increase the penalty otauress is insufficient.
            if alm_violation > self.tau * prev_alm_violation:
                self.mu *= self.rho
            prev_alm_violation = alm_violation

            if verbose:
                print(
                    f"Outer iter {outer_iter}: "
                    f"constraint violation = {cons_violation:.4e}, "
                    f"ALM violation = {alm_violation:.4e}, "
                    f"μ = {self.mu:.1e}"
                )
        else:
            if verbose:
                print("Reached maximum outer iterations")

        # Prepare solver info
        solver_info = {
            "status": convergence_status,
            "opti_vars": x,
            "outer_iterations": outer_iterations,
            "inner_iterations": sum(self.inner_iter_history),
            "final_constraint_violation": cons_violation,
            "final_penalty": self.mu,
            "constraint_violation_history": np.array(self.cons_violation_history),
            "alm_violation_history": np.array(self.alm_violation_history),
            "penalty_history": np.array(self.mu_history),
            "inner_iter_counts": np.array(self.inner_iter_history),
        }

        return solver_info

    def eval_objective(self, x: np.ndarray) -> float:
        """Evaluates the objective function at x.

        Args:
            x: Point in parameter space

        Returns:
            Objective function value
        """
        if self.cost_value_and_grad is None:
            raise RuntimeError("Objective value-and-gradient function not set")
        cost_value, _ = self.cost_value_and_grad(x)
        return float(np.asarray(cost_value))

    def eval_gradient(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the objective gradient at x.

        Args:
            x: Point in parameter space

        Returns:
            Objective gradient
        """
        if self.cost_value_and_grad is None:
            raise RuntimeError("Objective value-and-gradient function not set")
        _, cost_grad = self.cost_value_and_grad(x)
        return np.asarray(cost_grad).flatten()

    def reset(self) -> None:
        """Resets the solver state (multipliers and penalty parameter)."""
        self.lambda_eq = np.zeros(self.dim_eq_cons)
        self.lambda_ieq = np.zeros(self.dim_ieq_cons)
        self.mu = self.beta
        self.inner_iter_history = []
        self.cons_violation_history = []
        self.alm_violation_history = []
        self.mu_history = []

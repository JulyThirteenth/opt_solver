from opt_uncons_solver import practical_lbfgs_optimizer


import numpy as np
from copy import copy
from typing import Callable, Tuple


class PHR_ALM_Solver:
    """Powell-Hestenes-Rockafellar Augmented Lagrangian Method (PHR-ALM) solver.

    This class implements the PHR-ALM algorithm for constrained optimization
    problems with equality and inequality constraints.

    Attributes:
        dim_opti_vars (int): Dimension of optimization variables
        dim_eq_cons (int): Dimension of equality constraints
        dim_ieq_cons (int): Dimension of inequality constraints
        rho (float): Penalty parameter growth factor
        gamma (float): Step size for multiplier updates
        beta (float): Initial penalty parameter
        outer_max_iter (int): Maximum outer iterations
        inner_max_iter (int): Maximum inner iterations
        lambda_eq (np.ndarray): Equality constraint multipliers
        lambda_ieq (np.ndarray): Inequality constraint multipliers
        mu (float): Current penalty parameter
        cost_func (Callable): Objective function
        cost_grad (Callable): Objective gradient
        eq_cons_func (Callable): Equality constraint function
        eq_cons_grad (Callable): Equality constraint gradient
        ieq_cons_func (Callable): Inequality constraint function
        ieq_cons_grad (Callable): Inequality constraint gradient
    """

    def __init__(
        self,
        dim_opti_vars: int,
        dim_eq_cons: int = 0,
        dim_ieq_cons: int = 0,
        rho: float = 1.0,
        gamma: float = 1.0,
        beta: float = 1e3,
        outer_max_iter: int = 60,
        inner_max_iter: int = 1000
    ):
        """Initializes the PHR-ALM solver.

        Args:
            dim_opti_vars: Dimension of optimization variables
            dim_eq_cons: Dimension of equality constraints
            dim_ieq_cons: Dimension of inequality constraints
            rho: Penalty parameter growth factor (ρ > 1)
            gamma: Step size for multiplier updates (γ > 0)
            beta: Initial penalty parameter (β > 0)
            outer_max_iter: Maximum outer iterations
            inner_max_iter: Maximum inner iterations
        """
        # Validate input parameters
        if dim_opti_vars <= 0:
            raise ValueError(
                f"dim_opti_vars must be positive, got {dim_opti_vars}")
        if dim_eq_cons < 0:
            raise ValueError(
                f"dim_eq_cons must be non-negative, got {dim_eq_cons}")
        if dim_ieq_cons < 0:
            raise ValueError(
                f"dim_ieq_cons must be non-negative, got {dim_ieq_cons}")
        if rho <= 1.0:
            raise ValueError(f"rho must be > 1, got {rho}")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if beta <= 0.0:
            raise ValueError(f"beta must be positive, got {beta}")
        if outer_max_iter <= 0:
            raise ValueError(
                f"outer_max_iter must be positive, got {outer_max_iter}")
        if inner_max_iter <= 0:
            raise ValueError(
                f"inner_max_iter must be positive, got {inner_max_iter}")

        # Store problem dimensions
        self.dim_opti_vars = dim_opti_vars
        self.dim_eq_cons = dim_eq_cons
        self.dim_ieq_cons = dim_ieq_cons

        # Store algorithm parameters
        self.rho = rho
        self.gamma = gamma
        self.beta = beta
        self.outer_max_iter = outer_max_iter
        self.inner_max_iter = inner_max_iter

        # Initialize multipliers and penalty parameter
        self.lambda_eq = np.zeros(dim_eq_cons)
        self.lambda_ieq = np.zeros(dim_ieq_cons)
        self.mu = beta

        # Initialize function pointers
        self.cost_func = None
        self.cost_grad = None
        self.eq_cons_func = None
        self.eq_cons_grad = None
        self.ieq_cons_func = None
        self.ieq_cons_grad = None

        # Initialize history
        self.outer_iter_history = []
        self.inner_iter_history = []
        self.cons_violation_history = []
        self.mu_history = []

    def set_cost(self, cost_func: Callable[[np.ndarray], float],
                 cost_grad: Callable[[np.ndarray], np.ndarray]) -> None:
        """Sets the objective function and its gradient.

        Args:
            cost_func: Objective function f(x) → float
            cost_grad: Objective gradient ∇f(x) → np.ndarray
        """
        self.cost_func = cost_func
        self.cost_grad = cost_grad

    def set_eq_cons(self, eq_cons_func: Callable[[np.ndarray], np.ndarray],
                    eq_cons_grad: Callable[[np.ndarray], np.ndarray]) -> None:
        """Sets the equality constraint function and its gradient.

        Args:
            eq_cons_func: Equality constraints h(x) = 0 → np.ndarray
            eq_cons_grad: Equality constraint gradient ∇h(x) → np.ndarray
        """
        if self.dim_eq_cons == 0:
            raise RuntimeError(
                "dim_eq_cons is 0 - cannot set equality constraints")
        self.eq_cons_func = eq_cons_func
        self.eq_cons_grad = eq_cons_grad

    def set_ieq_cons(self, ieq_cons_func: Callable[[np.ndarray], np.ndarray],
                     ieq_cons_grad: Callable[[np.ndarray], np.ndarray]) -> None:
        """Sets the inequality constraint function and its gradient.

        Args:
            ieq_cons_func: Inequality constraints g(x) ≤ 0 → np.ndarray
            ieq_cons_grad: Inequality constraint gradient ∇g(x) → np.ndarray
        """
        if self.dim_ieq_cons == 0:
            raise RuntimeError(
                "dim_ieq_cons is 0 - cannot set inequality constraints")
        self.ieq_cons_func = ieq_cons_func
        self.ieq_cons_grad = ieq_cons_grad

    def _validate_functions(self) -> None:
        """Validates that all required functions are set."""
        if self.cost_func is None or self.cost_grad is None:
            raise RuntimeError("Cost function and gradient not set")
        if self.dim_eq_cons > 0 and (self.eq_cons_func is None or self.eq_cons_grad is None):
            raise RuntimeError(
                "Equality constraint function and gradient not set")
        if self.dim_ieq_cons > 0 and (self.ieq_cons_func is None or self.ieq_cons_grad is None):
            raise RuntimeError(
                "Inequality constraint function and gradient not set")

    def _augmented_lagrangian(self, x: np.ndarray) -> float:
        """Computes the augmented Lagrangian function.

        Args:
            x: Current point in parameter space

        Returns:
            Value of the augmented Lagrangian at x
        """
        f_val = np.asarray(self.cost_func(x))
        aug_val = f_val

        # Add equality constraint terms
        if self.dim_eq_cons > 0:
            c_eq = np.asarray(self.eq_cons_func(x))
            scaled_eq = c_eq + self.lambda_eq / self.mu
            aug_val += (self.mu / 2) * np.sum(scaled_eq**2)

        # Add inequality constraint terms
        if self.dim_ieq_cons > 0:
            c_ieq = np.asarray(self.ieq_cons_func(x))
            scaled_ieq = c_ieq + self.lambda_ieq / self.mu
            phi = np.maximum(0, scaled_ieq)  # Projection operator
            aug_val += (self.mu / 2) * np.sum(phi**2)

        # print(f"is aug_val ndarray: {isinstance(aug_val, np.ndarray)}")

        return aug_val

    def _augmented_lagrangian_grad(self, x: np.ndarray) -> np.ndarray:
        """Computes the gradient of the augmented Lagrangian function.

        Args:
            x: Current point in parameter space

        Returns:
            Gradient of the augmented Lagrangian at x
        """
        grad = np.asarray(self.cost_grad(x)).flatten()

        # Add equality constraint gradients
        if self.dim_eq_cons > 0:
            c_eq = np.asarray(self.eq_cons_func(x)).flatten()
            grad_eq = np.asarray(self.eq_cons_grad(x))
            scaled_eq = c_eq + self.lambda_eq / self.mu
            # print(f"eq:{c_eq.shape, grad_eq.shape, scaled_eq.shape, self.lambda_eq.shape}")
            grad += self.mu * np.dot(scaled_eq, grad_eq)

        # Add inequality constraint gradients
        if self.dim_ieq_cons > 0:
            c_ieq = np.asarray(self.ieq_cons_func(x)).flatten()
            grad_ieq = np.asarray(self.ieq_cons_grad(x))
            scaled_ieq = c_ieq + self.lambda_ieq / self.mu
            phi = np.maximum(0, scaled_ieq)  # Projection operator
            # print(f"ieq:{c_ieq.shape, grad_ieq.shape, phi.shape, self.lambda_ieq.shape}")
            grad += self.mu * np.dot(phi, grad_ieq)

        # print(f"grad shape: {grad.shape}")

        # print(f"is aug_grad ndarray: {isinstance(grad, np.ndarray)}")
        return grad

    def _solve_inner_problem(self, x0: np.ndarray) -> np.ndarray:
        """Solves the inner unconstrained optimization problem.

        Args:
            x0: Initial point for the inner problem

        Returns:
            Optimized solution for the inner problem
        """
        # Use L-BFGS to solve the inner unconstrained problem
        result = practical_lbfgs_optimizer(
            x_init=x0,
            cost_func=self._augmented_lagrangian,
            grad_func=self._augmented_lagrangian_grad,
            max_iterations=self.inner_max_iter,
            verbose=False
        )

        # Record inner iteration count
        inner_iters = len(result["costs"]) - 1
        self.inner_iter_history.append(inner_iters)

        return result["x_opt"]
        # result = minimize(
        #     self._augmented_lagrangian,
        #     x0,
        #     method='L-BFGS-B',
        #     jac=self._augmented_lagrangian_grad,
        #     options={'disp': False}
        # )
        # return result.x

    def _compute_constraint_violation(self, x: np.ndarray) -> float:
        """Computes the total constraint violation.

        Args:
            x: Current point in parameter space

        Returns:
            Total constraint violation norm
        """
        violation = 0.0

        # Equality constraint violation
        if self.dim_eq_cons > 0:
            c_eq = self.eq_cons_func(x)
            violation += np.linalg.norm(c_eq, 2)

        # Inequality constraint violation
        if self.dim_ieq_cons > 0:
            c_ieq = self.ieq_cons_func(x)
            violation += np.linalg.norm(np.maximum(0, c_ieq), 2)

        return violation

    def solve(
        self,
        x0: np.ndarray,
        tol: float = 1e-6,
        verbose: bool = False
    ) -> Tuple[np.ndarray, dict]:
        """Solves the constrained optimization problem using PHR-ALM.

        Args:
            x0: Initial point in parameter space
            tol: Tolerance for constraint violation
            verbose: Whether to print progress information

        Returns:
            Tuple containing:
                x_opt: Optimized solution
                info: Dictionary with solver information
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

        # Outer loop of ALM
        for outer_iter in range(self.outer_max_iter):
            # Solve inner unconstrained problem
            x = self._solve_inner_problem(x)

            # Compute constraint violation
            cons_violation = self._compute_constraint_violation(x)
            self.cons_violation_history.append(cons_violation)
            self.mu_history.append(self.mu)
            self.outer_iter_history.append(outer_iter)

            # Check convergence
            if cons_violation < tol:
                convergence_status = "converged"
                if verbose:
                    print(f"Converged at outer iteration {outer_iter}")
                break

            # Update multipliers
            if self.dim_eq_cons > 0:
                c_eq = np.asarray(self.eq_cons_func(x)).flatten()
                self.lambda_eq += self.mu * c_eq

            if self.dim_ieq_cons > 0:
                c_ieq = np.asarray(self.ieq_cons_func(x)).flatten()
                self.lambda_ieq = np.maximum(
                    0, self.lambda_ieq + self.mu * c_ieq)

            # Update penalty parameter
            self.mu *= self.rho

            if verbose:
                print(
                    f"Outer iter {outer_iter}: "
                    f"constraint violation = {cons_violation:.4e}, "
                    f"μ = {self.mu:.1e}"
                )
        else:
            if verbose:
                print("Reached maximum outer iterations")

        # Prepare solver info
        solver_info = {
            "status": convergence_status,
            "opti_vars": x,
            "outer_iterations": len(self.outer_iter_history),
            "inner_iterations": sum(self.inner_iter_history),
            "final_constraint_violation": cons_violation,
            "final_penalty": self.mu,
            "constraint_violation_history": np.array(self.cons_violation_history),
            "penalty_history": np.array(self.mu_history),
            "inner_iter_counts": np.array(self.inner_iter_history)
        }

        return solver_info

    def eval_objective(self, x: np.ndarray) -> float:
        """Evaluates the objective function at x.

        Args:
            x: Point in parameter space

        Returns:
            Objective function value
        """
        if self.cost_func is None:
            raise RuntimeError("Objective function not set")
        return self.cost_func(x)

    def eval_gradient(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the objective gradient at x.

        Args:
            x: Point in parameter space

        Returns:
            Objective gradient
        """
        if self.cost_grad is None:
            raise RuntimeError("Objective gradient not set")
        return self.cost_grad(x)

    def reset(self) -> None:
        """Resets the solver state (multipliers and penalty parameter)."""
        self.lambda_eq = np.zeros(self.dim_eq_cons)
        self.lambda_ieq = np.zeros(self.dim_ieq_cons)
        self.mu = self.beta
        self.outer_iter_history = []
        self.inner_iter_history = []
        self.cons_violation_history = []
        self.mu_history = []


if __name__ == "__main__":
    # 定义目标函数和约束

    def cost(x): return x[0]**2 + x[1]**2
    def cost_grad(x): return np.array([2*x[0], 2*x[1]])

    def eq_cons(x): return np.array([x[0] + x[1] - 1])
    def eq_cons_grad(x): return np.array([[1], [1]])

    def ieq_cons(x): return np.array([x[0] - x[1] - 0.5])
    def ieq_cons_grad(x): return np.array([[1], [-1]])

    # 创建问题实例
    solver = PHR_ALM_Solver(
        dim_opti_vars=2,
        dim_eq_cons=1,
        dim_ieq_cons=1,
        rho=1.5,
        gamma=0.01,
        beta=10.0
    )

    # 设置函数指针
    solver.set_cost(cost, cost_grad)
    solver.set_eq_cons(eq_cons, eq_cons_grad)
    solver.set_ieq_cons(ieq_cons, ieq_cons_grad)

    # 求解问题
    x0 = np.array([1.0, 1.0])
    solution = solver.solve(x0, tol=1e-6)

    print(f"Optimal solution: {solution['opti_vars']}")
    print(f"Objective value: {solver.eval_objective(solution['opti_vars'])}")
    print(f"Equality constraint: {eq_cons(solution['opti_vars'])}")
    print(f"Inequality constraint: {ieq_cons(solution['opti_vars'])}")

import numpy as np
from typing import Union, List, Dict, Callable
from .line_search import armijo_line_search

def _validate_optimizer_params(
    beta: float,
    c: float,
    alpha_init: float,
    max_iter: int,
    tol: float,
    lambda_init: float | None = None,
    max_regularizations: float | None = None,
) -> None:
    """Validates common optimization parameters."""
    if not (0 < beta < 1):
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    if not (0 < c < 1):
        raise ValueError(f"c must be in (0, 1), got {c}")
    if alpha_init <= 0:
        raise ValueError(f"alpha_init must be positive, got {alpha_init}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}")
    if lambda_init is not None and lambda_init <= 0:
        raise ValueError(f"lambda_init must be positive, got {lambda_init}")
    if max_regularizations is not None and max_regularizations <= 0:
        raise ValueError(
            f"max_regularizations must be positive, got {max_regularizations}"
        )
    

def damped_newton_optimizer(
    x_init: Union[float, List[float], np.ndarray],
    cost_func: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    hess_func: Callable[[np.ndarray], np.ndarray],
    max_iter: int,
    tol: float,
    alpha_init: float = 1.0,
    beta: float = 0.5,
    c: float = 1e-4,
    lambda_init: float = 1e-3,
    max_regularizations: int = 50,
    verbose: bool = False,
) -> Dict[str, Union[str, np.ndarray, List[np.ndarray], List[float]]]:
    """Modified Newton's method with Hessian regularization and Armijo line search.

    Args:
        x_init: Initial parameter vector
        cost_func: Objective function to minimize
        grad_func: Gradient function
        hess_func: Hessian function
        max_iter: Maximum number of iterations
        tol: Convergence tolerance (gradient norm)
        alpha_init: Initial step size for line search
        beta: Backtracking factor for line search
        c: Armijo condition constant
        lambda_init: Initial regularization strength
        max_regularizations: Maximum Hessian regularization attempts per iteration
        verbose: Whether to print progress information

    Returns:
        Dictionary containing:
            result: Optimization outcome ("converged", "max_iterations_reached",
                   "line_search_failed", or "regularization_failed")
            x_opt: Optimized parameters
            trajectory: List of visited parameters
            costs: Cost values at each iteration
            error_message: Error details if optimization fails (optional)
    """
    # Validate parameters
    _validate_optimizer_params(
        beta, c, alpha_init, max_iter, tol, lambda_init, max_regularizations
    )

    # Initialize variables
    x_k = np.atleast_1d(x_init).astype(float)
    trajectory = [x_k.copy()]
    costs = [cost_func(x_k)]
    grad_norm = np.inf
    result_status = "max_iterations_reached"
    error_message = None

    # Main optimization loop
    for iteration in range(1, max_iter + 1):
        g_k = np.asarray(grad_func(x_k), dtype=float).flatten()
        grad_norm = np.linalg.norm(g_k)

        if verbose:
            print(
                f"Iteration {iteration:3d} | Cost = {costs[-1]:.6e} | "
                f"Grad Norm = {grad_norm:.3e}"
            )

        # Check convergence
        if grad_norm <= tol:
            result_status = "converged"
            if verbose:
                print(f"Convergence achieved at iteration {iteration}")
            break

        # Compute Hessian and regularize if needed
        H_k = np.asarray(hess_func(x_k), dtype=float)
        lambda_reg = lambda_init
        regularization_success = False

        for reg_count in range(max_regularizations):
            try:
                # Regularize Hessian
                H_mod = H_k + lambda_reg * np.eye(x_k.size)

                # Solve Newton system
                d_k = -np.linalg.solve(H_mod, g_k)

                # Check descent direction
                if np.dot(g_k, d_k) < 0:
                    regularization_success = True
                    break

                # Not descent direction, increase regularization
                lambda_reg *= 10
                if verbose:
                    print(f"  Hessian regularized: lambda = {lambda_reg:.1e}")

            except np.linalg.LinAlgError:
                lambda_reg *= 10
                if verbose:
                    print(f"  Hessian regularization: lambda = {lambda_reg:.1e}")

        # Check if regularization succeeded
        if not regularization_success:
            result_status = "regularization_failed"
            error_message = (
                f"Hessian regularization failed. Lambda reached {lambda_reg:.1e} "
                f"after {max_regularizations} attempts"
            )
            if verbose:
                print(f"Regularization failed at iteration {iteration}")
            break

        # Perform line search
        try:
            alpha = armijo_line_search(
                x_k, g_k, d_k, cost_func, alpha_init, beta, c, verbose=verbose
            )
        except (ValueError, RuntimeError) as e:
            result_status = "line_search_failed"
            error_message = str(e)
            if verbose:
                print(f"Line search failed at iteration {iteration}: {error_message}")
            break

        # Update parameters
        x_k = x_k + alpha * d_k
        trajectory.append(x_k.copy())
        costs.append(cost_func(x_k))  # Track current cost

    # Prepare results
    result = {
        "result": result_status,
        "x_opt": x_k,
        "trajectory": trajectory,
        "costs": costs,
    }
    if error_message:
        result["error_message"] = error_message

    return result

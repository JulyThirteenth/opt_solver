import numpy as np
from typing import Union, List, Dict, Callable
from .line_search import armijo_line_search
from .utils import _plot_convergence

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


def gradient_descent_optimizer(
    x_init: Union[float, List[float], np.ndarray],
    cost_func: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    max_iter: int,
    alpha_init: float,
    beta: float,
    c: float,
    tol: float,
    verbose: bool = False,
    plot_convergence: bool = False,
) -> Dict[str, Union[str, np.ndarray, List[np.ndarray], List[float]]]:
    """Performs gradient descent optimization with Armijo line search.

    Args:
        x_init: Initial parameter vector
        cost_func: Objective function to minimize
        grad_func: Gradient function
        max_iter: Maximum number of iterations
        alpha_init: Initial step size for line search
        beta: Backtracking factor for line search
        c: Armijo condition constant
        tol: Convergence tolerance (gradient norm)
        verbose: Whether to print progress information
        plot_convergence: Whether to plot cost vs iterations

    Returns:
        Dictionary containing:
            result: Optimization outcome ("converged", "max_iterations_reached",
                   or "line_search_failed")
            x_opt: Optimized parameters
            trajectory: List of visited parameters
            costs: Cost values at each iteration
            error_message: Error details if line search fails (optional)
    """
    # Validate parameters
    _validate_optimizer_params(beta, c, alpha_init, max_iter, tol)

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

        # Compute descent direction
        d_k = -g_k

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

    # Plot convergence if requested
    if plot_convergence:
        _plot_convergence(costs, "Gradient Descent")

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

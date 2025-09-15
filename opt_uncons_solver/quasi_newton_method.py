import numpy as np
from copy import copy
from typing import Union, List, Dict, Callable
from .line_search import lewis_overton_line_search
from .utils import _plot_convergence


def _validate_lbfgs_params(
    memo_size: int,
    g_epsilon: float,
    past_time: int,
    f_epsilon: float,
    max_iterations: int,
    max_linesearch: int,
    min_alpha: float,
    max_alpha: float,
    c1: float,
    c2: float,
    cautious_factor: float,
    machine_prec: float,
) -> None:
    """Validates L-BFGS optimization parameters."""
    if memo_size <= 0:
        raise ValueError(f"memo_size must be positive, got {memo_size}")
    if g_epsilon < 0.0:
        raise ValueError(f"g_epsilon must be non-negative, got {g_epsilon}")
    if past_time < 0:
        raise ValueError(f"past_time must be non-negative, got {past_time}")
    if f_epsilon < 0.0:
        raise ValueError(f"f_epsilon must be non-negative, got {f_epsilon}")
    if max_iterations < 0:
        raise ValueError(f"max_iterations must be non-negative, got {max_iterations}")
    if max_linesearch <= 0:
        raise ValueError(f"max_linesearch must be positive, got {max_linesearch}")
    if min_alpha <= 0.0:
        raise ValueError(f"min_alpha must be positive, got {min_alpha}")
    if max_alpha <= min_alpha:
        raise ValueError(
            f"max_alpha ({max_alpha}) must be greater than min_alpha ({min_alpha})"
        )
    if not (0 < c1 < 1):
        raise ValueError(f"c1 must be in (0, 1), got {c1}")
    if not (c1 <= c2 < 1):
        raise ValueError(f"c2 must be in [{c1}, 1), got {c2}")
    if cautious_factor <= 0.0:
        raise ValueError(f"cautious_factor must be positive, got {cautious_factor}")
    if machine_prec <= 0.0:
        raise ValueError(f"machine_prec must be positive, got {machine_prec}")


def practical_lbfgs_optimizer(
    x_init: Union[float, List[float], np.ndarray],
    cost_func: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    memo_size: int = 8,
    g_epsilon: float = 1e-5,
    past_time: int = 3,
    f_epsilon: float = 1e-6,
    max_iterations: int = 1000,
    max_linesearch: int = 64,
    min_alpha: float = 1e-20,
    max_alpha: float = 1e20,
    c1: float = 1e-4,
    c2: float = 0.9,
    cautious_factor: float = 1e-6,
    machine_prec: float = 1e-16,
    verbose: bool = False,
    plot_convergence: bool = False,
) -> Dict[str, Union[str, np.ndarray, List[np.ndarray], List[float]]]:
    """Limited-memory BFGS optimization with Lewis-Overton line search.

    Args:
        x_init: Initial parameter vector
        cost_func: Objective function to minimize
        grad_func: Gradient function
        memo_size: Number of previous steps to store for Hessian approximation
        g_epsilon: Gradient norm convergence tolerance
        past_time: Number of past iterations for function value stopping criterion
        f_epsilon: Relative function value change stopping criterion
        max_iterations: Maximum number of optimization iterations
        max_linesearch: Maximum line search iterations per step
        min_alpha: Minimum step size for line search
        max_alpha: Maximum step size for line search
        c1: Armijo condition constant for line search
        c2: Curvature condition constant for line search
        cautious_factor: Cautious update factor
        machine_prec: Machine precision for numerical stability
        verbose: Whether to print progress information
        plot_convergence: Whether to plot cost vs iterations

    Returns:
        Dictionary containing:
            result: Optimization outcome ("converged", "max_iterations_reached",
                   "line_search_failed", "func_tolerance_reached", or "stationary_point")
            x_opt: Optimized parameters
            trajectory: List of visited parameters
            costs: Cost values at each iteration
    """
    # Validate parameters
    _validate_lbfgs_params(
        memo_size,
        g_epsilon,
        past_time,
        f_epsilon,
        max_iterations,
        max_linesearch,
        min_alpha,
        max_alpha,
        c1,
        c2,
        cautious_factor,
        machine_prec,
    )

    result_status = "stationary_point"

    x = np.atleast_1d(x_init).astype(float)
    f = cost_func(x)
    g = np.asarray(grad_func(x), dtype=float).flatten()
    x_prev = np.zeros_like(x)
    g_prev = np.zeros_like(g)
    # Record xs and costs
    trajectory = [x.copy()]
    costs = [f]
    # Store the initial value of the cost function for stop criterion.
    f_past = np.zeros(max(1, past_time))
    f_past[0] = f
    # Compute the direction, we assume the initial hessian matrix H_0 as the identity matrix.
    d = -g
    iteration_count = 1
    # Make sure that the initial variables are not a stationary point.
    x_norm_inf = np.linalg.norm(x, ord=np.inf)
    g_norm_inf = np.linalg.norm(g, ord=np.inf)
    if g_norm_inf / max(1.0, x_norm_inf) > g_epsilon:
        alpha = 1.0 / np.linalg.norm(d).item()
        bound = 0
        end = 0
        n, m = x.shape[0], memo_size
        lm_alpha = np.zeros(m)
        lm_s = np.zeros((n, m))
        lm_y = np.zeros((n, m))
        lm_ys = np.zeros(m)
        while True:
            # Store the current position and gradient vectors.
            x_prev = copy(x)
            g_prev = copy(g)
            alpha = alpha if alpha < max_alpha else 0.5 * max_alpha
            try:
                # Perform line search
                alpha = lewis_overton_line_search(
                    x_prev,
                    g_prev,
                    d,
                    cost_func,
                    grad_func,
                    alpha,
                    min_alpha,
                    max_alpha,
                    c1,
                    c2,
                    max_linesearch,
                    machine_prec,
                    verbose,
                )
            except (ValueError, RuntimeError) as e:
                result_status = "line_search_failed"
                if verbose:
                    print(
                        f"Line search failed at iteration {iteration_count}: {str(e)}"
                    )
                break
            x = x_prev + alpha * d
            f = cost_func(x)
            g = np.asarray(grad_func(x), dtype=float).flatten()
            # Record xs and costs
            trajectory.append(copy(x))
            costs.append(f)
            # Convergence test.
            # The criterion is given by the following formula:
            #   ||g(x)||_inf / max(1, ||x||_inf) < g_epsilon
            x_norm_inf = np.linalg.norm(x, ord=np.inf)
            g_norm_inf = np.linalg.norm(g, ord=np.inf)
            if g_norm_inf / max(1.0, x_norm_inf) <= g_epsilon:
                result_status = "converged"
                break
            # Test for stopping criterion.
            # The criterion is given by the following formula:
            #   |f(past_x) - f(x)| / max(1, |f(x)|) < f_epsilon.
            if 0 < past_time:
                # We don't test the stopping criterion while k < past.
                if past_time <= iteration_count:
                    if (
                        abs(f_past[iteration_count % past_time] - f) / max(1.0, abs(f))
                        < f_epsilon
                    ):
                        result_status = "func_tolerance_reached"
                        break
                f_past[iteration_count % past_time] = f

            # Check if reach the max_iterations
            if max_iterations != 0 and max_iterations <= iteration_count:
                result_status = "max_iterations_reached"
                break

            iteration_count += 1

            # Update vectors s and y:
            #   s_{k+1} = x_{k+1} - x_{k} = \alpha * d_{k}.
            #   y_{k+1} = g_{k+1} - g_{k}.
            lm_s[:, end] = x - x_prev
            lm_y[:, end] = g - g_prev
            # Compute scalars ys and yy:
            #   ys = y^t \cdot s = 1 / \rho.
            #   yy = y^t \cdot y.
            # Notice that yy is used for scaling the hessian matrix H_0 (Cholesky factor).
            ys = np.dot(lm_y[:, end], lm_s[:, end])
            yy = np.dot(lm_y[:, end], lm_y[:, end])
            lm_ys[end] = ys

            # Compute the negative of gradients.
            d = -g

            # Only cautious update is performed here as long as
            # (y^t \cdot s) / ||s_{k+1}||^2 > \epsilon * ||g_{k}||^\alpha,
            # where \epsilon is the cautious factor and a proposed value
            # for \alpha is 1.
            # This is not for enforcing the PD of the approxomated Hessian
            # since ys > 0 is already ensured by the weak Wolfe condition.
            # This is to ensure the global convergence as described in:
            # Dong-Hui Li and Masao Fukushima. On the global convergence of
            # the BFGS method for nonconvex unconstrained optimization problems.
            # SIAM Journal on Optimization, Vol 11, No 4, pp. 1054-1064, 2011.
            cau = (
                cautious_factor
                * np.linalg.norm(g_prev)
                * np.dot(lm_s[:, end], lm_s[:, end])
            )
            # print(f"ys:{ys}, cau:{cau}, lm_s[:, end]:{lm_s[:, end]}")
            if ys > cau:
                # Recursive formula to compute dir = -(H \cdot g).
                # This is described in page 779 of:
                # Jorge Nocedal.
                # Updating Quasi-Newton Matrices with Limited Storage.
                # Mathematics of Computation, Vol. 35, No. 151,
                # pp. 773--782, 1980.
                bound += 1
                bound = m if m < bound else bound
                end = (end + 1) % m
                idx = end
                for i in range(bound):
                    idx = (idx - 1 + m) % m
                    lm_alpha[idx] = np.dot(lm_s[:, idx], d) / lm_ys[idx]
                    d = d - lm_alpha[idx] * lm_y[:, idx]
                d = d * (ys / yy)
                for i in range(bound):
                    beta = np.dot(lm_y[:, idx], d) / lm_ys[idx]
                    d = d + (lm_alpha[idx] - beta) * lm_s[:, idx]
                    idx = (idx + 1) % m

            # The search direction d is ready. We try alpha = 1 first.
            alpha = 1.0

    if plot_convergence:
        _plot_convergence(costs, "LBFGS")

    return {
        "result": result_status,
        "x_opt": x,
        "trajectory": trajectory,
        "costs": costs,
    }

import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from typing import Callable, Union, List, Dict


def _plot_convergence(costs: List[float], method_name: str) -> None:
    """Plots convergence curve for optimization results."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(costs)), costs, marker='o', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title(f'Convergence Curve: {method_name}')
    plt.grid(True, alpha=0.3)
    plt.semilogy()
    plt.tight_layout()
    plt.show()


def armijo_line_search(
    x_k: np.ndarray,
    g_k: np.ndarray,
    s_k: np.ndarray,
    cost_func: Callable[[np.ndarray], float],
    alpha: float,
    beta: float,
    c: float,
    min_alpha: float = 1e-8,
    max_backtracks: int = 100,
    verbose: bool = False
) -> float:
    """Performs Armijo line search to determine step size.

    Args:
        x_k: Current point in parameter space
        g_k: Gradient at current point
        s_k: Search direction
        cost_func: Objective function to minimize
        alpha: Initial step size
        beta: Backtracking factor (must be in (0, 1))
        c: Armijo condition constant (must be in (0, 1))
        min_alpha: Minimum acceptable step size
        max_backtracks: Maximum number of backtracking iterations
        verbose: Whether to print debugging information

    Returns:
        Step size satisfying Armijo condition

    Raises:
        ValueError: If parameters violate constraints or line search fails
        RuntimeError: If maximum backtracking iterations exceeded
    """
    # Validate parameters
    if not (0 < beta < 1):
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    if not (0 < c < 1):
        raise ValueError(f"c must be in (0, 1), got {c}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if min_alpha <= 0:
        raise ValueError(f"min_alpha must be positive, got {min_alpha}")
    if max_backtracks <= 0:
        raise ValueError(
            f"max_backtracks must be positive, got {max_backtracks}")

    f_init = cost_func(x_k)
    dg_init = np.dot(g_k, s_k)

    # Check sufficient descent condition
    if dg_init >= 0:
        raise ValueError(
            f"Search direction s_k not a descent direction (g·s = {dg_init:.3e})"
        )

    # Armijo condition loop
    for count in range(max_backtracks + 1):
        x_proposed = x_k + alpha * s_k
        f_proposed = cost_func(x_proposed)
        armijo_bound = f_init + c * alpha * dg_init

        if f_proposed <= armijo_bound:
            return alpha

        alpha *= beta
        if verbose:
            print(f"  Line search backtrack {count}: alpha = {alpha:.5e}")

        if alpha < min_alpha:
            raise RuntimeError(
                f"Line search failed: step size reduced below minimum threshold "
                f"({min_alpha:.1e}) after {count} backtracks"
            )

    raise RuntimeError(
        f"Line search failed: maximum backtracks ({max_backtracks}) exceeded"
    )


def _validate_optimizer_params(
    beta: float,
    c: float,
    alpha_init: float,
    max_iter: int,
    tol: float,
    lambda_init: float = None,
    max_regularizations: float = None
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
    plot_convergence: bool = False
) -> Dict[str, Union[str, np.ndarray, List[np.ndarray]]]:
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
        g_k = grad_func(x_k)
        grad_norm = np.linalg.norm(g_k)
        costs.append(cost_func(x_k))  # Track current cost

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
                print(
                    f"Line search failed at iteration {iteration}: {error_message}")
            break

        # Update parameters
        x_k = x_k + alpha * d_k
        trajectory.append(x_k.copy())

    # Final cost calculation
    final_cost = cost_func(x_k)
    if costs[-1] != final_cost:
        costs.append(final_cost)

    # Plot convergence if requested
    if plot_convergence:
        _plot_convergence(costs, "Gradient Descent")

    # Prepare results
    result = {
        "result": result_status,
        "x_opt": x_k,
        "trajectory": trajectory,
        "costs": costs
    }
    if error_message:
        result["error_message"] = error_message

    return result


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
    plot_convergence: bool = False
) -> Dict[str, Union[str, np.ndarray, List[np.ndarray]]]:
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
        plot_convergence: Whether to plot cost vs iterations

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
        g_k = grad_func(x_k)
        grad_norm = np.linalg.norm(g_k)
        costs.append(cost_func(x_k))  # Track current cost

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
        H_k = hess_func(x_k)
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
                    print(
                        f"  Hessian regularization: lambda = {lambda_reg:.1e}")

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
                print(
                    f"Line search failed at iteration {iteration}: {error_message}")
            break

        # Update parameters
        x_k = x_k + alpha * d_k
        trajectory.append(x_k.copy())

    # Final cost calculation
    final_cost = cost_func(x_k)
    if costs[-1] != final_cost:
        costs.append(final_cost)

    # Plot convergence if requested
    if plot_convergence:
        _plot_convergence(costs, "Damped Newton")

    # Prepare results
    result = {
        "result": result_status,
        "x_opt": x_k,
        "trajectory": trajectory,
        "costs": costs
    }
    if error_message:
        result["error_message"] = error_message

    return result


def lewis_overton_line_search(
    x_k: np.ndarray,
    g_k: np.ndarray,
    s_k: np.ndarray,
    cost_func: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    alpha_init: float,
    alpha_min: float = 1e-20,
    alpha_max: float = 1e20,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 64,
    machine_prec: float = 1e-16,
    verbose: bool = False
) -> float:
    """Performs Lewis-Overton line search to satisfy weak Wolfe conditions.

    Args:
        x_k: Current point in parameter space
        g_k: Gradient at current point
        s_k: Search direction
        cost_func: Objective function to minimize
        grad_func: Gradient function
        alpha_init: Initial step size (must be positive)
        alpha_min: Minimum allowed step size
        alpha_max: Maximum allowed step size
        c1: Armijo condition constant (must be in (0, 1))
        c2: Curvature condition constant (must be in [c1, 1))
        max_iter: Maximum number of line search iterations
        machine_prec: Machine precision threshold for stopping
        verbose: Whether to print debugging information

    Returns:
        Step size satisfying weak Wolfe conditions

    Raises:
        ValueError: If parameters violate constraints or line search fails
        RuntimeError: If maximum iterations exceeded or invalid function values
    """
    # Validate input parameters
    if alpha_init <= 0:
        raise ValueError(f"alpha_init must be positive, got {alpha_init}")
    if alpha_min <= 0:
        raise ValueError(f"alpha_min must be positive, got {alpha_min}")
    if alpha_max <= alpha_min:
        raise ValueError(
            f"alpha_max ({alpha_max}) must be greater than alpha_min ({alpha_min})"
        )
    if not (0 < c1 < 1):
        raise ValueError(f"c1 must be in (0, 1), got {c1}")
    if not (c1 <= c2 < 1):
        raise ValueError(f"c2 must be in [{c1}, 1), got {c2}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    if machine_prec <= 0:
        raise ValueError(f"machine_prec must be positive, got {machine_prec}")

    # Calculate initial directional derivative and validate descent direction
    dir_deriv_init = np.dot(g_k, s_k)
    if dir_deriv_init >= 0:
        raise ValueError(
            f"Search direction not descent (g·s = {dir_deriv_init:.3e})"
        )

    # Precompute constants for Wolfe conditions
    f_init = cost_func(x_k)
    armijo_bound_coeff = c1 * dir_deriv_init
    curvature_bound = c2 * dir_deriv_init

    # Initialize search bounds
    alpha_low = 0.0
    alpha_high = alpha_max
    alpha = alpha_init
    bracketed = False

    # Line search loop
    for iter_count in range(1, max_iter + 1):
        # Evaluate trial point
        x_trial = x_k + alpha * s_k
        f_trial = cost_func(x_trial)

        # Check for invalid function values
        if not np.isfinite(f_trial):
            raise RuntimeError(
                f"Non-finite cost value {f_trial} at alpha = {alpha:.3e}"
            )

        # Calculate gradient at trial point
        g_trial = grad_func(x_trial)
        dir_deriv_trial = np.dot(g_trial, s_k)

        if verbose:
            print(
                f"[Iter {iter_count}] alpha = {alpha:.5e}, "
                f"f_trial = {f_trial:.5e}, armijo_bound = {f_init + alpha * armijo_bound_coeff:.5e}, "
                f"dir_deriv_trial = {dir_deriv_trial:.5e}, curvature_bound = {curvature_bound:.5e}"
            )

        # Check Armijo condition (sufficient decrease)
        armijo_bound = f_init + alpha * armijo_bound_coeff
        if f_trial > armijo_bound:
            # Armijo condition failed - reduce step size
            alpha_high = alpha
            bracketed = True
        else:
            # Check curvature condition (sufficient flatness)
            if dir_deriv_trial < curvature_bound:
                # Curvature condition failed - increase step size
                alpha_low = alpha
            else:
                # Both conditions satisfied
                return alpha

        # Update alpha based on bracketing status
        if bracketed:
            # Bracketed interval - bisect
            alpha_new = 0.5 * (alpha_low + alpha_high)

            # Check for insufficient progress
            interval_width = alpha_high - alpha_low
            if interval_width < machine_prec * alpha_high:
                raise RuntimeError(
                    f"Search interval width ({interval_width:.3e}) below machine precision "
                    f"threshold ({machine_prec * alpha_high:.3e})"
                )
        else:
            # Not yet bracketed - expand search
            alpha_new = 2.0 * alpha

        # Ensure alpha stays within bounds
        if alpha_new < alpha_min:
            raise RuntimeError(
                f"Step size alpha ({alpha_new:.3e}) below minimum ({alpha_min:.3e})"
            )
        if alpha_new > alpha_max:
            raise RuntimeError(
                f"Step size alpha ({alpha_new:.3e}) exceeds maximum ({alpha_max:.3e})"
            )

        alpha = alpha_new

    raise RuntimeError(
        f"Line search failed to converge after {max_iter} iterations"
    )


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
    machine_prec: float
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
        raise ValueError(
            f"max_iterations must be non-negative, got {max_iterations}")
    if max_linesearch <= 0:
        raise ValueError(
            f"max_linesearch must be positive, got {max_linesearch}")
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
        raise ValueError(
            f"cautious_factor must be positive, got {cautious_factor}")
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
    plot_convergence: bool = False
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
                   "line_search_failed", or "stationary_point")
            x_opt: Optimized parameters
            trajectory: List of visited parameters
            costs: Cost values at each iteration
    """
    # Validate parameters
    # _validate_lbfgs_params(
    #     memo_size, g_epsilon, past_time, f_epsilon, max_iterations,
    #     max_linesearch, min_alpha, max_alpha, c1, c2, cautious_factor, machine_prec
    # )

    result_status = "stationary_point"

    x = np.array(x_init, dtype=float)
    f = cost_func(x)
    g = grad_func(x)
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
        alpha = 1.0 / np.linalg.norm(d)
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
                    x_prev, g_prev, d, cost_func, grad_func, alpha,
                    min_alpha, max_alpha, c1, c2, max_linesearch, machine_prec, verbose
                )
            except (ValueError, RuntimeError) as e:
                result_status = "line_search_failed"
                if verbose:
                    print(
                        f"Line search failed at iteration {iteration_count}: {str(e)}")
                break
            x = x_prev + alpha * d
            f = cost_func(x)
            g = grad_func(x)
            # Record xs and costs
            trajectory.append(copy(x))
            costs.append(f)
            # Convergence test.
            # The criterion is given by the following formula:
            #   ||g(x)||_inf / max(1, ||x||_inf) < g_epsilon
            x_norm_inf = np.linalg.norm(x, ord=np.inf)
            g_norm_inf = np.linalg.norm(g, ord=np.inf)
            if g_norm_inf / max(1.0, x_norm_inf) <= g_epsilon:
                result_status = 'converged'
                break
            # Test for stopping criterion.
            # The criterion is given by the following formula:
            #   |f(past_x) - f(x)| / max(1, |f(x)|) < f_epsilon.
            if 0 < past_time:
                # We don't test the stopping criterion while k < past.
                if past_time <= iteration_count:
                    if abs(f_past[iteration_count % past_time] - f) / max(1.0, abs(f)) < f_epsilon:
                        result_status = 'stop'
                        break
                f_past[iteration_count % past_time] = f

            # Check if reach the max_iterations
            if max_iterations != 0 and max_iterations <= iteration_count:
                result_status = 'maximum_iteration_reached'
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
            cau = cautious_factor * \
                np.linalg.norm(g_prev) * np.dot(lm_s[:, end], lm_s[:, end])
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
        _plot_convergence(costs, 'LBFGS')

    return {
        "result": result_status,
        "x_opt": x,
        "trajectory": trajectory,
        "costs": costs
    }


if __name__ == "__main__":

    # Rosenbrock function (2D)
    def rosenbrock_func(x: np.ndarray) -> float:
        """Compute Rosenbrock function value."""
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
        """Compute gradient of Rosenbrock function."""
        grad_x = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
        grad_y = 200 * (x[1] - x[0]**2)
        return np.array([grad_x, grad_y])

    def rosenbrock_hess(x):
        h11 = 2 - 400 * (x[1] - 3 * x[0]**2)
        h12 = -400 * x[0]
        h22 = 200
        return np.array([[h11, h12],
                        [h12, h22]])

    x_init = np.array([-1.2, 1.0])  # Standard test starting point

    result = gradient_descent_optimizer(
        x_init=x_init,
        cost_func=rosenbrock_func,
        grad_func=rosenbrock_grad,
        max_iter=20000,
        alpha_init=1.0,
        beta=0.5,
        c=1e-4,
        tol=1e-6,
        verbose=False,
        plot_convergence=True
    )

    print("\nOptimization result:", result["result"])
    print("Optimal x:", result["x_opt"])
    print("Final cost:", rosenbrock_func(result["x_opt"]))

    result = damped_newton_optimizer(
        x_init=x_init,
        cost_func=rosenbrock_func,
        grad_func=rosenbrock_grad,
        hess_func=rosenbrock_hess,
        max_iter=1000,
        tol=1e-6,
        alpha_init=1.0,
        beta=0.5,
        c=1e-4,
        lambda_init=1e-3,
        verbose=False,
        plot_convergence=True
    )

    print("\nOptimization result:", result["result"])
    print("Optimal x:", result["x_opt"])
    print("Final cost:", rosenbrock_func(result["x_opt"]))

    result = practical_lbfgs_optimizer(
        x_init=x_init,
        cost_func=rosenbrock_func,
        grad_func=rosenbrock_grad,
        g_epsilon=1e-8,
        past_time=3,
        f_epsilon=1e-8,
        verbose=False,
        plot_convergence=True
    )

    print("\nOptimization result:", result["result"])
    print("Optimal x:", result["x_opt"])
    print("Final cost:", rosenbrock_func(result["x_opt"]))

import numpy as np
from typing import Callable, Optional, Tuple


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
    verbose: bool = False,
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
        raise ValueError(f"max_backtracks must be positive, got {max_backtracks}")

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


def lewis_overton_line_search(
    x_k: np.ndarray,
    g_k: np.ndarray,
    s_k: np.ndarray,
    value_and_grad: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    alpha_init: float,
    alpha_min: float = 1e-20,
    alpha_max: float = 1e20,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 64,
    machine_prec: float = 1e-16,
    verbose: bool = False,
    f_k: Optional[float] = None,
) -> Tuple[float, float, np.ndarray]:
    """Performs Lewis-Overton line search to satisfy weak Wolfe conditions.

    Args:
        x_k: Current point in parameter space
        g_k: Gradient at current point
        s_k: Search direction
        value_and_grad: Combined objective and gradient function
        alpha_init: Initial step size (must be positive)
        alpha_min: Minimum allowed step size
        alpha_max: Maximum allowed step size
        c1: Armijo condition constant (must be in (0, 1))
        c2: Curvature condition constant (must be in [c1, 1))
        max_iter: Maximum number of line search iterations
        machine_prec: Machine precision threshold for stopping
        verbose: Whether to print debugging information
        f_k: Optional objective value at x_k

    Returns:
        Tuple containing:
            alpha: Step size satisfying weak Wolfe conditions
            f_trial: Objective value at x_k + alpha * s_k
            g_trial: Gradient at x_k + alpha * s_k

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
        raise ValueError(f"Search direction not descent (g·s = {dir_deriv_init:.3e})")

    # Precompute constants for Wolfe conditions
    if f_k is None:
        f_init, _ = value_and_grad(x_k)
    else:
        f_init = f_k
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
        f_trial, g_trial = value_and_grad(x_trial)
        f_trial = float(np.asarray(f_trial))
        g_trial = np.asarray(g_trial, dtype=float).flatten()

        # Check for invalid function values
        if not np.isfinite(f_trial):
            raise RuntimeError(
                f"Non-finite cost value {f_trial} at alpha = {alpha:.3e}"
            )

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
                return alpha, f_trial, g_trial

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

    raise RuntimeError(f"Line search failed to converge after {max_iter} iterations")

from typing import Dict, List, Union
import numpy as np


def _format_linear_cgd_result(
    result_status: str,
    x: np.ndarray,
    trajectory: List[np.ndarray],
    residuals: List[float],
    error_message: str | None = None,
) -> Dict[str, Union[str, int, float, np.ndarray, List[np.ndarray], List[float]]]:
    result: Dict[
        str, Union[str, int, float, np.ndarray, List[np.ndarray], List[float]]
    ] = {
        "result": result_status,
        "x_opt": x,
        "trajectory": trajectory,
        "residuals": residuals,
        "iterations": max(0, len(residuals) - 1),
        "final_residual": residuals[-1],
    }
    if error_message:
        result["error_message"] = error_message
    return result


def linear_cgd(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    eps: float = 1e-6,
    max_iter: int | None = None,
    check_symmetric: bool = True,
    symmetry_tol: float = 1e-10,
    store_trajectory: bool = True,
    return_info: bool = True,
) -> Union[
    np.ndarray,
    Dict[str, Union[str, int, float, np.ndarray, List[np.ndarray], List[float]]],
]:
    """Solves A x = b using linear conjugate gradient.

    This routine assumes A is symmetric positive definite. By default it returns
    a dictionary matching the project optimizer result style. Set
    return_info=False to return only the solution vector.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if b.ndim != 1:
        raise ValueError("b must be a 1D vector.")
    if A.shape[0] != b.shape[0]:
        raise ValueError("A and b have incompatible shapes.")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if max_iter is not None and max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    if symmetry_tol < 0.0:
        raise ValueError(f"symmetry_tol must be non-negative, got {symmetry_tol}")

    if check_symmetric and not np.allclose(
        A, A.T, rtol=symmetry_tol, atol=symmetry_tol
    ):
        raise ValueError("A must be symmetric for conjugate gradient.")

    b = b.astype(float, copy=False)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).flatten().copy()
    if x.shape != b.shape:
        raise ValueError("x0 and b must have the same shape.")

    max_iter = A.shape[0] if max_iter is None else max_iter
    trajectory = [x.copy()] if store_trajectory else []

    r = b - A @ x
    p = r.copy()
    rs_old = float(r @ r)
    residual = float(np.sqrt(rs_old))
    residuals = [residual]

    if residual <= eps:
        if return_info:
            return _format_linear_cgd_result("converged", x, trajectory, residuals)
        return x

    for _ in range(max_iter):
        Ap = A @ p
        denominator = float(p @ Ap)
        if denominator <= 0.0 or np.isclose(denominator, 0.0):
            error_message = "CG breakdown: A may not be symmetric positive definite."
            if return_info:
                return _format_linear_cgd_result(
                    "breakdown", x, trajectory, residuals, error_message
                )
            raise np.linalg.LinAlgError(error_message)

        alpha = rs_old / denominator
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = float(r @ r)
        residual = float(np.sqrt(rs_new))
        residuals.append(residual)
        if store_trajectory:
            trajectory.append(x.copy())

        if residual <= eps:
            if return_info:
                return _format_linear_cgd_result("converged", x, trajectory, residuals)
            return x

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    if return_info:
        return _format_linear_cgd_result(
            "max_iterations_reached", x, trajectory, residuals
        )
    return x

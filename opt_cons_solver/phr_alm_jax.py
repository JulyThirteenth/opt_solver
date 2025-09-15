from typing import Callable, Optional

import jax
import jax.numpy as jnp


def build_jax_lbfgs_optimizer(
    value_and_grad: Callable,
    memo_size: int = 8,
    g_epsilon: float = 1e-5,
    max_iterations: int = 300,
    max_linesearch: int = 32,
    c1: float = 1e-4,
    c2: float = 0.9,
    min_alpha: float = 1e-12,
    cautious_factor: float = 1e-6,
):
    """Builds a JAX-native L-BFGS optimizer for an augmented callback.

    The returned function has signature:
        solve(x0, lambda_eq, lambda_ieq, mu, g_tol=None) -> (x_opt, iterations)

    value_and_grad must have signature:
        value_and_grad(x, lambda_eq, lambda_ieq, mu) -> (value, grad)
    """
    def two_loop(g, lm_s, lm_y, lm_ys, end, bound, initial_scale):
        m = lm_ys.shape[0]

        def first_loop(carry, i):
            d, alpha, idx = carry
            idx = (idx - 1 + m) % m
            use_entry = i < bound
            denom = jnp.where(jnp.abs(lm_ys[idx]) > 1e-30, lm_ys[idx], 1.0)
            alpha_i = jnp.where(use_entry, jnp.dot(lm_s[idx], d) / denom, 0.0)
            d = jnp.where(use_entry, d - alpha_i * lm_y[idx], d)
            alpha = alpha.at[idx].set(alpha_i)
            return (d, alpha, idx), None

        def second_loop(carry, i):
            d, alpha, idx = carry
            use_entry = i < bound
            denom = jnp.where(jnp.abs(lm_ys[idx]) > 1e-30, lm_ys[idx], 1.0)
            beta = jnp.where(use_entry, jnp.dot(lm_y[idx], d) / denom, 0.0)
            d = jnp.where(use_entry, d + (alpha[idx] - beta) * lm_s[idx], d)
            idx = (idx + 1) % m
            return (d, alpha, idx), None

        alpha = jnp.zeros(m, dtype=g.dtype)
        (d, alpha, idx), _ = jax.lax.scan(
            first_loop, (-g, alpha, end), jnp.arange(memo_size)
        )
        d = d * initial_scale
        (d, _, _), _ = jax.lax.scan(second_loop, (d, alpha, idx), jnp.arange(memo_size))
        return d

    def lewis_overton_line_search(x, f, g, d, lambda_eq, lambda_ieq, mu, alpha_init):
        dir_deriv = jnp.dot(g, d)
        is_descent = dir_deriv < 0.0
        armijo_bound_coeff = c1 * dir_deriv
        curvature_bound = c2 * dir_deriv

        def cond_fun(state):
            (
                i,
                _alpha,
                _alpha_low,
                _alpha_high,
                _bracketed,
                accepted,
                _f_trial,
                _g_trial,
            ) = state
            return (i < max_linesearch) & (~accepted)

        def body_fun(state):
            i, alpha, alpha_low, alpha_high, bracketed, _accepted, f_trial, g_trial = (
                state
            )
            x_trial = x + alpha * d
            value_new, grad_new = value_and_grad(x_trial, lambda_eq, lambda_ieq, mu)
            dir_deriv_new = jnp.dot(grad_new, d)
            armijo_ok = value_new <= f + alpha * armijo_bound_coeff
            curvature_ok = dir_deriv_new >= curvature_bound
            accepted_new = is_descent & armijo_ok & curvature_ok

            armijo_failed = is_descent & (~armijo_ok)
            curvature_failed = is_descent & armijo_ok & (~curvature_ok)
            alpha_low = jnp.where(curvature_failed, alpha, alpha_low)
            alpha_high = jnp.where(armijo_failed, alpha, alpha_high)
            bracketed = bracketed | armijo_failed

            alpha_bisect = 0.5 * (alpha_low + alpha_high)
            alpha_expand = 2.0 * alpha
            alpha_next = jnp.where(bracketed, alpha_bisect, alpha_expand)
            alpha_next = jnp.maximum(alpha_next, min_alpha)

            f_trial = jnp.where(accepted_new, value_new, f_trial)
            g_trial = jnp.where(accepted_new, grad_new, g_trial)
            alpha_out = jnp.where(accepted_new, alpha, alpha_next)
            return (
                i + 1,
                alpha_out,
                alpha_low,
                alpha_high,
                bracketed,
                accepted_new,
                f_trial,
                g_trial,
            )

        init_state = (
            jnp.array(0),
            alpha_init,
            jnp.array(0.0, dtype=alpha_init.dtype),
            jnp.array(1e20, dtype=alpha_init.dtype),
            jnp.array(False),
            jnp.array(False),
            f,
            g,
        )
        _, alpha_out, _, _, _, accepted, f_trial, g_trial = jax.lax.while_loop(
            cond_fun, body_fun, init_state
        )
        alpha_out = jnp.maximum(alpha_out, min_alpha)
        x_new = jnp.where(accepted, x + alpha_out * d, x)
        f_new = jnp.where(accepted, f_trial, f)
        g_new = jnp.where(accepted, g_trial, g)
        return x_new, f_new, g_new, alpha_out, accepted

    @jax.jit
    def minimize_kernel(x_init, lambda_eq, lambda_ieq, mu, g_tol):
        x = jnp.asarray(x_init)
        lambda_eq = jnp.asarray(lambda_eq)
        lambda_ieq = jnp.asarray(lambda_ieq)
        g_tol = jnp.asarray(g_tol, dtype=x.dtype)
        f, g = value_and_grad(x, lambda_eq, lambda_ieq, mu)

        n = x.shape[0]
        lm_s = jnp.zeros((memo_size, n), dtype=x.dtype)
        lm_y = jnp.zeros((memo_size, n), dtype=x.dtype)
        lm_ys = jnp.zeros(memo_size, dtype=x.dtype)
        d = -g
        alpha = 1.0 / jnp.maximum(jnp.linalg.norm(d), 1.0)
        end = jnp.array(0)
        bound = jnp.array(0)
        iterations = jnp.array(0)

        def converged(x, g):
            return (
                jnp.linalg.norm(g, ord=jnp.inf)
                / jnp.maximum(1.0, jnp.linalg.norm(x, ord=jnp.inf))
                <= g_tol
            )

        def cond_fun(state):
            (
                x,
                _f,
                g,
                _d,
                _alpha,
                _lm_s,
                _lm_y,
                _lm_ys,
                _end,
                _bound,
                iterations,
            ) = state
            return (iterations < max_iterations) & (~converged(x, g))

        def body_fun(state):
            (
                x,
                f,
                g,
                d,
                alpha,
                lm_s,
                lm_y,
                lm_ys,
                end,
                bound,
                iterations,
            ) = state
            x_prev = x
            g_prev = g
            d = jnp.where(jnp.dot(g, d) < 0.0, d, -g)
            x, f, g, step_alpha, accepted = lewis_overton_line_search(
                x, f, g, d, lambda_eq, lambda_ieq, mu, alpha
            )

            s = x - x_prev
            y = g - g_prev
            ys = jnp.dot(y, s)
            yy = jnp.dot(y, y)
            cau = cautious_factor * jnp.linalg.norm(g_prev) * jnp.dot(s, s)
            update_ok = accepted & (ys > cau) & (yy > 1e-30)

            lm_s = lm_s.at[end].set(jnp.where(update_ok, s, lm_s[end]))
            lm_y = lm_y.at[end].set(jnp.where(update_ok, y, lm_y[end]))
            lm_ys = lm_ys.at[end].set(jnp.where(update_ok, ys, lm_ys[end]))

            bound_new = jnp.where(update_ok, jnp.minimum(bound + 1, memo_size), bound)
            end_new = jnp.where(update_ok, (end + 1) % memo_size, end)
            scale = ys / jnp.maximum(yy, 1e-30)
            d_lbfgs = two_loop(g, lm_s, lm_y, lm_ys, end_new, bound_new, scale)
            d = jnp.where(update_ok, d_lbfgs, -g)
            alpha = jnp.where(accepted, 1.0, step_alpha)
            iterations = iterations + jnp.where(accepted, 1, max_iterations)

            return (
                x,
                f,
                g,
                d,
                alpha,
                lm_s,
                lm_y,
                lm_ys,
                end_new,
                bound_new,
                iterations,
            )

        state = (
            x,
            f,
            g,
            d,
            alpha,
            lm_s,
            lm_y,
            lm_ys,
            end,
            bound,
            iterations,
        )
        x, _, _, _, _, _, _, _, _, _, iterations = jax.lax.while_loop(
            cond_fun, body_fun, state
        )
        return x, iterations

    def minimize(x_init, lambda_eq, lambda_ieq, mu, g_tol=None):
        if g_tol is None:
            g_tol = g_epsilon
        return minimize_kernel(x_init, lambda_eq, lambda_ieq, mu, g_tol)

    minimize.kernel = minimize_kernel
    return minimize


def build_jax_phr_alm_solver(
    augmented_value_and_grad: Callable,
    eq_constraints: Optional[Callable] = None,
    ieq_constraints: Optional[Callable] = None,
    dim_eq_cons: int = 0,
    dim_ieq_cons: int = 0,
    rho: float = 1.5,
    tau: float = 0.25,
    beta: float = 1e3,
    outer_max_iter: int = 60,
    inner_max_iter: int = 1000,
    lbfgs_memo_size: int = 8,
    lbfgs_g_epsilon: float = 1e-5,
    lbfgs_initial_g_epsilon: Optional[float] = None,
    lbfgs_adaptive_decay_iterations: int = 10,
    lbfgs_max_linesearch: int = 32,
    lbfgs_c1: float = 1e-4,
    lbfgs_c2: float = 0.9,
    lbfgs_min_alpha: float = 1e-12,
    lbfgs_cautious_factor: float = 1e-6,
):
    """Builds a JAX-jitted PHR-ALM solver backed by the local JAX L-BFGS."""
    if dim_eq_cons < 0:
        raise ValueError(f"dim_eq_cons must be non-negative, got {dim_eq_cons}")
    if dim_ieq_cons < 0:
        raise ValueError(f"dim_ieq_cons must be non-negative, got {dim_ieq_cons}")
    if dim_eq_cons > 0 and eq_constraints is None:
        raise ValueError("eq_constraints is required when dim_eq_cons > 0")
    if dim_ieq_cons > 0 and ieq_constraints is None:
        raise ValueError("ieq_constraints is required when dim_ieq_cons > 0")
    if rho <= 1.0:
        raise ValueError(f"rho must be > 1, got {rho}")
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}")
    if outer_max_iter <= 0:
        raise ValueError(f"outer_max_iter must be positive, got {outer_max_iter}")
    if inner_max_iter <= 0:
        raise ValueError(f"inner_max_iter must be positive, got {inner_max_iter}")
    if lbfgs_g_epsilon <= 0.0:
        raise ValueError(f"lbfgs_g_epsilon must be positive, got {lbfgs_g_epsilon}")
    if lbfgs_initial_g_epsilon is None:
        lbfgs_initial_g_epsilon = lbfgs_g_epsilon
    if lbfgs_initial_g_epsilon < lbfgs_g_epsilon:
        raise ValueError(
            "lbfgs_initial_g_epsilon must be greater than or equal to "
            f"lbfgs_g_epsilon, got {lbfgs_initial_g_epsilon}"
        )
    if lbfgs_adaptive_decay_iterations <= 0:
        raise ValueError(
            "lbfgs_adaptive_decay_iterations must be positive, got "
            f"{lbfgs_adaptive_decay_iterations}"
        )

    solve_inner = build_jax_lbfgs_optimizer(
        augmented_value_and_grad,
        memo_size=lbfgs_memo_size,
        g_epsilon=lbfgs_g_epsilon,
        max_iterations=inner_max_iter,
        max_linesearch=lbfgs_max_linesearch,
        c1=lbfgs_c1,
        c2=lbfgs_c2,
        min_alpha=lbfgs_min_alpha,
        cautious_factor=lbfgs_cautious_factor,
    )

    def eval_eq(x):
        if dim_eq_cons == 0:
            return jnp.zeros((0,), dtype=x.dtype)
        return jnp.ravel(eq_constraints(x))

    def eval_ieq(x):
        if dim_ieq_cons == 0:
            return jnp.zeros((0,), dtype=x.dtype)
        return jnp.ravel(ieq_constraints(x))

    def constraint_violation(c_eq, c_ieq):
        return jnp.linalg.norm(c_eq, ord=2) + jnp.linalg.norm(
            jnp.maximum(0.0, c_ieq), ord=2
        )

    def alm_violation(c_eq, c_ieq, lambda_ieq, mu):
        return jnp.linalg.norm(c_eq, ord=2) + jnp.linalg.norm(
            jnp.maximum(c_ieq, -lambda_ieq / mu), ord=2
        )

    def inner_gradient_tolerance(outer_iter, prev_alm):
        dtype = prev_alm.dtype
        final_tol = jnp.asarray(lbfgs_g_epsilon, dtype=dtype)
        initial_tol = jnp.asarray(lbfgs_initial_g_epsilon, dtype=dtype)
        decay_progress = jnp.minimum(
            outer_iter.astype(dtype) / float(lbfgs_adaptive_decay_iterations),
            jnp.asarray(1.0, dtype=dtype),
        )
        scheduled_tol = initial_tol * (final_tol / initial_tol) ** decay_progress
        violation_tol = jnp.clip(0.05 * prev_alm, final_tol, initial_tol)
        return jnp.minimum(scheduled_tol, violation_tol)

    @jax.jit
    def solve_kernel(x0, lambda_eq0, lambda_ieq0, mu0, tol):
        x = jnp.asarray(x0)
        lambda_eq = jnp.asarray(lambda_eq0, dtype=x.dtype)
        lambda_ieq = jnp.asarray(lambda_ieq0, dtype=x.dtype)
        mu = jnp.asarray(mu0, dtype=x.dtype)
        tol = jnp.asarray(tol, dtype=x.dtype)

        c_eq0 = eval_eq(x)
        c_ieq0 = eval_ieq(x)
        prev_alm = alm_violation(c_eq0, c_ieq0, lambda_ieq, mu)

        cons_history = jnp.zeros((outer_max_iter,), dtype=x.dtype)
        alm_history = jnp.zeros((outer_max_iter,), dtype=x.dtype)
        mu_history = jnp.zeros((outer_max_iter,), dtype=x.dtype)
        inner_tol_history = jnp.zeros((outer_max_iter,), dtype=x.dtype)
        inner_iter_history = jnp.zeros((outer_max_iter,), dtype=jnp.int32)
        outer_iter = jnp.array(0, dtype=jnp.int32)
        total_inner_iter = jnp.array(0, dtype=jnp.int32)
        final_cons = constraint_violation(c_eq0, c_ieq0)
        converged = jnp.array(False)

        def cond_fun(state):
            (
                _x,
                _lambda_eq,
                _lambda_ieq,
                _mu,
                _prev_alm,
                outer_iter,
                _total_inner_iter,
                _final_cons,
                converged,
                _cons_history,
                _alm_history,
                _mu_history,
                _inner_tol_history,
                _inner_iter_history,
            ) = state
            return (outer_iter < outer_max_iter) & (~converged)

        def body_fun(state):
            (
                x,
                lambda_eq,
                lambda_ieq,
                mu,
                prev_alm,
                outer_iter,
                total_inner_iter,
                _final_cons,
                _converged,
                cons_history,
                alm_history,
                mu_history,
                inner_tol_history,
                inner_iter_history,
            ) = state

            inner_g_tol = inner_gradient_tolerance(outer_iter, prev_alm)
            x, inner_iters = solve_inner(x, lambda_eq, lambda_ieq, mu, inner_g_tol)
            c_eq = eval_eq(x)
            c_ieq = eval_ieq(x)
            cons = constraint_violation(c_eq, c_ieq)
            alm = alm_violation(c_eq, c_ieq, lambda_ieq, mu)
            converged = cons < tol

            cons_history = cons_history.at[outer_iter].set(cons)
            alm_history = alm_history.at[outer_iter].set(alm)
            mu_history = mu_history.at[outer_iter].set(mu)
            inner_tol_history = inner_tol_history.at[outer_iter].set(inner_g_tol)
            inner_iter_history = inner_iter_history.at[outer_iter].set(inner_iters)

            lambda_eq_next = lambda_eq + mu * c_eq
            lambda_ieq_next = jnp.maximum(0.0, lambda_ieq + mu * c_ieq)
            mu_next = jnp.where(alm > tau * prev_alm, mu * rho, mu)

            lambda_eq = jnp.where(converged, lambda_eq, lambda_eq_next)
            lambda_ieq = jnp.where(converged, lambda_ieq, lambda_ieq_next)
            mu = jnp.where(converged, mu, mu_next)

            return (
                x,
                lambda_eq,
                lambda_ieq,
                mu,
                alm,
                outer_iter + 1,
                total_inner_iter + inner_iters,
                cons,
                converged,
                cons_history,
                alm_history,
                mu_history,
                inner_tol_history,
                inner_iter_history,
            )

        init_state = (
            x,
            lambda_eq,
            lambda_ieq,
            mu,
            prev_alm,
            outer_iter,
            total_inner_iter,
            final_cons,
            converged,
            cons_history,
            alm_history,
            mu_history,
            inner_tol_history,
            inner_iter_history,
        )
        return jax.lax.while_loop(cond_fun, body_fun, init_state)

    def solve(x0, lambda_eq0=None, lambda_ieq0=None, mu0=None, tol=1e-6):
        x = jnp.asarray(x0)
        if lambda_eq0 is None:
            lambda_eq0 = jnp.zeros((dim_eq_cons,), dtype=x.dtype)
        if lambda_ieq0 is None:
            lambda_ieq0 = jnp.zeros((dim_ieq_cons,), dtype=x.dtype)
        if mu0 is None:
            mu0 = jnp.asarray(beta, dtype=x.dtype)

        (
            x,
            lambda_eq,
            lambda_ieq,
            mu,
            _prev_alm,
            outer_iterations,
            total_inner_iterations,
            final_constraint_violation,
            converged,
            cons_history,
            alm_history,
            mu_history,
            inner_tol_history,
            inner_iter_history,
        ) = solve_kernel(x, lambda_eq0, lambda_ieq0, mu0, tol)

        outer_iterations_int = int(outer_iterations)
        total_inner_iterations_int = int(total_inner_iterations)
        converged_bool = bool(converged)
        return {
            "status": "converged" if converged_bool else "max_outer_iterations",
            "opti_vars": x,
            "lambda_eq": lambda_eq,
            "lambda_ieq": lambda_ieq,
            "final_penalty": mu,
            "outer_iterations": outer_iterations_int,
            "inner_iterations": total_inner_iterations_int,
            "final_constraint_violation": final_constraint_violation,
            "constraint_violation_history": cons_history[:outer_iterations_int],
            "alm_violation_history": alm_history[:outer_iterations_int],
            "penalty_history": mu_history[:outer_iterations_int],
            "inner_tolerance_history": inner_tol_history[:outer_iterations_int],
            "inner_iter_counts": inner_iter_history[:outer_iterations_int],
        }

    solve.kernel = solve_kernel
    solve.solve_inner = solve_inner
    return solve

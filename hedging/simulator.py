"""Delta-hedging backtest utilities for structured products."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def run_hedging_simulation(
    realized_paths: np.ndarray,
    pricer: Callable[..., tuple[float, float]],
    maturity: float,
    risk_free_rate: float,
    transaction_cost_rate: float = 0.0,
    **pricer_kwargs: Any,
) -> dict[str, np.ndarray]:
    """Backtest a discrete delta hedge against realized spot paths.

    The routine re-prices the structured product along a realized path and
    tracks the P&L of a self-financing hedge portfolio. This is the standard
    post-trade sanity check for understanding whether the model delta is stable
    enough to hedge a path-dependent payoff.

    Args:
        realized_paths: Realized market paths with shape ``(T,)`` for a single
            path or ``(P, T)`` for multiple scenarios.
        pricer: Callable returning ``(price, delta)`` for the current state.
        maturity: Final maturity in years.
        risk_free_rate: Annualized continuously compounded cash accrual rate.
        transaction_cost_rate: Proportional transaction cost applied to each
            underlying rebalance.
        **pricer_kwargs: Extra keyword arguments forwarded to ``pricer``.

    Returns:
        A dictionary containing daily hedge P&L, cumulative P&L, and the final
        hedging-error distribution.
    """
    paths = np.asarray(realized_paths, dtype=np.float64)
    is_single_path = paths.ndim == 1
    if is_single_path:
        paths = paths.reshape(1, -1)

    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("realized_paths must be shape (T,) or (P, T) with T >= 2.")
    if maturity <= 0.0:
        raise ValueError("maturity must be strictly positive.")
    if transaction_cost_rate < 0.0:
        raise ValueError("transaction_cost_rate must be non-negative.")

    n_paths, n_steps = paths.shape
    dt = maturity / (n_steps - 1)

    daily_pnl = np.zeros((n_paths, n_steps - 1), dtype=np.float64)
    hedging_error = np.zeros(n_paths, dtype=np.float64)

    for i in range(n_paths):
        path = paths[i]
        s0 = path[0]

        price_t, delta_t = pricer(
            spot=s0,
            time_index=0,
            time_to_maturity=maturity,
            path_history=path[:1],
            **pricer_kwargs,
        )

        # We start from the model value and immediately fund the underlying hedge
        # out of that premium so the portfolio stays self-financing.
        cash_t = float(price_t) - float(delta_t) * s0
        init_tc = transaction_cost_rate * abs(float(delta_t)) * s0
        cash_t -= init_tc

        for t in range(n_steps - 1):
            s_t = path[t]
            s_next = path[t + 1]
            tau_next = maturity - (t + 1) * dt

            price_next, delta_next = pricer(
                spot=s_next,
                time_index=t + 1,
                time_to_maturity=max(0.0, tau_next),
                path_history=path[: t + 2],
                **pricer_kwargs,
            )

            d_s = s_next - s_t
            option_change = float(price_next) - float(price_t)
            tc_rebalance = transaction_cost_rate * abs(float(delta_next) - float(delta_t)) * s_next

            pnl_t = float(delta_t) * d_s + cash_t * risk_free_rate * dt - option_change - tc_rebalance
            daily_pnl[i, t] = pnl_t

            cash_t = cash_t * (1.0 + risk_free_rate * dt) - (float(delta_next) - float(delta_t)) * s_next - tc_rebalance
            delta_t = float(delta_next)
            price_t = float(price_next)

        hedging_error[i] = np.sum(daily_pnl[i])

    cumulative_pnl = np.cumsum(daily_pnl, axis=1)

    if is_single_path:
        return {
            "daily_pnl": daily_pnl[0],
            "cumulative_pnl": cumulative_pnl[0],
            "hedging_error_distribution": hedging_error,
        }

    return {
        "daily_pnl": daily_pnl,
        "cumulative_pnl": cumulative_pnl,
        "hedging_error_distribution": hedging_error,
    }

"""Vectorized payoff engine for cliquet structured products."""

from __future__ import annotations

import numpy as np


class CliquetProduct:
    """Represent common cliquet payoff profiles in a single class.

    A cliquet typically resets periodically and accumulates local performance.
    In practice, the same Monte Carlo plumbing can support several payoff
    conventions, so a single parameterized class is more maintainable than
    multiple tiny payoff subclasses.
    """

    def __init__(
        self,
        payoff_type: str = "capped_coupons",
        participation: float = 1.0,
        cap: float = 0.02,
        maturity: float = 1.0,
        risk_free_rate: float = 0.0,
        notional: float = 100.0,
    ) -> None:
        """Store the payoff convention for the cliquet.

        Args:
            payoff_type: Either ``"capped_coupons"`` or ``"max_return"``.
            participation: Participation applied to positive local returns for
                the capped-coupon variant.
            cap: Local coupon cap for the capped-coupon variant.
            maturity: Final maturity in years.
            risk_free_rate: Continuously compounded discount rate.
            notional: Product notional.
        """
        valid_types = {"capped_coupons", "max_return"}
        if payoff_type not in valid_types:
            raise ValueError(f"payoff_type must be one of {sorted(valid_types)}.")

        self.payoff_type = payoff_type
        self.participation = float(participation)
        self.cap = float(cap)
        self.maturity = float(maturity)
        self.risk_free_rate = float(risk_free_rate)
        self.notional = float(notional)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Compute discounted cliquet payoffs path by path.

        Args:
            paths: Simulated spot matrix of shape ``(M_sims, N_obs)``, including
                the initial spot in the first column.

        Returns:
            Present value payoff for each simulated path.
        """
        paths = np.asarray(paths, dtype=np.float64)
        if paths.ndim != 2:
            raise ValueError("paths must be a 2D array with shape (M_sims, N_obs).")
        if paths.shape[1] < 2:
            raise ValueError("paths must contain at least 2 time points.")
        if np.any(paths[:, 0] <= 0.0):
            raise ValueError("Initial spot values must be strictly positive.")

        if self.payoff_type == "capped_coupons":
            local_returns = paths[:, 1:] / paths[:, :-1] - 1.0
            # Vectorized clipping is materially faster than iterating over each
            # observation date when the Monte Carlo sample gets large.
            positive_returns = np.maximum(local_returns, 0.0)
            coupons = np.minimum(self.participation * positive_returns, self.cap)
            gross_return = np.sum(coupons, axis=1)
            payoff = self.notional * gross_return
        else:
            best_performance = np.max(paths / paths[:, [0]], axis=1) - 1.0
            payoff = self.notional * best_performance

        discount = np.exp(-self.risk_free_rate * self.maturity)
        return payoff * discount

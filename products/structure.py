"""Legacy structured product definitions kept for backward compatibility."""

from __future__ import annotations

import numpy as np


class PhoenixAutocall:
    """Legacy Phoenix autocall payoff implementation.

    This class predates the unified ``AutocallProduct`` engine but is still
    useful as a simple reference implementation. It prices a memory-coupon
    Phoenix note with early autocall redemption and a capital-at-risk maturity
    payoff.
    """

    def __init__(
        self,
        autocall_barrier: float,
        coupon_barrier: float,
        protection_barrier: float,
        coupon_rate: float,
        observation_indices: np.ndarray,
        maturity: float,
        risk_free_rate: float = 0.0,
        notional: float = 100.0,
    ) -> None:
        """Store the contractual terms of the Phoenix autocall.

        Args:
            autocall_barrier: Early redemption barrier checked on each
                observation date.
            coupon_barrier: Barrier above which the coupon is paid.
            protection_barrier: Final capital protection threshold checked at
                maturity.
            coupon_rate: Coupon paid per observation, expressed as a fraction of
                notional.
            observation_indices: Observation dates expressed as path indices.
            maturity: Final maturity in years.
            risk_free_rate: Continuously compounded discount rate.
            notional: Product notional.
        """
        self.autocall_barrier = float(autocall_barrier)
        self.coupon_barrier = float(coupon_barrier)
        self.protection_barrier = float(protection_barrier)
        self.coupon_rate = float(coupon_rate)
        self.maturity = float(maturity)
        self.risk_free_rate = float(risk_free_rate)
        self.notional = float(notional)

        obs = np.asarray(observation_indices, dtype=np.int64)
        if obs.ndim != 1 or obs.size == 0:
            raise ValueError("observation_indices must be a non-empty 1D array.")
        self.observation_indices = np.unique(obs)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Compute discounted payoffs for a batch of simulated paths.

        Args:
            paths: Simulated spot matrix of shape ``(M, N)``.

        Returns:
            Present value payoff for each path.
        """
        paths = np.asarray(paths, dtype=np.float64)
        if paths.ndim != 2:
            raise ValueError("paths must be a 2D array of shape (M, N).")

        n_paths, n_steps = paths.shape
        if n_steps < 2:
            raise ValueError("paths must contain at least 2 time points.")
        if np.any(self.observation_indices <= 0) or np.any(self.observation_indices >= n_steps):
            raise ValueError("observation_indices must be within [1, N-1].")

        dt = self.maturity / (n_steps - 1)
        s_initial = paths[:, 0]
        if np.any(s_initial <= 0.0):
            raise ValueError("Initial spot values must be positive.")

        discounted_payoff = np.zeros(n_paths, dtype=np.float64)
        alive = np.ones(n_paths, dtype=np.bool_)
        unpaid_coupon_count = np.zeros(n_paths, dtype=np.int64)

        for obs_idx in self.observation_indices:
            spot_obs = paths[:, obs_idx]
            alive_mask = alive
            if not np.any(alive_mask):
                break

            # We keep unpaid coupons as a running counter so the memory feature
            # remains transparent to audit and easy to compare with term sheets.
            unpaid_coupon_count[alive_mask] += 1

            coupon_hit = alive_mask & (spot_obs > self.coupon_barrier)
            coupon_amount = np.where(
                coupon_hit,
                self.notional * self.coupon_rate * unpaid_coupon_count.astype(np.float64),
                0.0,
            )
            discount_obs = np.exp(-self.risk_free_rate * (obs_idx * dt))
            discounted_payoff += coupon_amount * discount_obs
            unpaid_coupon_count[coupon_hit] = 0

            autocall_hit = alive_mask & (spot_obs > self.autocall_barrier)
            discounted_payoff[autocall_hit] += self.notional * discount_obs
            alive[autocall_hit] = False

        if np.any(alive):
            st = paths[:, -1]
            protected = st >= self.protection_barrier
            final_redemption = np.where(protected, self.notional, self.notional * (st / s_initial))
            discount_mat = np.exp(-self.risk_free_rate * self.maturity)
            discounted_payoff[alive] += final_redemption[alive] * discount_mat

        return discounted_payoff

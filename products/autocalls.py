"""Vectorized payoff engine for autocallable structured products."""

from __future__ import annotations

import numpy as np


class AutocallProduct:
    """Price a family of autocallables with one vectorized payoff engine.

    This class consolidates several common desk variants:
    standard autocalls, Phoenix notes, memory Phoenix notes, step-down
    autocalls, and airbag structures. The design intentionally avoids splitting
    each payoff flavor into a separate subclass because the core mechanics are
    the same and NumPy masks are a better fit than deep class hierarchies.
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
        is_memory: bool = False,
        airbag_floor: float | None = None,
        step_down_barriers: list[float] | np.ndarray | None = None,
    ) -> None:
        """Store structural terms for the product.

        Args:
            autocall_barrier: Level that triggers early redemption on an
                observation date.
            coupon_barrier: Level above which the coupon is paid on an
                observation date.
            protection_barrier: Capital protection threshold checked at
                maturity for paths that were not autocalled.
            coupon_rate: Coupon paid per observation, expressed as a fraction of
                notional.
            observation_indices: Observation dates expressed as column indices
                in the simulated path matrix.
            maturity: Final maturity in years.
            risk_free_rate: Continuously compounded discount rate used to return
                present values.
            notional: Product notional.
            is_memory: Whether missed coupons are accumulated and recovered when
                the coupon barrier is finally breached.
            airbag_floor: Downside floor applied to performance at maturity for
                airbag structures.
            step_down_barriers: Optional time-dependent autocall barriers, one
                value per observation date.
        """
        self.autocall_barrier = float(autocall_barrier)
        self.coupon_barrier = float(coupon_barrier)
        self.protection_barrier = float(protection_barrier)
        self.coupon_rate = float(coupon_rate)
        self.maturity = float(maturity)
        self.risk_free_rate = float(risk_free_rate)
        self.notional = float(notional)
        self.is_memory = bool(is_memory)
        self.airbag_floor = None if airbag_floor is None else float(airbag_floor)

        obs = np.asarray(observation_indices, dtype=np.int64)
        if obs.ndim != 1 or obs.size == 0:
            raise ValueError("observation_indices must be a non-empty 1D array.")
        self.observation_indices = np.unique(obs)

        if step_down_barriers is None:
            self.step_down_barriers = None
        else:
            barriers = np.asarray(step_down_barriers, dtype=np.float64)
            if barriers.ndim != 1 or barriers.size != self.observation_indices.size:
                raise ValueError(
                    "step_down_barriers must be a 1D array/list with one value per observation date."
                )
            self.step_down_barriers = barriers

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Compute the discounted payoff for every Monte Carlo path.

        Args:
            paths: Simulated spot matrix of shape ``(M_sims, N_obs)``, including
                the initial spot in the first column.

        Returns:
            Present value payoff for each simulated path.
        """
        paths = np.asarray(paths, dtype=np.float64)
        if paths.ndim != 2:
            raise ValueError("paths must be a 2D array with shape (M_sims, N_obs).")

        n_paths, n_steps = paths.shape
        if n_steps < 2:
            raise ValueError("paths must contain at least 2 time points.")
        if np.any(self.observation_indices <= 0) or np.any(self.observation_indices >= n_steps):
            raise ValueError("observation_indices must be within [1, N_obs-1].")

        s0 = paths[:, 0]
        if np.any(s0 <= 0.0):
            raise ValueError("Initial spot values must be strictly positive.")

        dt = self.maturity / (n_steps - 1)
        call_barriers = (
            self.step_down_barriers
            if self.step_down_barriers is not None
            else np.full(self.observation_indices.size, self.autocall_barrier, dtype=np.float64)
        )

        discounted_payoff = np.zeros(n_paths, dtype=np.float64)
        alive = np.ones(n_paths, dtype=np.bool_)
        coupon_stack = np.zeros(n_paths, dtype=np.float64)

        for obs_k, obs_idx in enumerate(self.observation_indices):
            alive_mask = alive
            if not np.any(alive_mask):
                break

            spot_obs = paths[:, obs_idx]
            discount_obs = np.exp(-self.risk_free_rate * (obs_idx * dt))

            coupon_hit = alive_mask & (spot_obs >= self.coupon_barrier)
            if self.is_memory:
                # Memory coupons are naturally represented by a running stack:
                # this keeps the implementation vectorized and avoids Python
                # loops over paths.
                coupon_stack[alive_mask] += 1.0
                coupon_multiplier = np.where(coupon_hit, coupon_stack, 0.0)
                coupon_stack[coupon_hit] = 0.0
            else:
                coupon_multiplier = coupon_hit.astype(np.float64)

            coupon_amount = self.notional * self.coupon_rate * coupon_multiplier
            discounted_payoff += coupon_amount * discount_obs

            autocall_hit = alive_mask & (spot_obs >= call_barriers[obs_k])
            discounted_payoff[autocall_hit] += self.notional * discount_obs
            alive[autocall_hit] = False
            coupon_stack[~alive] = 0.0

        if np.any(alive):
            st = paths[:, -1]
            performance = st / s0

            if self.airbag_floor is None:
                maturity_redemption = np.where(
                    st >= self.protection_barrier,
                    self.notional,
                    self.notional * performance,
                )
            else:
                # The airbag feature softens the downside once the protection
                # barrier is breached by flooring terminal performance.
                down_redemption = self.notional * np.maximum(performance, self.airbag_floor)
                maturity_redemption = np.where(
                    st >= self.protection_barrier,
                    self.notional,
                    down_redemption,
                )

            discount_mat = np.exp(-self.risk_free_rate * self.maturity)
            discounted_payoff[alive] += maturity_redemption[alive] * discount_mat

        return discounted_payoff

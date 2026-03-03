"""Monte Carlo path generation under the Heston stochastic-volatility model."""

import numpy as np
from numba import jit


@jit(nopython=True)
def generate_paths(
    S0: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    T: float,
    M: int,
    N: int,
) -> np.ndarray:
    """Simulate spot paths under the Heston model.

    The Heston model keeps the spot diffusion lognormal conditional on an
    instantaneous variance process that mean-reverts over time. In practice, it
    is a standard workhorse when a desk wants to capture smile dynamics more
    realistically than Black-Scholes while keeping Monte Carlo implementation
    relatively compact.

    Args:
        S0: Initial spot level at time zero.
        v0: Initial variance level.
        kappa: Mean-reversion speed of the variance process.
        theta: Long-run variance target.
        sigma: Volatility of variance, often called "vol of vol".
        rho: Instantaneous correlation between spot and variance shocks.
        T: Final maturity in years.
        M: Number of Monte Carlo paths.
        N: Number of observation dates per path, including time zero.

    Returns:
        A NumPy array of shape ``(M, N)`` containing simulated spot paths.
    """
    paths = np.empty((M, N), dtype=np.float64)
    dt = T / (N - 1)
    sqrt_dt = np.sqrt(dt)

    corr = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    chol = np.linalg.cholesky(corr)

    for i in range(M):
        s = S0
        v = v0
        paths[i, 0] = s

        for t in range(1, N):
            z1 = np.random.normal()
            z2 = np.random.normal()

            eps_s = chol[0, 0] * z1 + chol[0, 1] * z2
            eps_v = chol[1, 0] * z1 + chol[1, 1] * z2

            v_pos = v if v > 0.0 else 0.0

            # We evolve the spot in log space to preserve positivity pathwise.
            s = s * np.exp((-0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * eps_s)

            # Full truncation is the pragmatic choice here: it keeps the variance
            # process numerically stable without adding the overhead of a more
            # elaborate exact or QE scheme.
            v = v + kappa * (theta - v_pos) * dt + sigma * np.sqrt(v_pos) * sqrt_dt * eps_v
            if v < 0.0:
                v = 0.0

            paths[i, t] = s

    return paths

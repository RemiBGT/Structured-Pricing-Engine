"""Monte Carlo path generation under the Black-Scholes model."""

import numpy as np
from numba import jit


@jit(nopython=True)
def generate_bs_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    M: int,
    N: int,
) -> np.ndarray:
    """Simulate geometric Brownian motion paths under Black-Scholes.

    In the Black-Scholes setting, volatility is constant and the spot follows a
    lognormal diffusion. Because the SDE admits a closed-form transition, we can
    use the exact lognormal step rather than a crude Euler scheme.

    Args:
        S0: Initial spot level at time zero.
        r: Continuously compounded risk-free rate.
        sigma: Constant volatility parameter.
        T: Final maturity in years.
        M: Number of Monte Carlo paths.
        N: Number of observation dates per path, including time zero.

    Returns:
        A NumPy array of shape ``(M, N)`` containing simulated spot paths.
    """
    paths = np.empty((M, N), dtype=np.float64)
    dt = T / (N - 1)
    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * np.sqrt(dt)

    for i in range(M):
        s = S0
        paths[i, 0] = s
        for t in range(1, N):
            z = np.random.normal()
            # The exact GBM transition is both cleaner and more accurate than an
            # Euler discretization for this model.
            s = s * np.exp(drift + diffusion * z)
            paths[i, t] = s

    return paths

from __future__ import annotations

from typing import Protocol

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from models.black_scholes import generate_bs_paths
from models.heston import generate_paths as generate_heston_paths
from products.autocalls import AutocallProduct
from products.cliquets import CliquetProduct


T_MATURITY = 1.0
RISK_FREE_RATE = 0.02
N_STEPS = 64
DEFAULT_SIMULATIONS = 10_000
DEFAULT_RNG_SEED = 7
SPOT_BUMP_REL = 0.01
SIGMA_BUMP_ABS = 0.01
DEFAULT_BS_SIGMA = 0.20
DEFAULT_HESTON_VOL0 = 0.20
DEFAULT_HESTON_KAPPA = 2.0
DEFAULT_HESTON_THETA = 0.04
DEFAULT_HESTON_SIGMA = 0.50
DEFAULT_HESTON_RHO = -0.5

AUTOCALL_BARRIER_RATIO = 1.00
COUPON_BARRIER_RATIO = 0.80
PROTECTION_BARRIER_RATIO = 0.60
COUPON_RATE = 0.02
STEP_DOWN_DECREMENT_RATIO = 0.02
AIRBAG_FLOOR = 0.70

CLIQUET_PARTICIPATION = 1.0
CLIQUET_CAP = 0.03

NOTIONAL = 100.0


class PayoffProduct(Protocol):
    """Protocol for product engines exposing a vectorized payoff method."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Return the discounted payoff for each Monte Carlo path."""


def _observation_indices(n_steps: int) -> np.ndarray:
    """Build approximately monthly observation dates over the simulation grid.

    Args:
        n_steps: Number of simulated time points, including time zero.

    Returns:
        A strictly increasing array of observation indices. The final maturity
        index is always included.
    """
    monthly_step = max(1, int(round((n_steps - 1) / 12)))
    obs = np.arange(monthly_step, n_steps, monthly_step, dtype=np.int64)
    if obs.size == 0 or obs[-1] != n_steps - 1:
        obs = np.append(obs, n_steps - 1)
    return np.unique(obs)


def _build_step_down_barriers(
    spot0: float,
    base_ratio: float,
    decrement_ratio: float,
    n_obs: int,
) -> list[float]:
    """Create a declining autocall barrier schedule.

    A step-down autocall lowers the call trigger as time passes, which increases
    the probability of early redemption on later observation dates.

    Args:
        spot0: Initial spot level.
        base_ratio: Initial barrier ratio expressed versus spot.
        decrement_ratio: Amount subtracted at each observation.
        n_obs: Number of observation dates.

    Returns:
        A list of absolute barrier levels, one per observation date.
    """
    obs_order = np.arange(n_obs, dtype=np.float64)
    step_down_ratios = np.maximum(base_ratio - decrement_ratio * obs_order, 0.0)
    return (step_down_ratios * spot0).tolist()


def _build_product(
    *,
    spot0: float,
    product_type: str,
    cliquet_variant: str,
    autocall_barrier_ratio: float,
    coupon_barrier_ratio: float,
    protection_barrier_ratio: float,
    coupon_rate: float,
    use_step_down: bool,
    step_down_decrement_ratio: float,
    airbag_floor: float | None,
    cliquet_participation: float,
    cliquet_cap: float,
) -> PayoffProduct:
    """Instantiate the product selected in the dashboard.

    Args:
        spot0: Initial spot level.
        product_type: Product family selected by the user.
        cliquet_variant: Variant selected when the product is a cliquet.
        autocall_barrier_ratio: Autocall barrier expressed as a ratio of spot.
        coupon_barrier_ratio: Coupon barrier expressed as a ratio of spot.
        protection_barrier_ratio: Maturity protection barrier ratio.
        coupon_rate: Coupon paid per observation.
        use_step_down: Whether the autocall barrier decreases over time.
        step_down_decrement_ratio: Step-down decrement applied at each
            observation.
        airbag_floor: Minimum terminal performance retained by the airbag
            structure once protection is breached.
        cliquet_participation: Participation used by capped-coupon cliquets.
        cliquet_cap: Local cap used by capped-coupon cliquets.

    Returns:
        A product object exposing a vectorized ``payoff`` method.
    """
    if product_type == "Cliquet":
        payoff_type = "capped_coupons" if cliquet_variant == "Capped Coupons" else "max_return"
        return CliquetProduct(
            payoff_type=payoff_type,
            participation=cliquet_participation,
            cap=cliquet_cap,
            maturity=T_MATURITY,
            risk_free_rate=RISK_FREE_RATE,
            notional=NOTIONAL,
        )

    obs_idx = _observation_indices(N_STEPS)
    step_down_barriers = (
        _build_step_down_barriers(
            spot0=spot0,
            base_ratio=autocall_barrier_ratio,
            decrement_ratio=step_down_decrement_ratio,
            n_obs=obs_idx.size,
        )
        if use_step_down
        else None
    )

    return AutocallProduct(
        autocall_barrier=autocall_barrier_ratio * spot0,
        coupon_barrier=coupon_barrier_ratio * spot0,
        protection_barrier=protection_barrier_ratio * spot0,
        coupon_rate=coupon_rate,
        observation_indices=obs_idx,
        maturity=T_MATURITY,
        risk_free_rate=RISK_FREE_RATE,
        notional=NOTIONAL,
        is_memory=(product_type == "Memory Phoenix"),
        airbag_floor=airbag_floor if product_type == "Airbag" else None,
        step_down_barriers=step_down_barriers,
    )


def _simulate_paths(
    *,
    model_name: str,
    s0: float,
    model_params: dict[str, float],
    n_sims: int,
    seed: int,
) -> np.ndarray:
    """Dispatch path generation to the selected stochastic model.

    Args:
        model_name: Either ``"Heston"`` or ``"Black-Scholes"``.
        s0: Spot level used for the current simulation run.
        model_params: Model calibration parameters required by the selected
            engine.
        n_sims: Number of Monte Carlo paths.
        seed: Random seed used to enforce common random numbers across bumps.

    Returns:
        Simulated spot paths of shape ``(n_sims, N_STEPS)``.
    """
    np.random.seed(seed)
    if model_name == "Heston":
        return generate_heston_paths(
            S0=s0,
            v0=float(model_params["v0"]),
            kappa=float(model_params["kappa"]),
            theta=float(model_params["theta"]),
            sigma=float(model_params["sigma"]),
            rho=float(model_params["rho"]),
            T=T_MATURITY,
            M=n_sims,
            N=N_STEPS,
        )
    if model_name == "Black-Scholes":
        return generate_bs_paths(
            S0=s0,
            r=RISK_FREE_RATE,
            sigma=float(model_params["sigma"]),
            T=T_MATURITY,
            M=n_sims,
            N=N_STEPS,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def calculate_price_and_greeks(
    *,
    product: PayoffProduct,
    model_name: str,
    model_params: dict[str, float],
    n_sims: int,
    seed: int = DEFAULT_RNG_SEED,
    spot_bump_rel: float = SPOT_BUMP_REL,
    sigma_bump_abs: float = SIGMA_BUMP_ABS,
) -> dict[str, float | np.ndarray]:
    """Estimate price and first-order Greeks by bump-and-reprice.

    This mirrors the desk-style finite-difference approach used in many legacy
    C++ pricers: compute a base price, then re-run the Monte Carlo engine with
    shocked inputs. The same seed is reused across bumps so the shocks are
    compared under common random numbers, which dramatically reduces the noise
    of Delta, Gamma, and Vega estimates.

    Args:
        product: Structured product payoff engine.
        model_name: Either ``"Heston"`` or ``"Black-Scholes"``.
        model_params: Model parameters required by the selected engine.
        n_sims: Number of Monte Carlo paths.
        seed: Seed reused for each bump scenario.
        spot_bump_rel: Relative spot shock used for Delta and Gamma.
        sigma_bump_abs: Absolute shock applied to the model volatility
            parameter. For Heston this bumps vol of vol, and for Black-Scholes
            it bumps the constant volatility.

    Returns:
        A dictionary containing price, Greeks, standard error, the base payoff
        distribution, and the first 50 simulated paths for plotting.
    """
    s0 = float(model_params["S0"])
    ds = spot_bump_rel * s0
    if ds <= 0.0:
        raise ValueError("spot_bump_rel * S0 must be strictly positive.")
    if sigma_bump_abs <= 0.0:
        raise ValueError("sigma_bump_abs must be strictly positive.")

    bumped_model_params = dict(model_params)
    bumped_model_params["sigma"] = max(1e-8, float(model_params["sigma"]) + sigma_bump_abs)

    # Reusing the same seed across scenarios is the simplest way to introduce
    # common random numbers and stabilize finite-difference Greeks.
    base_paths = _simulate_paths(
        model_name=model_name,
        s0=s0,
        model_params=model_params,
        n_sims=n_sims,
        seed=seed,
    )
    up_paths = _simulate_paths(
        model_name=model_name,
        s0=s0 * (1.0 + spot_bump_rel),
        model_params=model_params,
        n_sims=n_sims,
        seed=seed,
    )
    down_paths = _simulate_paths(
        model_name=model_name,
        s0=s0 * (1.0 - spot_bump_rel),
        model_params=model_params,
        n_sims=n_sims,
        seed=seed,
    )
    sigma_up_paths = _simulate_paths(
        model_name=model_name,
        s0=s0,
        model_params=bumped_model_params,
        n_sims=n_sims,
        seed=seed,
    )

    payoff_base = product.payoff(base_paths)
    p0 = float(np.mean(payoff_base))
    p_up = float(np.mean(product.payoff(up_paths)))
    p_down = float(np.mean(product.payoff(down_paths)))
    p_sigma_up = float(np.mean(product.payoff(sigma_up_paths)))

    delta = (p_up - p_down) / (2.0 * ds)
    gamma = (p_up - 2.0 * p0 + p_down) / (ds * ds)
    vega = (p_sigma_up - p0) / sigma_bump_abs

    if payoff_base.size > 1:
        std_error = float(np.std(payoff_base, ddof=1) / np.sqrt(payoff_base.size))
    else:
        std_error = 0.0

    return {
        "Price": p0,
        "Delta": float(delta),
        "Gamma": float(gamma),
        "Vega": float(vega),
        "StandardError": std_error,
        "PayoffDistribution": payoff_base,
        "Paths50": base_paths[:50],
    }


def run_simulation(
    *,
    product: PayoffProduct,
    model_name: str,
    model_params: dict[str, float],
    n_sims: int,
    seed: int = DEFAULT_RNG_SEED,
) -> dict[str, float | np.ndarray]:
    """Execute the pricing workflow for the current dashboard selection.

    Args:
        product: Structured product payoff engine.
        model_name: Either ``"Heston"`` or ``"Black-Scholes"``.
        model_params: Parameters required by the selected model.
        n_sims: Number of Monte Carlo paths.
        seed: Seed used for Monte Carlo reproducibility.

    Returns:
        The pricing and Greek estimates returned by
        ``calculate_price_and_greeks``.
    """
    return calculate_price_and_greeks(
        product=product,
        model_name=model_name,
        model_params=model_params,
        n_sims=n_sims,
        seed=seed,
    )


def _plot_paths(paths: np.ndarray) -> go.Figure:
    """Build the Plotly figure used to display simulated spot paths.

    Args:
        paths: Simulated spot paths, already truncated to a visually safe
            sample.

    Returns:
        A Plotly line figure.
    """
    figure = go.Figure()
    horizon = np.linspace(0.0, T_MATURITY, paths.shape[1])
    for i in range(paths.shape[0]):
        figure.add_trace(
            go.Scatter(
                x=horizon,
                y=paths[i],
                mode="lines",
                line={"width": 1},
                opacity=0.6,
                showlegend=False,
            )
        )
    figure.update_layout(
        title="Monte Carlo Paths (50)",
        xaxis_title="Time (Years)",
        yaxis_title="Spot",
        template="plotly_white",
        height=380,
    )
    return figure


def _plot_payoff_histogram(payoff_distribution: np.ndarray) -> go.Figure:
    """Build the histogram of simulated discounted payoffs.

    Args:
        payoff_distribution: One discounted payoff per Monte Carlo path.

    Returns:
        A Plotly histogram figure.
    """
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=payoff_distribution,
            nbinsx=30,
            marker={"color": "#3B9C6D"},
            opacity=0.85,
            name="Discounted Payoff",
        )
    )
    figure.update_layout(
        title="Product Payoff Distribution",
        xaxis_title="Discounted Payoff",
        yaxis_title="Frequency",
        template="plotly_white",
        height=380,
    )
    return figure


st.set_page_config(page_title="Structured Products Pricing & Hedging", layout="wide")
st.title("Structured Products Pricing & Hedging Engine")
# TODO: Move the dashboard wiring into smaller UI-focused functions as the app grows.

st.sidebar.header("Model Inputs")
model_name = st.sidebar.selectbox("Model", ("Heston", "Black-Scholes"))
spot0 = st.sidebar.number_input("Spot", min_value=1.0, value=100.0, step=1.0)
n_simulations = st.sidebar.number_input(
    "Number of Simulations",
    min_value=1_000,
    value=DEFAULT_SIMULATIONS,
    step=1_000,
)

bs_sigma = DEFAULT_BS_SIGMA
heston_vol0 = DEFAULT_HESTON_VOL0
heston_kappa = DEFAULT_HESTON_KAPPA
heston_theta = DEFAULT_HESTON_THETA
heston_sigma = DEFAULT_HESTON_SIGMA
heston_rho = DEFAULT_HESTON_RHO

if model_name == "Heston":
    heston_vol0 = st.sidebar.slider(
        "Initial Volatility (sqrt(v0))",
        min_value=0.01,
        max_value=2.0,
        value=DEFAULT_HESTON_VOL0,
        step=0.01,
    )
    heston_kappa = st.sidebar.number_input(
        "Kappa",
        min_value=0.01,
        max_value=10.0,
        value=DEFAULT_HESTON_KAPPA,
        step=0.1,
    )
    heston_theta = st.sidebar.number_input(
        "Theta",
        min_value=0.0001,
        max_value=2.0,
        value=DEFAULT_HESTON_THETA,
        step=0.005,
    )
    heston_sigma = st.sidebar.number_input(
        "Sigma (Vol of Vol)",
        min_value=0.01,
        max_value=5.0,
        value=DEFAULT_HESTON_SIGMA,
        step=0.01,
    )
    heston_rho = st.sidebar.number_input(
        "Rho",
        min_value=-0.999,
        max_value=0.999,
        value=DEFAULT_HESTON_RHO,
        step=0.01,
    )
else:
    bs_sigma = st.sidebar.slider(
        "Volatility (sigma)",
        min_value=0.01,
        max_value=2.0,
        value=DEFAULT_BS_SIGMA,
        step=0.01,
    )

st.sidebar.header("Product Inputs")
product_type = st.sidebar.selectbox("Product Type", ("Phoenix", "Memory Phoenix", "Airbag", "Cliquet"))

autocall_barrier_ratio = AUTOCALL_BARRIER_RATIO
coupon_barrier_ratio = COUPON_BARRIER_RATIO
protection_barrier_ratio = PROTECTION_BARRIER_RATIO
coupon_rate = COUPON_RATE
use_step_down = False
step_down_decrement_ratio = STEP_DOWN_DECREMENT_RATIO
airbag_floor: float | None = None
cliquet_variant = "Capped Coupons"
cliquet_participation = CLIQUET_PARTICIPATION
cliquet_cap = CLIQUET_CAP

if product_type in {"Phoenix", "Memory Phoenix", "Airbag"}:
    autocall_barrier_ratio = st.sidebar.number_input(
        "Autocall Barrier Ratio",
        min_value=0.01,
        max_value=2.0,
        value=AUTOCALL_BARRIER_RATIO,
        step=0.01,
    )
    coupon_barrier_ratio = st.sidebar.number_input(
        "Coupon Barrier Ratio",
        min_value=0.01,
        max_value=2.0,
        value=COUPON_BARRIER_RATIO,
        step=0.01,
    )
    protection_barrier_ratio = st.sidebar.number_input(
        "Protection Barrier Ratio",
        min_value=0.01,
        max_value=2.0,
        value=PROTECTION_BARRIER_RATIO,
        step=0.01,
    )
    coupon_rate = st.sidebar.number_input(
        "Coupon Rate (per observation)",
        min_value=0.0,
        max_value=0.50,
        value=COUPON_RATE,
        step=0.005,
    )
    use_step_down = st.sidebar.checkbox("Enable Step-Down", value=False)
    if use_step_down:
        step_down_decrement_ratio = st.sidebar.number_input(
            "Step-down Decrement",
            min_value=0.0,
            max_value=0.20,
            value=STEP_DOWN_DECREMENT_RATIO,
            step=0.005,
        )
    if product_type == "Airbag":
        airbag_floor = st.sidebar.slider(
            "Airbag Floor",
            min_value=0.0,
            max_value=1.0,
            value=AIRBAG_FLOOR,
            step=0.01,
        )
else:
    cliquet_variant = st.sidebar.selectbox("Cliquet Variant", ("Capped Coupons", "Max Return"))
    if cliquet_variant == "Capped Coupons":
        cliquet_participation = st.sidebar.number_input(
            "Participation",
            min_value=0.0,
            max_value=5.0,
            value=CLIQUET_PARTICIPATION,
            step=0.05,
        )
        cliquet_cap = st.sidebar.number_input(
            "Coupon Cap",
            min_value=0.0,
            max_value=0.5,
            value=CLIQUET_CAP,
            step=0.005,
        )

if st.button("Run Simulation", type="primary"):
    with st.spinner("Running Monte Carlo pricing and Greeks..."):
        n_sims = int(n_simulations)
        product = _build_product(
            spot0=spot0,
            product_type=product_type,
            cliquet_variant=cliquet_variant,
            autocall_barrier_ratio=autocall_barrier_ratio,
            coupon_barrier_ratio=coupon_barrier_ratio,
            protection_barrier_ratio=protection_barrier_ratio,
            coupon_rate=coupon_rate,
            use_step_down=use_step_down,
            step_down_decrement_ratio=step_down_decrement_ratio,
            airbag_floor=airbag_floor,
            cliquet_participation=cliquet_participation,
            cliquet_cap=cliquet_cap,
        )

        if model_name == "Heston":
            model_params = {
                "S0": float(spot0),
                "v0": float(heston_vol0 * heston_vol0),
                "kappa": float(heston_kappa),
                "theta": float(heston_theta),
                "sigma": float(heston_sigma),
                "rho": float(heston_rho),
            }
        else:
            model_params = {
                "S0": float(spot0),
                "sigma": float(bs_sigma),
            }

        # The UI should stay responsive even when the pricing run uses a large
        # Monte Carlo sample, so we keep only a small path subset for plotting.
        greeks = run_simulation(
            product=product,
            model_name=model_name,
            model_params=model_params,
            n_sims=n_sims,
            seed=DEFAULT_RNG_SEED,
        )

        st.session_state["sim_results"] = {
            "model_name": model_name,
            "product_type": product_type,
            "price": float(greeks["Price"]),
            "delta": float(greeks["Delta"]),
            "gamma": float(greeks["Gamma"]),
            "vega": float(greeks["Vega"]),
            "std_error": float(greeks["StandardError"]),
            "n_simulations": n_sims,
            "mc_paths_50": np.asarray(greeks["Paths50"], dtype=np.float64),
            "payoff_distribution": np.asarray(greeks["PayoffDistribution"], dtype=np.float64),
        }

if "sim_results" in st.session_state:
    results = st.session_state["sim_results"]

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Price", f"{results['price']:.4f}")
    metric_col2.metric("Delta", f"{results['delta']:.6f}")
    metric_col3.metric("Gamma", f"{results['gamma']:.8f}")
    metric_col4.metric("Vega", f"{results['vega']:.6f}")

    info_col1, info_col2 = st.columns(2)
    info_col1.metric("Standard Error", f"{results['std_error']:.6f}")
    info_col2.metric("Simulations", f"{results['n_simulations']:,}")

    st.caption(f"Model: {results['model_name']} | Product: {results['product_type']}")
    # TODO: Add confidence intervals around the price to make Monte Carlo noise explicit.
    st.plotly_chart(_plot_paths(results["mc_paths_50"]), use_container_width=True)
    st.plotly_chart(_plot_payoff_histogram(results["payoff_distribution"]), use_container_width=True)

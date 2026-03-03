# Structured Products Pricing & Hedging Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([YOUR_LINK_HERE](https://structured-pricing-engine-nvxiv5s6z4crzdwyvo6wmq.streamlit.app/))

A Python-based pricing and risk engine for path-dependent structured products, built to combine quantitative clarity with practical performance. The project prices autocallables and cliquets with fully vectorized NumPy payoff logic, accelerates Monte Carlo simulations with Numba JIT, and exposes the workflow through an interactive Streamlit dashboard.

The core design targets the usual bottlenecks of retail structured-products prototyping: expensive path generation, path-dependent coupon logic, and noisy finite-difference risk estimation. By combining vectorized payoff evaluation with compiled Monte Carlo loops, the engine remains fast enough for interactive scenario analysis while staying transparent and easy to extend.

## Key Features

- Monte Carlo pricing under both Black-Scholes and Heston dynamics
- Numba-accelerated path generation for near C++-style performance in Python
- Vectorized NumPy payoff engines with no Python loop over simulation paths
- Flexible autocall framework supporting:
  - Standard Autocalls
  - Phoenix notes
  - Memory Phoenix notes
  - Step-Down autocalls
  - Airbag-style maturity redemption
- Cliquet payoff engine supporting:
  - Capped local coupon accumulation
  - Max-return structures
- Finite-difference Greeks computed by bump-and-reprice:
  - Delta
  - Gamma
  - Vega
- Common Random Numbers (shared RNG seed across bumps) to improve Greek stability
- Streamlit dashboard for model selection, product configuration, pricing, and payoff visualization
- Plotly charts for path inspection and payoff distribution analysis

## Tech Stack

- Python
- NumPy
- Numba
- Streamlit
- Plotly

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```text
.
|-- app.py
|-- hedging/
|   `-- simulator.py
|-- models/
|   |-- black_scholes.py
|   `-- heston.py
|-- products/
|   |-- autocalls.py
|   |-- cliquets.py
|   `-- structure.py
`-- _check_import.py
```

## What the Engine Covers

### Pricing Models

- **Black-Scholes**: constant-volatility geometric Brownian motion, useful as a clean baseline model
- **Heston**: stochastic-volatility dynamics with mean-reverting variance, better suited for volatility-smile-aware scenario work

### Product Families

- **Autocalls**: early redemption when the call barrier is breached on an observation date
- **Phoenix / Memory Phoenix**: periodic conditional coupons with optional coupon memory
- **Step-Down Autocalls**: declining call barrier schedule over time
- **Airbag Structures**: downside softening via a terminal performance floor after protection breach
- **Cliquets**: local reset payoffs driven by capped coupons or best observed performance

### Risk Measures

The dashboard computes Greeks via finite differences using a desk-style bump-and-reprice workflow. This keeps the risk logic explicit and easy to audit while remaining compatible with non-linear, path-dependent payoffs that do not admit simple closed-form sensitivities.

## Methodology

### 1. Monte Carlo Path Generation

The engine simulates underlying paths under either:

- **Black-Scholes**, using the exact geometric Brownian motion transition
- **Heston**, using a log-Euler spot update coupled with a full-truncation Euler scheme for variance

Both simulation kernels are compiled with Numba so that the expensive path-generation loops run at compiled speed instead of pure Python speed.

### 2. Vectorized Payoff Evaluation

Once paths are generated, payoffs are computed in batch using NumPy arrays and boolean masks. This is especially important for path-dependent structures such as Phoenix notes, Memory Phoenix notes, and Step-Down autocalls, where coupon and redemption logic can otherwise become a major bottleneck.

Rather than iterating over each simulation path in Python, the payoff engines operate directly on full path matrices:

- conditional coupon payments are handled with masks
- memory effects are tracked with vectorized coupon stacks
- early redemption is applied through alive/dead path masks
- maturity redemption is computed in one pass across the remaining paths

### 3. Finite-Difference Greeks

The pricing engine computes Greeks with a bump-and-reprice methodology:

- **Price**: base Monte Carlo estimate
- **Delta / Gamma**: spot bumped up and down
- **Vega**: volatility parameter bumped upward

To reduce Monte Carlo noise, the engine reuses the same RNG seed across base and bumped scenarios. This common-random-numbers approach materially improves convergence of finite-difference sensitivities without introducing extra implementation complexity.

## Design Principles

- **Performance first**: simulation loops are compiled with Numba, while payoff logic is vectorized with NumPy masks
- **Model transparency**: pricing assumptions and payoff mechanics are readable and easy to inspect
- **Practical extensibility**: new product variants can be added by extending parameterized payoff engines instead of introducing deep class hierarchies
- **Interactive workflow**: the Streamlit UI is designed for rapid prototyping, parameter sweeps, and visual sanity checks

## Limitations / Assumptions

- Greeks are computed by finite differences, so they remain sensitive to bump size and Monte Carlo noise.
- The Heston implementation uses a practical full-truncation Euler discretization rather than a higher-order or exact variance scheme.
- Vega is defined as a bump on the model volatility parameter:
  - Black-Scholes: constant volatility `sigma`
  - Heston: volatility of variance (`vol of vol`)
- Discounting currently uses a flat deterministic risk-free rate.
- There is currently no market calibration layer, no volatility-surface fitting, and no historical backtesting workflow exposed in the UI.

## Roadmap

- Add automated unit tests for payoff edge cases and Greek stability
- Add calibration helpers and richer risk reporting (confidence intervals, scenario grids)

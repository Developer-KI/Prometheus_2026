# Heston Stochastic-Vol × Short Straddle Strategy

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Every REFIT_DAYS (default 5):                                  │
│                                                                 │
│   Options Chain ──► Heston Calibration ──► QE Monte-Carlo       │
│   (strikes, mids,    (v0,κ,θ,σᵥ,ρ)        3000 paths × 5d     │
│    expiries, IVs)                           → vol density       │
│                                                                 │
│   Fallback: returns + VIX variance proxy (Euler MLE)            │
├─────────────────────────────────────────────────────────────────┤
│  Every trading day:                                             │
│                                                                 │
│   Vol Density ──► Signal Gates ──► Half-Kelly Sizer ──► Trade   │
│   + RV Panel       G1: VRP breadth     tail dampener     ATM    │
│   + IV (√v0)       G2: IV>RV panel     kurtosis damp    short   │
│   + Regime         G3: tail contain                     strdl   │
│                    G4: kurtosis cap                              │
│                    G5: skew cap                                  │
│                    G6: regime penalty                            │
├─────────────────────────────────────────────────────────────────┤
│  Risk Management:                                               │
│   • Max-loss stop: 2× premium collected                         │
│   • Assignment auto-unwind (liquidate underlying)               │
│   • Max 4% portfolio risk per trade                             │
│   • 5% notional cap                                             │
└─────────────────────────────────────────────────────────────────┘
```

## Heston Model

The Heston stochastic-volatility model:

```
dS/S  = (r − q) dt + √v dW₁
dv    = κ(θ − v) dt + σᵥ √v dW₂
corr(dW₁, dW₂) = ρ
```

**Parameters**: v0 (instantaneous variance), κ (mean-reversion speed),
θ (long-run variance), σᵥ (vol-of-vol), ρ (correlation, typically negative for equities).

### Calibration Mode 1 — Options Chain (Primary)

Fits all 5 parameters to the cross-section of market option prices using
vega-weighted least-squares. Gil-Pelaez / Fourier inversion for Heston
pricing with the "little Heston trap" formulation to avoid branch-cut
discontinuities. Falls back to differential evolution if L-BFGS-B
converges poorly.

### Calibration Mode 2 — Returns + VIX (Fallback)

When the chain is too thin (< 8 usable contracts), uses Euler-discretised
approximate MLE on log-returns and VIX-implied variance as an observable
proxy for the latent variance process. Vectorised numpy for speed.

### Forecast

Andersen Quadratic-Exponential (QE) scheme for variance-path simulation.
Returns a distribution of annualised realised vol over the forecast horizon.

### Key Parameters to Tune

All configurable as class constants at the top of `HestonVolSell`:

| Parameter            | Default | Description                           |
| -------------------- | ------- | ------------------------------------- |
| `REFIT_DAYS`         | 5       | Recalibrate Heston every N days       |
| `FC_HORIZON`         | 5       | MC forecast horizon (trading days)    |
| `MC_SIMS`            | 3000    | Monte-Carlo paths                     |
| `TARGET_DTE_MIN/MAX` | 1/5     | DTE range for straddle selection      |
| `MAX_RISK_FRAC`      | 0.04    | Max portfolio % at risk per trade     |
| `MAX_LOSS_MULT`      | 2.0     | Stop-loss as multiple of premium      |
| `VRP_STRONG`         | 0.10    | Strong VRP threshold for full signal  |
| `VRP_WEAK`           | 0.05    | Weak VRP threshold for partial signal |
| `IV_HAIRCUT`         | 0.02    | Conservative IV adjustment            |
| `MONEYNESS_BAND`     | 0.15    | Chain filter: ±15% from ATM           |
| `CHAIN_MIN_OPTIONS`  | 8       | Min contracts for chain calibration   |

### Live Trading Notes

- The strategy uses `market_order` for fills — consider switching to
  `limit_order` with mid-price for live to reduce slippage.
- VIX data feed must be active for the fallback calibration path.
- Monitor Heston parameter stability in logs — κ hitting bounds
  (30.0) or ρ near 0 may indicate calibration issues.
- The Feller condition (2κθ > σᵥ²) should generally hold; violations
  logged as warnings indicate the variance process may hit zero.

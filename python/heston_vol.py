"""
Heston Stochastic-Vol Selling Strategy
=========================================================
Heston model calibrated from:
  • Live options chain  (yfinance)        — for forward-looking use
  • Returns + VIX9D proxy                 — for backtesting (no historical chains)

Forecast density via Andersen QE Monte-Carlo
5-day MC forecast  ×  2-DTE ATM short straddles
VIX9D as IV proxy  /  Half-Kelly + tail-risk gate

Robustness suite
────────────────
• Slippage & transaction-cost sweep (with per-leg bid-ask)
• Parameter sensitivity stress test

No-Lookahead Protocol
─────────────────────
• Heston fitted on returns[…t−1] + VIX9D[…t−1]
• RV panel uses OHLC[…t−1]
• IV = VIX9D close[t−1]
• Entry = Open[t]   Exit = Close[t]
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
from dataclasses import dataclass, field
import warnings, os, time, copy

warnings.filterwarnings("ignore")
_cdf = stats.norm.cdf
_pdf = stats.norm.pdf
_SQ252 = np.sqrt(252)
_ISQ2PI = 1.0 / np.sqrt(2 * np.pi)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# ═══════════════════════════════════════════════════════════════
#  BLACK-SCHOLES HELPERS
# ═══════════════════════════════════════════════════════════════
def _d1d2(S, K, T, r, v):
    T = max(T, 1e-10)
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * v ** 2) * T) / (v * sq)
    return d1, d1 - v * sq


def bs_call(S, K, T, r, v):
    d1, d2 = _d1d2(S, K, T, r, v)
    return S * _cdf(d1) - K * np.exp(-r * T) * _cdf(d2)


def bs_put(S, K, T, r, v):
    d1, d2 = _d1d2(S, K, T, r, v)
    return K * np.exp(-r * T) * _cdf(-d2) - S * _cdf(-d1)


def straddle_px(S, K, T, r, v):
    return bs_call(S, K, T, r, v) + bs_put(S, K, T, r, v)


def bs_gamma(S, K, T, r, v):
    T = max(T, 1e-10)
    d1, _ = _d1d2(S, K, T, r, v)
    return _pdf(d1) / (S * v * np.sqrt(T))


def bs_vega(S, K, T, r, v):
    """BS vega — dollar vega per 1 vol point."""
    T = max(T, 1e-10)
    d1, _ = _d1d2(S, K, T, r, v)
    return S * _pdf(d1) * np.sqrt(T)


def bs_iv_newton(price, S, K, T, r, otype="call", tol=1e-6, maxiter=50):
    """Recover implied vol from a market price via Newton-Raphson."""
    sig = 0.20
    for _ in range(maxiter):
        px = bs_call(S, K, T, r, sig) if otype == "call" else bs_put(S, K, T, r, sig)
        diff = px - price
        vg = bs_vega(S, K, T, r, sig)
        if abs(vg) < 1e-12:
            break
        sig -= diff / vg
        sig = np.clip(sig, 0.01, 5.0)
        if abs(diff) < tol:
            break
    return sig


# ═══════════════════════════════════════════════════════════════
#  REALISED VOL PANEL
# ═══════════════════════════════════════════════════════════════
class RVol:
    @staticmethod
    def cc(c, w=22):
        return np.log(c / c.shift(1)).rolling(w).std() * _SQ252

    @staticmethod
    def parkinson(h, l, w=22):
        return np.sqrt(
            (np.log(h / l) ** 2 / (4 * np.log(2))).rolling(w).mean() * 252
        )

    @staticmethod
    def rogers_satchell(o, h, l, c, w=22):
        lho = np.log(h / o)
        llo = np.log(l / o)
        lco = np.log(c / o)
        return np.sqrt(
            (lho * (lho - lco) + llo * (llo - lco)).rolling(w).mean() * 252
        )

    @staticmethod
    def yang_zhang(o, h, l, c, w=22):
        k = 0.34 / (1.34 + (w + 1) / (w - 1))
        loc_ = np.log(o / c.shift(1))
        lco = np.log(c / o)
        lho = np.log(h / o)
        llo = np.log(l / o)
        rs = lho * (lho - lco) + llo * (llo - lco)
        v = (
            loc_.rolling(w).var()
            + k * lco.rolling(w).var()
            + (1 - k) * rs.rolling(w).mean()
        )
        return np.sqrt(v.clip(lower=0) * 252)

    @staticmethod
    def bipower(c, w=22):
        lr = np.log(c / c.shift(1))
        mu1 = np.sqrt(2 / np.pi)
        return np.sqrt(
            ((1 / mu1 ** 2) * lr.abs() * lr.abs().shift(1)).rolling(w).mean() * 252
        )


def build_rv_panel(spy, window_size=22):
    p = pd.DataFrame(index=spy.index)
    p["rv_cc"] = RVol.cc(spy["Close"], w=window_size)
    p["rv_pk"] = RVol.parkinson(spy["High"], spy["Low"], w=window_size)
    p["rv_rs"] = RVol.rogers_satchell(
        spy["Open"], spy["High"], spy["Low"], spy["Close"], w=window_size
    )
    p["rv_yz"] = RVol.yang_zhang(
        spy["Open"], spy["High"], spy["Low"], spy["Close"], w=window_size
    )
    p["rv_bp"] = RVol.bipower(spy["Close"], w=window_size)
    return p


# ═══════════════════════════════════════════════════════════════
#  HESTON MODEL
# ═══════════════════════════════════════════════════════════════
#  dS/S  = (r − q) dt + √v dW₁
#  dv    = κ(θ − v) dt + σ_v √v dW₂
#  corr(dW₁, dW₂) = ρ
#
#  Parameters: v0, kappa, theta, sigma_v, rho   (5)
# ═══════════════════════════════════════════════════════════════

class HestonModel:
    """
    Heston stochastic-volatility model.

    Calibration modes
    -----------------
    1. ``calibrate_from_chain``  — fit to live options chain (yfinance)
    2. ``calibrate_from_returns`` — fit to historical returns + VIX proxy
       (for backtesting where historical chains are unavailable)

    Forecast
    --------
    ``forecast_density`` — Andersen QE Monte-Carlo, returns annualised
    realised-vol distribution over *horizon* days.
    """

    def __init__(self):
        # Heston parameters
        self.v0: float = 0.04        # initial variance
        self.kappa: float = 2.0      # mean-reversion speed
        self.theta: float = 0.04     # long-run variance
        self.sigma_v: float = 0.3    # vol-of-vol
        self.rho: float = -0.7       # correlation
        self._fitted = False

    # ─── characteristic function (Gatheral / "little Heston trap") ───
    @staticmethod
    def _cf_logreturn(u, T, v0, kappa, theta, sigma_v, rho):
        """
        CF of the *centred* log-return  E[exp(iu · (ln S_T/S − (r−q)T))].
        The risk-neutral drift is handled separately in the pricing routine.

        Uses the Albrecher et al. formulation ("little Heston trap") to
        avoid the branch-cut discontinuity.

        Parameters
        ----------
        u : complex or float — transform variable (may be complex for P₁)
        """
        iu = 1j * u
        sv2 = sigma_v * sigma_v

        # Riccati coefficients
        beta = kappa - rho * sigma_v * iu
        d = np.sqrt(beta * beta + sv2 * (iu + u * u))

        g = (beta - d) / (beta + d)
        edt = np.exp(-d * T)

        C = (kappa * theta / sv2) * (
            (beta - d) * T - 2.0 * np.log((1.0 - g * edt) / (1.0 - g))
        )
        D = ((beta - d) / sv2) * (1.0 - edt) / (1.0 - g * edt)

        return np.exp(C + D * v0)

    # ─── option pricing via Gil-Pelaez inversion ───
    def price_european(self, S, K, T, r, q=0.0, otype="call"):
        """
        Price a European option under Heston via Gil-Pelaez inversion.

        C = S e^{-qT} P₁ − K e^{-rT} P₂

        where  P_j = ½ + (1/π) ∫₀^∞ Re[ e^{iφx} g_j(φ) / (iφ) ] dφ

        x = ln(S/K) + (r−q)T   (log forward-moneyness)

        g₁(φ) = _cf_logreturn(φ − i)   (stock-measure CF, S·e^{(r-q)T} cancels)
        g₂(φ) = _cf_logreturn(φ)        (risk-neutral CF)
        """
        v0, kap, th, sv, rho_ = (
            self.v0, self.kappa, self.theta, self.sigma_v, self.rho,
        )
        x = np.log(S / K) + (r - q) * T

        def integrand_P1(phi):
            g = self._cf_logreturn(phi - 1j, T, v0, kap, th, sv, rho_)
            return np.real(np.exp(1j * phi * x) * g / (1j * phi))

        def integrand_P2(phi):
            g = self._cf_logreturn(phi, T, v0, kap, th, sv, rho_)
            return np.real(np.exp(1j * phi * x) * g / (1j * phi))

        limit_hi = 200.0
        P1 = 0.5 + (1.0 / np.pi) * quad(integrand_P1, 1e-8, limit_hi, limit=200)[0]
        P2 = 0.5 + (1.0 / np.pi) * quad(integrand_P2, 1e-8, limit_hi, limit=200)[0]

        call = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
        if otype == "put":
            return call - S * np.exp(-q * T) + K * np.exp(-r * T)
        return call

    def price_european_vec(self, S, Ks, Ts, r, q=0.0, otypes=None):
        """Vectorised (loop) pricing for a batch of strikes/expiries."""
        if otypes is None:
            otypes = ["call"] * len(Ks)
        return np.array([
            self.price_european(S, K, T, r, q, ot)
            for K, T, ot in zip(Ks, Ts, otypes)
        ])

    # ═══════════════════════════════════════════════════════════
    #  CALIBRATION  MODE 1 — from live options chain
    # ═══════════════════════════════════════════════════════════
    def calibrate_from_chain(self, chain_df, S, r, q=0.0,
                             moneyness_band=0.15, min_price=0.10):
        """
        Fit (v0, κ, θ, σᵥ, ρ) to a cross-section of market option prices.

        Parameters
        ----------
        chain_df : DataFrame
            Must contain columns: strike, lastPrice (or mid), impliedVolatility,
            optionType ('C'/'P'), expiration (datetime or str).
        S : float    current spot
        r : float    risk-free rate
        q : float    dividend yield
        """
        df = chain_df.copy()

        # ── normalise column names ──
        col_map = {
            "Strike": "strike", "strike": "strike",
            "lastPrice": "mid", "Last Price": "mid",
            "impliedVolatility": "iv", "Implied Volatility": "iv",
            "optionType": "otype",
            "expiration": "expiry", "Expiration": "expiry",
        }
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns},
                  inplace=True)

        # mid from bid/ask if available
        if "bid" in df.columns and "ask" in df.columns:
            df["mid"] = 0.5 * (df["bid"] + df["ask"])
        elif "mid" not in df.columns and "lastPrice" in df.columns:
            df["mid"] = df["lastPrice"]

        # parse expiry → T
        if "expiry" in df.columns:
            df["expiry"] = pd.to_datetime(df["expiry"])
            today = pd.Timestamp.now().normalize()
            df["T"] = (df["expiry"] - today).dt.days / 365.0
        elif "T" not in df.columns:
            raise ValueError("Chain must have 'expiration' or 'T' column.")

        # filter
        df = df[(df["T"] > 5 / 365) & (df["T"] < 1.5)]
        df = df[df["mid"] > min_price]
        lo, hi = S * (1 - moneyness_band), S * (1 + moneyness_band)
        df = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
        if len(df) < 5:
            raise ValueError(f"Only {len(df)} options after filtering — need ≥5.")

        Ks = df["strike"].values.astype(float)
        Ts = df["T"].values.astype(float)
        mids = df["mid"].values.astype(float)
        otypes = ["call" if o == "C" else "put" for o in df["otype"].values]

        # vega weights (BS vega at market IV)
        ivs = df["iv"].values.astype(float) if "iv" in df.columns else np.full(len(df), 0.2)
        vegas = np.array([
            max(bs_vega(S, K, T, r, max(iv, 0.05)), 0.01)
            for K, T, iv in zip(Ks, Ts, ivs)
        ])
        w = vegas / vegas.sum()

        def obj(p):
            v0_, kap_, th_, sv_, rho_ = p
            old = (self.v0, self.kappa, self.theta, self.sigma_v, self.rho)
            self.v0, self.kappa, self.theta, self.sigma_v, self.rho = (
                v0_, kap_, th_, sv_, rho_,
            )
            try:
                model_px = self.price_european_vec(S, Ks, Ts, r, q, otypes)
            except Exception:
                self.v0, self.kappa, self.theta, self.sigma_v, self.rho = old
                return 1e12
            self.v0, self.kappa, self.theta, self.sigma_v, self.rho = old
            err = (model_px - mids) ** 2 * w
            return float(np.sum(err))

        bounds = [
            (1e-4, 1.0),     # v0
            (0.1, 30.0),     # kappa
            (1e-4, 1.0),     # theta
            (0.05, 2.0),     # sigma_v
            (-0.99, 0.0),    # rho  (equity → negative)
        ]
        x0 = [0.04, 2.0, 0.04, 0.3, -0.7]

        best = minimize(obj, x0, bounds=bounds, method="L-BFGS-B",
                        options={"maxiter": 200, "ftol": 1e-10})

        # optional: DE global search fallback
        if best.fun > 0.1 * len(Ks):
            try:
                de = differential_evolution(obj, bounds, maxiter=80, seed=42,
                                            tol=1e-8, polish=True)
                if de.fun < best.fun:
                    best = de
            except Exception:
                pass

        self.v0, self.kappa, self.theta, self.sigma_v, self.rho = best.x
        self._fitted = True
        return self._summary()

    # ═══════════════════════════════════════════════════════════
    #  CALIBRATION  MODE 2 — from returns + VIX proxy (backtest)
    # ═══════════════════════════════════════════════════════════
    def calibrate_from_returns(self, returns, vix_proxy, dt=1.0 / 252):
        """
        Approximate MLE using Euler discretisation.

        We treat VIX9D as an observable proxy for √v, so:
            v_t ≈ (VIX9D_t / 100)²

        The Euler transition densities give a tractable likelihood:
            r_t | v_t  ~ N(0, v_t·Δt)
            v_{t+1} | v_t ~ N(v_t + κ(θ−v_t)Δt,  σᵥ²·v_t·Δt)

        Parameters
        ----------
        returns : 1-D array   — log returns
        vix_proxy : 1-D array — VIX9D close (percentage, e.g. 15.0)
        dt : float             — time step
        """
        r = np.asarray(returns, dtype=np.float64)
        vix = np.asarray(vix_proxy, dtype=np.float64)
        v_obs = (vix / 100.0) ** 2   # annualised variance proxy
        N = min(len(r), len(v_obs)) - 1

        rt = r[:N]
        vt = np.maximum(v_obs[:N], 1e-8)      # v[0]…v[N-1]
        vt1 = np.maximum(v_obs[1:N + 1], 1e-8)  # v[1]…v[N]

        # pre-compute constants
        _ln2pi = np.log(2.0 * np.pi)
        rt2 = rt ** 2

        def nll(p):
            kap, th, sv, rho_ = p
            sv2 = sv * sv

            # return likelihood: r_t ~ N(0, vt*dt)
            var_r = vt * dt
            ll_r = -0.5 * (_ln2pi + np.log(var_r) + rt2 / var_r)

            # variance transition likelihood
            mu_v = vt + kap * (th - vt) * dt
            var_v = np.maximum(sv2 * vt * dt, 1e-14)
            resid_v = vt1 - mu_v
            ll_v = -0.5 * (_ln2pi + np.log(var_v) + resid_v ** 2 / var_v)

            # bivariate normal correlation correction
            rho_eff = np.clip(rho_ * sv * vt * dt / np.sqrt(var_r * var_v + 1e-30),
                              -0.999, 0.999)
            z1 = rt / np.sqrt(var_r)
            z2 = resid_v / np.sqrt(var_v)
            denom = 1.0 - rho_eff ** 2
            denom = np.maximum(denom, 1e-6)
            ll_corr = (-0.5 * np.log(denom)
                       + rho_eff * z1 * z2 / denom
                       - 0.5 * rho_eff ** 2 * (z1 ** 2 + z2 ** 2) / denom)

            total = np.sum(ll_r) + np.sum(ll_v) + np.sum(ll_corr)
            return -total if np.isfinite(total) else 1e15

        bounds = [
            (0.1, 30.0),     # kappa — VIX-proxy mean-reversion can be fast
            (1e-4, 0.50),    # theta
            (0.05, 2.0),     # sigma_v
            (-0.99, -0.01),  # rho
        ]
        seeds = [
            [2.0, 0.04, 0.30, -0.70],
            [5.0, 0.06, 0.50, -0.50],
            [10.0, 0.03, 0.20, -0.80],
            [15.0, 0.05, 0.40, -0.40],
        ]

        best = None
        for s in seeds:
            try:
                res = minimize(nll, s, bounds=bounds, method="L-BFGS-B",
                               options={"maxiter": 300, "ftol": 1e-10})
                if best is None or res.fun < best.fun:
                    best = res
            except Exception:
                continue

        if best is None or not np.isfinite(best.fun):
            raise RuntimeError("Heston returns-based calibration failed.")

        self.kappa, self.theta, self.sigma_v, self.rho = best.x
        self.v0 = float(v_obs[-1])   # current variance from latest VIX9D
        self._fitted = True
        return self._summary()

    def _summary(self):
        return {
            "v0": round(self.v0, 6),
            "kappa": round(self.kappa, 4),
            "theta": round(self.theta, 6),
            "sigma_v": round(self.sigma_v, 4),
            "rho": round(self.rho, 4),
            "feller": 2 * self.kappa * self.theta > self.sigma_v ** 2,
            "long_run_vol": round(np.sqrt(self.theta) * _SQ252 if self.theta > 0 else 0, 4),
            "inst_vol": round(np.sqrt(max(self.v0, 0)) * _SQ252, 4),
        }

    # ═══════════════════════════════════════════════════════════
    #  FORECAST  — Andersen QE Monte-Carlo
    # ═══════════════════════════════════════════════════════════
    def forecast_density(self, horizon=5, n_sims=3000, seed=0):
        """
        Simulate Heston paths forward *horizon* days using the
        Quadratic-Exponential (QE) scheme (Andersen 2008).

        Returns
        -------
        1-D array of length n_sims: annualised realised vol for each path.
        """
        rng = np.random.default_rng(seed)
        dt = 1.0 / 252.0
        kap, th, sv, rho_ = self.kappa, self.theta, self.sigma_v, self.rho
        v0 = max(self.v0, 1e-8)

        # pre-compute QE constants
        exp_kdt = np.exp(-kap * dt)
        k1 = dt * 0.5 * (kap * rho_ / sv - 0.5) - rho_ / sv
        k2 = dt * 0.5 * (kap * rho_ / sv - 0.5) + rho_ / sv
        k3 = dt * 0.5 * (1.0 - rho_ ** 2)

        out = np.empty(n_sims)
        psi_c = 1.5   # QE switching threshold

        for i in range(n_sims):
            v = v0
            cum_var = 0.0
            for _ in range(horizon):
                # ── QE step for v ──
                m = th + (v - th) * exp_kdt
                s2 = (v * sv ** 2 * exp_kdt / kap * (1 - exp_kdt)
                      + th * sv ** 2 / (2 * kap) * (1 - exp_kdt) ** 2)
                s2 = max(s2, 1e-14)
                psi = s2 / (m * m) if m > 1e-12 else 10.0

                if psi <= psi_c:
                    # moment-matched shifted Gaussian
                    b2 = 2.0 / psi - 1.0 + np.sqrt(2.0 / psi) * np.sqrt(max(2.0 / psi - 1.0, 0.0))
                    a_ = m / (1.0 + b2)
                    v_next = a_ * (np.sqrt(b2) + rng.standard_normal()) ** 2
                else:
                    # exponential approximation
                    p_ = (psi - 1.0) / (psi + 1.0)
                    beta_ = (1.0 - p_) / max(m, 1e-12)
                    u_ = rng.random()
                    if u_ <= p_:
                        v_next = 0.0
                    else:
                        v_next = np.log((1.0 - p_) / max(1.0 - u_, 1e-12)) / beta_

                v_next = max(v_next, 0.0)
                cum_var += max(v, 0.0) * dt  # accumulate daily variance
                v = v_next

            # annualised realised vol for this path
            avg_var = cum_var / (horizon * dt)  # average annualised variance
            out[i] = np.sqrt(max(avg_var, 0.0))

        return out

    @property
    def inst_vol(self):
        return np.sqrt(max(self.v0, 0.0))

    @property
    def long_run_vol(self):
        return np.sqrt(max(self.theta, 0.0))


# ═══════════════════════════════════════════════════════════════
#  REGIME DETECTION  (simple variance-ratio approach)
#  — replaces MS-GARCH regime indicator —
#  Keeps same interface: regime ∈ {0, 1}, prob_turb ∈ [0,1]
# ═══════════════════════════════════════════════════════════════
def detect_regime(v0, theta, rv_cc, rv_panel_median, lookback_vols=None):
    """
    Lightweight turbulence indicator derived from Heston state.

    Turbulence signals:
    • v0 >> θ  (inst vol well above long-run)
    • RV elevated relative to recent history
    • Vol-of-vol implied by Heston is high
    """
    # ratio of instantaneous to long-run variance
    vr = v0 / max(theta, 1e-8)

    # z-score of current RV vs panel median
    rv_z = (rv_cc - rv_panel_median) / max(rv_panel_median, 1e-4) if rv_panel_median > 0 else 0

    # sigmoid blend
    raw = 0.5 * (vr - 1.0) + 0.3 * rv_z
    prob = 1.0 / (1.0 + np.exp(-2.0 * raw))   # ∈ (0, 1)
    regime = int(prob > 0.50)
    return regime, float(np.clip(prob, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════
#  SIGNAL  (sell-only, multi-gate — same as original)
# ═══════════════════════════════════════════════════════════════
@dataclass
class Sig:
    d_mean: float
    d_med: float
    d_p10: float
    d_p90: float
    d_p95: float
    d_skew: float
    d_kurt: float
    d_iqr: float
    iv: float
    rv_cc: float
    rv_pk: float
    rv_rs: float
    rv_yz: float
    rv_bp: float
    vrp: float
    vrp_pct: float
    rv_cons: float
    regime: int
    prob_turb: float
    trade: str
    conf: float
    reason: str


def make_signal(
    den, iv, rcc, rpk, rrs, ryz, rbp, regime, prob_turb,
    vrp_strong=0.10, vrp_weak=0.05,
    tail_gate_p95=True, kurt_max=4.0, skew_max=0.5,
) -> Sig:
    dm = float(np.mean(den))
    dmed = float(np.median(den))
    p10 = float(np.percentile(den, 10))
    p90 = float(np.percentile(den, 90))
    p95 = float(np.percentile(den, 95))
    dsk = float(stats.skew(den))
    dku = float(stats.kurtosis(den))
    diq = float(np.percentile(den, 75) - np.percentile(den, 25))

    rvs = [rcc, rpk, rrs, ryz, rbp]
    rvc = float(np.median(rvs))
    vrp = iv - dm
    vpct = vrp / iv if iv > 0 else 0

    reasons = []
    sc = 0.0

    # G1 — VRP breadth
    if vpct > vrp_strong:
        sc += 0.30
        reasons.append(f"VRP={vpct:+.1%}")
    elif vpct > vrp_weak:
        sc += 0.15
        reasons.append(f"VRP(w)={vpct:+.1%}")

    # G2 — IV above RV panel consensus
    nb = sum(iv > r for r in rvs)
    if nb == 5:
        sc += 0.30
        reasons.append("IV>5RV")
    elif nb >= 4:
        sc += 0.15
        reasons.append(f"IV>{nb}RV")

    # G3 — Tail containment
    if tail_gate_p95:
        if p95 < iv:
            sc += 0.25
            reasons.append("p95<IV")
        elif p90 < iv:
            sc += 0.10
            reasons.append("p90<IV")

    # G4 — Kurtosis not extreme
    if dku < kurt_max:
        sc += 0.10
        reasons.append(f"k={dku:.1f}")

    # G5 — Skew not strongly positive
    if dsk < skew_max:
        sc += 0.05
        reasons.append(f"sk={dsk:.1f}")

    # G6 — Regime: penalise selling into turbulence
    if regime == 1 and prob_turb > 0.70:
        sc *= 0.60
        reasons.append("TURB_PENALTY")

    if sc >= 0.55:
        trade = "SELL_STRADDLE"
    elif sc >= 0.40:
        trade = "SELL_SMALL"
    else:
        trade = "NO_TRADE"

    return Sig(
        d_mean=round(dm, 5), d_med=round(dmed, 5),
        d_p10=round(p10, 5), d_p90=round(p90, 5), d_p95=round(p95, 5),
        d_skew=round(dsk, 4), d_kurt=round(dku, 4), d_iqr=round(diq, 5),
        iv=round(iv, 5),
        rv_cc=round(rcc, 5), rv_pk=round(rpk, 5), rv_rs=round(rrs, 5),
        rv_yz=round(ryz, 5), rv_bp=round(rbp, 5),
        vrp=round(vrp, 5), vrp_pct=round(vpct, 5), rv_cons=round(rvc, 5),
        regime=regime, prob_turb=round(prob_turb, 4),
        trade=trade, conf=round(min(sc, 1.0), 4),
        reason=" | ".join(reasons),
    )


# ═══════════════════════════════════════════════════════════════
#  HALF-KELLY + TAIL GATE  (unchanged)
# ═══════════════════════════════════════════════════════════════
def size_pos(sig, spot, capital, dte, r, max_risk=0.04):
    if sig.trade == "NO_TRADE":
        return 0, 0.0

    edge = max(sig.vrp_pct, 0.0)
    kh = edge * sig.conf * 0.50

    tr = sig.d_p95 / sig.iv if sig.iv > 0 else 1.0
    if tr > 0.95:
        td = 0.25
    elif tr > 0.85:
        td = 0.55
    else:
        td = 1.0

    kd = max(0.25, 1.0 - sig.d_kurt / 15.0)
    frac = kh * td * kd
    if sig.trade == "SELL_SMALL":
        frac *= 0.50

    risk = capital * min(frac, max_risk)
    prem = straddle_px(spot, spot, dte / 252, r, sig.iv)
    c1 = prem * 100
    if c1 < 0.05:
        return 0, 0.0

    n = max(int(risk / c1), 0)
    n = min(n, max(int(capital * 0.05 / c1), 0))
    return n, round(n * c1, 2)


# ═══════════════════════════════════════════════════════════════
#  P&L  (short straddle, BS reprice + gamma adjustment — unchanged)
# ═══════════════════════════════════════════════════════════════
_PARKINSON_SCALE = 1.0 / (4.0 * np.log(2))


def calc_pnl(
    trade, n, si, so, hi, lo, ivi, ivo, dte, r=0.05,
    slip_per_contract=0.0, spread_per_leg=0.00, max_loss_mult=2,
):
    if trade == "NO_TRADE" or n == 0:
        return 0.0

    K = si
    Ti = dte / 252
    pi = straddle_px(si, K, Ti, r, ivi)
    total_premium = pi * 100 * n

    # max-loss stop
    if max_loss_mult > 0 and np.isfinite(max_loss_mult) and total_premium > 0:
        worst_hi = straddle_px(hi, K, Ti, r, ivi)
        worst_lo = straddle_px(lo, K, Ti, r, ivi)
        worst_val = max(worst_hi, worst_lo)
        intraday_loss = (worst_val - pi) * 100 * n
        max_allowed = max_loss_mult * total_premium
        if intraday_loss > max_allowed:
            spread_cost = 4.0 * spread_per_leg * 100 * n
            slip_cost = slip_per_contract * n
            return round(-max_allowed - spread_cost - slip_cost, 2)

    To = max((dte - 1.0), 0.05) / 252
    po = straddle_px(so, K, To, r, ivo)
    mtm = (pi - po) * 100 * n

    # gamma cost (Parkinson-scaled)
    net_move_sq = (so - si) ** 2
    range_var = (hi - lo) ** 2 * _PARKINSON_SCALE
    excess_var = max(range_var - net_move_sq, 0.0)

    if excess_var > 0 and Ti > 1e-6:
        gam = bs_gamma(si, K, Ti, r, ivi)
        straddle_gamma = 2.0 * gam
        gamma_cost = 0.5 * straddle_gamma * excess_var * 100 * n
    else:
        gamma_cost = 0.0

    spread_cost = 4.0 * spread_per_leg * 100 * n
    slip_cost = slip_per_contract * n

    pnl = mtm - gamma_cost - spread_cost - slip_cost
    return round(pnl, 2)


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_data(years: int = None, start: str = None, end: str = None):
    if years is not None:
        end_time = end if end else datetime.now().strftime("%Y-%m-%d")
        start_time = (
            datetime.strptime(end_time, "%Y-%m-%d") - timedelta(days=years * 365)
        ).strftime("%Y-%m-%d")
    else:
        start_time, end_time = start, end
    if start_time is None:
        raise ValueError("Provide either 'years' or 'start'+'end'.")

    import yfinance as yf

    print("  Fetching SPY …")
    spy = yf.download("SPY", start=start_time, end=end_time,
                      progress=False, auto_adjust=True)
    print("  Fetching VIX9D …")
    v9d = yf.download("^VIX9D", start=start_time, end=end_time,
                      progress=False, auto_adjust=True)
    for df in (spy, v9d):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    ix = spy.index.intersection(v9d.index)
    spy, v9d = spy.loc[ix].copy(), v9d.loc[ix].copy()
    spy["ret"] = np.log(spy["Close"] / spy["Close"].shift(1))
    spy.dropna(inplace=True)
    v9d = v9d.loc[spy.index]
    print(f"  {len(spy)} days ({spy.index[0].date()} → {spy.index[-1].date()})")
    return spy, v9d


def load_live_chain(ticker="SPY"):
    """Fetch current options chain from yfinance for live Heston calibration."""
    import yfinance as yf
    tk = yf.Ticker(ticker)
    exps = tk.options
    if not exps:
        raise RuntimeError(f"No options expirations for {ticker}")

    rows = []
    today = pd.Timestamp.now().normalize()

    # grab first 3 expirations (short-dated)
    for exp_str in exps[:3]:
        exp_dt = pd.Timestamp(exp_str)
        dte_days = (exp_dt - today).days
        if dte_days < 2:
            continue
        T = dte_days / 365.0
        chain = tk.option_chain(exp_str)
        for otype, df in [("C", chain.calls), ("P", chain.puts)]:
            for _, row in df.iterrows():
                mid = 0.5 * (row.get("bid", 0) + row.get("ask", 0))
                if mid < 0.05:
                    continue
                rows.append({
                    "strike": float(row["strike"]),
                    "mid": mid,
                    "iv": float(row.get("impliedVolatility", 0.2)),
                    "otype": otype,
                    "T": T,
                    "volume": int(row.get("volume", 0) or 0),
                    "openInterest": int(row.get("openInterest", 0) or 0),
                })

    if not rows:
        raise RuntimeError("No valid options found in chain.")
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════
@dataclass
class Cfg:
    capital: float = 1_000_000
    lookback: int = 504
    refit: int = 5
    fc_horizon: int = 5
    mc_sims: int = 3000
    dte: float = 2.0
    rfr: float = 0.05
    max_risk: float = 0.04
    slip_per_contract: float = 0.0
    spread_per_leg: float = 0.00
    iv_haircut: float = 0.02
    max_loss_mult: float = 2.0
    vrp_strong: float = 0.10
    vrp_weak: float = 0.05
    kurt_max: float = 4.0
    skew_max: float = 0.5


def run_engine(spy, v9d, rvp, cfg, start_idx=None, end_idx=None, quiet=False):
    """Core backtest loop — Heston calibration from returns + VIX9D."""
    if start_idx is None:
        start_idx = cfg.lookback
    if end_idx is None:
        end_idx = len(spy)

    out = []
    mdl = None
    den = None
    last_fit = -cfg.refit
    cap = cfg.capital

    ret = spy["ret"].values
    opn = spy["Open"].values.astype(float)
    clo = spy["Close"].values.astype(float)
    hi  = spy["High"].values.astype(float)
    lo  = spy["Low"].values.astype(float)
    vxc = v9d["Close"].values.astype(float)
    idx = spy.index

    s = max(start_idx, cfg.lookback)
    e = min(end_idx, len(spy) - 2)
    tot = e - s
    t0 = time.time()

    if not quiet:
        print(
            f"\n  {tot} bars | refit/{cfg.refit}d | "
            f"{cfg.mc_sims} MC × {cfg.fc_horizon}d | ${cfg.capital:,.0f}\n"
        )

    for t in range(s, e):
        # ── periodic Heston refit ──
        if (t - last_fit) >= cfg.refit or mdl is None:
            r_win = ret[t - cfg.lookback : t]
            vix_win = vxc[t - cfg.lookback : t]

            # skip if VIX data has issues
            valid = ~np.isnan(vix_win) & (vix_win > 0)
            if valid.sum() < cfg.lookback * 0.8:
                continue

            try:
                mdl = HestonModel()
                summary = mdl.calibrate_from_returns(
                    r_win[valid[:len(r_win)]],
                    vix_win[valid],
                )
                # override v0 with latest VIX9D (no lookahead: t−1)
                mdl.v0 = (vxc[t - 1] / 100.0) ** 2
                den = mdl.forecast_density(cfg.fc_horizon, cfg.mc_sims, seed=t)
                last_fit = t
            except Exception:
                mdl = None
                den = None
                continue

        if mdl is None or den is None:
            continue

        # ── features: all from t−1 (no lookahead) ──
        raw_iv = vxc[t - 1] / 100.0
        if raw_iv < 0.01:
            continue
        iv = max(raw_iv - cfg.iv_haircut, 0.01)

        rv = rvp.iloc[t - 1]
        if pd.isna(rv["rv_cc"]):
            continue

        # regime detection from Heston state
        rv_med = float(np.median([
            rv["rv_cc"], rv["rv_pk"], rv["rv_rs"], rv["rv_yz"], rv["rv_bp"],
        ]))
        regime, prob_turb = detect_regime(
            mdl.v0, mdl.theta, float(rv["rv_cc"]), rv_med,
        )

        sig = make_signal(
            den, iv,
            float(rv["rv_cc"]), float(rv["rv_pk"]),
            float(rv["rv_rs"]), float(rv["rv_yz"]), float(rv["rv_bp"]),
            regime, prob_turb,
            cfg.vrp_strong, cfg.vrp_weak, True, cfg.kurt_max, cfg.skew_max,
        )

        # ── entry = Open[t],  exit = Close[t] ──
        si = opn[t]
        so = clo[t]
        day_hi = hi[t]
        day_lo = lo[t]
        ivo = max(vxc[t] / 100.0 - cfg.iv_haircut, 0.01)

        n, risk = size_pos(sig, si, cap, cfg.dte, cfg.rfr, cfg.max_risk)
        pnl = calc_pnl(
            sig.trade, n, si, so, day_hi, day_lo, iv, ivo,
            cfg.dte, cfg.rfr, cfg.slip_per_contract, cfg.spread_per_leg,
            cfg.max_loss_mult,
        )
        cap += pnl

        out.append({
            "date": idx[t], "trade": sig.trade, "conf": sig.conf,
            "n": n, "risk_usd": risk,
            "spot_in": round(si, 2), "spot_out": round(so, 2),
            "iv_in": round(iv, 4), "iv_out": round(ivo, 4),
            "fc_mean": sig.d_mean, "fc_p95": sig.d_p95,
            "fc_skew": sig.d_skew, "fc_kurt": sig.d_kurt,
            "rv_cons": sig.rv_cons, "vrp_pct": sig.vrp_pct,
            "regime": sig.regime, "prob_turb": sig.prob_turb,
            "heston_v0": round(mdl.v0, 6),
            "heston_kappa": round(mdl.kappa, 4),
            "heston_theta": round(mdl.theta, 6),
            "heston_sv": round(mdl.sigma_v, 4),
            "heston_rho": round(mdl.rho, 4),
            "pnl": pnl, "capital": round(cap, 2), "reason": sig.reason,
        })

        step = t - s
        if not quiet and step % 250 == 0:
            el = time.time() - t0
            print(
                f"    [{step / tot * 100:5.1f}%] {idx[t].date()} "
                f"cap=${cap:>14,.2f}  κ={mdl.kappa:.2f} θ={mdl.theta:.4f} "
                f"σᵥ={mdl.sigma_v:.2f} ρ={mdl.rho:.2f}  ({el:.0f}s)"
            )

    if not quiet:
        print(f"\n  Done {time.time() - t0:.1f}s")
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).set_index("date")


# ═══════════════════════════════════════════════════════════════
#  LIVE SIGNAL (calibrate from current options chain)
# ═══════════════════════════════════════════════════════════════
def run_live_signal(ticker="SPY", rfr=0.05, mc_sims=5000, fc_horizon=5):
    """
    One-shot live signal: calibrate Heston from current options chain,
    run MC forecast, produce signal.
    """
    import yfinance as yf

    print(f"\n{'='*60}")
    print(f"  LIVE HESTON SIGNAL — {ticker}")
    print(f"{'='*60}")

    # spot
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1y")
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    spot = float(hist["Close"].iloc[-1])
    print(f"  Spot: ${spot:.2f}")

    # options chain
    print("  Loading options chain …")
    chain = load_live_chain(ticker)
    print(f"  {len(chain)} option rows")

    # calibrate
    print("  Calibrating Heston from chain …")
    mdl = HestonModel()
    summary = mdl.calibrate_from_chain(chain, spot, rfr)
    print(f"  Heston params: {summary}")

    # forecast
    print(f"  Running {mc_sims} MC paths × {fc_horizon}d …")
    den = mdl.forecast_density(fc_horizon, mc_sims, seed=42)

    # RV panel (need OHLC history)
    rvp = build_rv_panel(hist)
    rv_last = rvp.iloc[-1]
    rcc = float(rv_last["rv_cc"]) if not pd.isna(rv_last["rv_cc"]) else 0.15
    rpk = float(rv_last["rv_pk"]) if not pd.isna(rv_last["rv_pk"]) else 0.15
    rrs = float(rv_last["rv_rs"]) if not pd.isna(rv_last["rv_rs"]) else 0.15
    ryz = float(rv_last["rv_yz"]) if not pd.isna(rv_last["rv_yz"]) else 0.15
    rbp = float(rv_last["rv_bp"]) if not pd.isna(rv_last["rv_bp"]) else 0.15

    # IV from Heston instantaneous vol
    iv = np.sqrt(mdl.v0)

    # regime
    rv_med = np.median([rcc, rpk, rrs, ryz, rbp])
    regime, prob_turb = detect_regime(mdl.v0, mdl.theta, rcc, rv_med)

    sig = make_signal(den, iv, rcc, rpk, rrs, ryz, rbp, regime, prob_turb)

    print(f"\n  Signal: {sig.trade}  (conf={sig.conf:.2f})")
    print(f"  Reason: {sig.reason}")
    print(f"  Forecast mean={sig.d_mean:.4f}  p95={sig.d_p95:.4f}  IV={sig.iv:.4f}")
    print(f"  VRP={sig.vrp_pct:+.1%}  Regime={'TURB' if regime else 'CALM'} ({prob_turb:.0%})")

    return sig, mdl, den


# ═══════════════════════════════════════════════════════════════
#  METRICS + REPORT
# ═══════════════════════════════════════════════════════════════
def calc_metrics(df, initial_cap):
    if df.empty:
        return {}
    tr = df[df["trade"] != "NO_TRADE"]
    nt = len(tr)
    tp = df["pnl"].sum()
    fin = df["capital"].iloc[-1]
    rp = (fin / initial_cap - 1) * 100
    yr = (df.index[-1] - df.index[0]).days / 365.25
    ca = ((fin / initial_cap) ** (1 / max(yr, 0.1)) - 1) * 100 if yr > 0 else 0

    cum = df["pnl"].cumsum()
    dd = cum - cum.cummax()
    mdd = dd.min()
    mddp = mdd / initial_cap * 100

    w = tr[tr["pnl"] > 0]
    l = tr[tr["pnl"] < 0]
    wr = len(w) / nt * 100 if nt else 0
    aw = w["pnl"].mean() if len(w) else 0
    al = l["pnl"].mean() if len(l) else 0
    pf = (
        abs(w["pnl"].sum() / l["pnl"].sum())
        if (len(l) and l["pnl"].sum() != 0)
        else float("inf")
    )

    dp = df["pnl"]
    sh = dp.mean() / dp.std() * np.sqrt(252) if dp.std() > 0 else 0
    dsd = dp[dp < 0].std() * _SQ252 if (dp < 0).any() else 1
    so = dp.mean() * 252 / dsd if dsd > 0 else 0

    return dict(
        total_pnl=tp, final_cap=fin, return_pct=rp, cagr=ca,
        max_dd=mdd, max_dd_pct=mddp, sharpe=sh, sortino=so,
        n_trades=nt, n_days=len(df), win_rate=wr,
        avg_win=aw, avg_loss=al, profit_factor=pf, years=yr,
    )


def format_report(m, label, cfg):
    if not m:
        return f"\n  {label}: no data.\n"
    return f"""
{'=' * 72}
  {label}
{'=' * 72}
  Model               : Heston Stochastic Volatility
  Period / Days        : {m.get('years', 0):.1f}y  ({m['n_days']:,} days)
  Capital              : ${cfg.capital:>14,.2f}  ->  ${m['final_cap']:>14,.2f}
  Return / CAGR        : {m['return_pct']:+.2f}%  /  {m['cagr']:+.2f}%

  Total P&L            : ${m['total_pnl']:>+14,.2f}
  Max Drawdown         : ${m['max_dd']:>+14,.2f}  ({m['max_dd_pct']:+.1f}%)
  Sharpe               :  {m['sharpe']:+.3f}
  Sortino              :  {m['sortino']:+.3f}

  Trades               : {m['n_trades']:,}
  Win Rate             :  {m['win_rate']:.1f}%
  Avg Win / Loss       : ${m['avg_win']:>+12,.2f}  /  ${m['avg_loss']:>+12,.2f}
  Profit Factor        :  {m['profit_factor']:.2f}

  Bid-Ask / Leg        : ${cfg.spread_per_leg:.2f}
  IV Haircut           :  {cfg.iv_haircut:.0%}
  Max-Loss Stop        :  {cfg.max_loss_mult:.1f}× premium
{'=' * 72}"""


# ═══════════════════════════════════════════════════════════════
#  STRESS TESTS
# ═══════════════════════════════════════════════════════════════
def stress_slippage(spy, v9d, rvp, base_cfg):
    print("\n" + "-" * 72)
    print("  SLIPPAGE & BID-ASK STRESS TEST")
    print("-" * 72)

    scenarios = [
        (0.0,  0.03, "Tight: $0 slip, $0.03/leg"),
        (0.0,  0.05, "Base:  $0 slip, $0.05/leg"),
        (2.0,  0.05, "Mod:   $2 slip, $0.05/leg"),
        (2.0,  0.08, "Wide:  $2 slip, $0.08/leg"),
        (5.0,  0.10, "Stress:$5 slip, $0.10/leg"),
        (5.0,  0.15, "Worst: $5 slip, $0.15/leg"),
    ]
    rows = []
    for sl, sp, label in scenarios:
        c = copy.copy(base_cfg)
        c.slip_per_contract = sl
        c.spread_per_leg = sp
        df = run_engine(spy, v9d, rvp, c, quiet=True)
        m = calc_metrics(df, c.capital)
        if not m:
            continue
        rows.append({"scenario": label, "slip_$": sl, "spread_$": sp, **m})
        print(
            f"    {label:<30s}  P&L=${m['total_pnl']:>+14,.2f}  "
            f"Sharpe={m['sharpe']:+.3f}  WR={m['win_rate']:.1f}%  "
            f"PF={m['profit_factor']:.2f}"
        )
    return pd.DataFrame(rows)


def stress_params(spy, v9d, rvp, base_cfg):
    print("\n" + "-" * 72)
    print("  PARAMETER SENSITIVITY STRESS TEST")
    print("-" * 72)

    tests = [
        ("BASE",            {}),
        ("VRP_strong=.08",  {"vrp_strong": 0.08}),
        ("VRP_strong=.12",  {"vrp_strong": 0.12}),
        ("VRP_strong=.15",  {"vrp_strong": 0.15}),
        ("VRP_weak=.03",    {"vrp_weak": 0.03}),
        ("VRP_weak=.07",    {"vrp_weak": 0.07}),
        ("kurt_max=3.0",    {"kurt_max": 3.0}),
        ("kurt_max=5.0",    {"kurt_max": 5.0}),
        ("skew_max=0.3",    {"skew_max": 0.3}),
        ("skew_max=1.0",    {"skew_max": 1.0}),
        ("max_risk=.02",    {"max_risk": 0.02}),
        ("max_risk=.06",    {"max_risk": 0.06}),
        ("refit=3",         {"refit": 3}),
        ("refit=10",        {"refit": 10}),
        ("mc_sims=1500",    {"mc_sims": 1500}),
        ("mc_sims=5000",    {"mc_sims": 5000}),
        ("iv_haircut=.01",  {"iv_haircut": 0.01}),
        ("iv_haircut=.03",  {"iv_haircut": 0.03}),
        ("spread=.08",      {"spread_per_leg": 0.08}),
        ("spread=.12",      {"spread_per_leg": 0.12}),
        ("stop=1.5x",       {"max_loss_mult": 1.5}),
        ("stop=2.0x",       {"max_loss_mult": 2.0}),
        ("stop=3.0x",       {"max_loss_mult": 3.0}),
        ("stop=inf(none)",  {"max_loss_mult": float("inf")}),
        ("lookback=252",    {"lookback": 252}),
        ("lookback=756",    {"lookback": 756}),
    ]
    rows = []
    for label, overrides in tests:
        c = copy.copy(base_cfg)
        for k, v in overrides.items():
            setattr(c, k, v)
        df = run_engine(spy, v9d, rvp, c, quiet=True)
        m = calc_metrics(df, c.capital)
        if not m:
            continue
        rows.append({"param": label, **m})
        print(
            f"    {label:<20s}  P&L=${m['total_pnl']:>+14,.2f}  "
            f"Sharpe={m['sharpe']:+.3f}  WR={m['win_rate']:.1f}%  "
            f"CAGR={m['cagr']:+.1f}%  MaxDD={m['max_dd_pct']:+.1f}%"
        )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════
def make_charts(df, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(
            6, 1, figsize=(16, 20), sharex=True,
            gridspec_kw={"height_ratios": [3, 1.2, 1.2, 1, 1, 1]},
        )
        fig.suptitle(
            "Heston Stochastic-Vol  ·  Vol-Sell Strategy",
            fontsize=13, weight="bold", y=0.98,
        )

        # 1) equity curve
        cum = df["pnl"].cumsum() + 1_000_000
        ax[0].fill_between(cum.index, cum, 1e6, where=cum >= 1e6, alpha=0.2, color="#2E7D32")
        ax[0].fill_between(cum.index, cum, 1e6, where=cum < 1e6, alpha=0.2, color="#C62828")
        ax[0].plot(cum.index, cum, color="#1B5E20", lw=1.2)
        ax[0].axhline(1e6, color="grey", lw=0.5, ls="--")
        ax[0].set_ylabel("Portfolio ($)")
        ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax[0].grid(alpha=0.2)
        ax[0].set_title("Equity Curve", fontsize=10, loc="left")

        # 2) VRP
        ax[1].bar(df.index, df["vrp_pct"], width=1, alpha=0.5,
                  color=np.where(df["vrp_pct"] > 0, "#4CAF50", "#F44336"))
        act = df[df["trade"] != "NO_TRADE"]
        if len(act):
            ax[1].scatter(act.index, act["vrp_pct"], s=8, color="#0D47A1",
                          zorder=5, label="Trade")
        ax[1].axhline(0, color="grey", lw=0.5)
        ax[1].set_ylabel("VRP%")
        ax[1].legend(fontsize=7, loc="upper right")
        ax[1].grid(alpha=0.2)

        # 3) Forecast vs IV vs RV
        ax[2].plot(df.index, df["fc_mean"], label="Heston Forecast",
                   color="#FF9800", lw=0.9)
        ax[2].plot(df.index, df["iv_in"], label="IV (VIX9D adj.)",
                   color="#7B1FA2", lw=0.9, alpha=0.8)
        ax[2].plot(df.index, df["rv_cons"], label="RV Consensus",
                   color="#00796B", lw=0.7, alpha=0.7)
        ax[2].set_ylabel("Ann. Vol")
        ax[2].legend(fontsize=7, loc="upper right")
        ax[2].grid(alpha=0.2)

        # 4) Regime probability
        ax[3].fill_between(df.index, df["prob_turb"], alpha=0.5, color="#E53935")
        ax[3].axhline(0.5, color="grey", lw=0.5, ls="--")
        ax[3].set_ylabel("P(Turbulent)")
        ax[3].set_ylim(0, 1)
        ax[3].grid(alpha=0.2)

        # 5) Heston params (κ and σᵥ)
        if "heston_kappa" in df.columns:
            ax2 = ax[4].twinx()
            ax[4].plot(df.index, df["heston_kappa"], color="#1565C0", lw=0.7,
                       alpha=0.8, label="κ (left)")
            ax2.plot(df.index, df["heston_sv"], color="#E65100", lw=0.7,
                     alpha=0.8, label="σᵥ (right)")
            ax[4].set_ylabel("κ", color="#1565C0")
            ax2.set_ylabel("σᵥ", color="#E65100")
            ax[4].legend(fontsize=7, loc="upper left")
            ax2.legend(fontsize=7, loc="upper right")
        ax[4].grid(alpha=0.2)
        ax[4].set_title("Heston Parameters", fontsize=9, loc="left")

        # 6) daily pnl
        col = np.where(df["pnl"] > 0, "#2E7D32", "#C62828")
        ax[5].bar(df.index, df["pnl"], width=1, color=col, alpha=0.6)
        ax[5].axhline(0, color="grey", lw=0.5)
        ax[5].set_ylabel("Daily P&L ($)")
        ax[5].grid(alpha=0.2)
        ax[5].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()
        p = os.path.join(out_dir, "equity_curve.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Chart -> {p}")
    except Exception as e:
        print(f"  (chart skipped: {e})")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main(stress_test: bool = False, live: bool = False):
    cfg = Cfg(max_loss_mult=1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if live:
        run_live_signal("SPY", cfg.rfr, cfg.mc_sims, cfg.fc_horizon)
        return

    print("-> Loading market data")
    spy, v9d = load_data(years=10)
    rvp = build_rv_panel(spy)

    print("\n-> Running full backtest  (Heston SV)")
    df = run_engine(spy, v9d, rvp, cfg)
    if df.empty:
        print("  No results — check data or model convergence.")
        return

    m = calc_metrics(df, cfg.capital)
    rpt = format_report(m, "HESTON VOL-SELL  ·  FULL BACKTEST", cfg)
    print(rpt)

    if stress_test:
        print("\n-> Stress tests")
        slip_df = stress_slippage(spy, v9d, rvp, cfg)
        param_df = stress_params(spy, v9d, rvp, cfg)
    else:
        slip_df = pd.DataFrame()
        param_df = pd.DataFrame()

    print("\n  Saving outputs …")
    csv_path = os.path.join(OUTPUT_DIR, "backtest_results.csv")
    df.to_csv(csv_path)
    print(f"  CSV    -> {csv_path}")

    slip_path = os.path.join(OUTPUT_DIR, "stress_slippage.csv")
    slip_df.to_csv(slip_path, index=False)

    param_path = os.path.join(OUTPUT_DIR, "stress_params.csv")
    param_df.to_csv(param_path, index=False)

    full_text = "\n\n".join([
        rpt,
        "\nSLIPPAGE & BID-ASK STRESS TEST\n" + slip_df.to_string(index=False) if not slip_df.empty else "",
        "\nPARAMETER SENSITIVITY STRESS TEST\n" + param_df.to_string(index=False) if not param_df.empty else "",
    ])
    report_path = os.path.join(OUTPUT_DIR, "full_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"  Report -> {report_path}")

    make_charts(df, OUTPUT_DIR)
    print("\n  All done.\n")


if __name__ == "__main__":
    import sys
    stress = "--stress" in sys.argv
    live = "--live" in sys.argv
    main(stress_test=stress, live=live)

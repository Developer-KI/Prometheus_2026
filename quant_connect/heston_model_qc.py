"""
heston_model.py
═══════════════
Pure model layer — no QuantConnect or broker dependencies.

Contains
────────
• Black-Scholes helpers  (pricing, greeks, IV solver)
• RVolPanel              (5 realised-vol estimators, rolling OHLC window)
• HestonModel            (CF pricing, chain calibration, returns calibration,
                          Andersen-QE Monte-Carlo forecast)
• detect_regime          (variance-ratio turbulence indicator)
• Signal / make_signal   (multi-gate sell signal)
• compute_n_contracts    (half-Kelly sizer with tail & kurtosis dampeners)
"""

import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
from scipy.stats import norm, skew as sp_skew, kurtosis as sp_kurtosis
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

# ─────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────
_SQ252 = np.sqrt(252)
_ncdf  = norm.cdf
_npdf  = norm.pdf


# ═══════════════════════════════════════════════════════════════════════
#  BLACK-SCHOLES HELPERS
# ═══════════════════════════════════════════════════════════════════════
def _d1d2(S: float, K: float, T: float, r: float, v: float):
    T = max(T, 1e-10)
    sq = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * v * v) * T) / (v * sq)
    return d1, d1 - v * sq


def bs_call(S: float, K: float, T: float, r: float, v: float) -> float:
    d1, d2 = _d1d2(S, K, T, r, v)
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def bs_put(S: float, K: float, T: float, r: float, v: float) -> float:
    d1, d2 = _d1d2(S, K, T, r, v)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def bs_vega(S: float, K: float, T: float, r: float, v: float) -> float:
    """Dollar vega: dPrice/dσ."""
    T = max(T, 1e-10)
    d1, _ = _d1d2(S, K, T, r, v)
    return S * _npdf(d1) * math.sqrt(T)


def bs_gamma(S: float, K: float, T: float, r: float, v: float) -> float:
    T = max(T, 1e-10)
    d1, _ = _d1d2(S, K, T, r, v)
    return _npdf(d1) / (S * v * math.sqrt(T))


def straddle_px(S: float, K: float, T: float, r: float, v: float) -> float:
    return bs_call(S, K, T, r, v) + bs_put(S, K, T, r, v)


def implied_vol(
    price: float, S: float, K: float, T: float, r: float,
    otype: str = "call", tol: float = 1e-6, maxiter: int = 50,
) -> float:
    """Newton-Raphson implied-vol solver."""
    sig = 0.20
    for _ in range(maxiter):
        px = bs_call(S, K, T, r, sig) if otype == "call" else bs_put(S, K, T, r, sig)
        diff = px - price
        vg = bs_vega(S, K, T, r, sig)
        if abs(vg) < 1e-12:
            break
        sig -= diff / vg
        sig = max(0.01, min(sig, 5.0))
        if abs(diff) < tol:
            break
    return sig


# ═══════════════════════════════════════════════════════════════════════
#  REALISED VOL PANEL  (rolling numpy, no pandas dependency)
# ═══════════════════════════════════════════════════════════════════════
class RVolPanel:
    """Maintains a rolling OHLC window and computes 5 RV estimators."""

    def __init__(self, window: int = 22):
        self.w = window
        self.closes = deque(maxlen=window + 1)
        self.opens  = deque(maxlen=window)
        self.highs  = deque(maxlen=window)
        self.lows   = deque(maxlen=window)

    @property
    def ready(self) -> bool:
        return len(self.closes) > self.w

    def update(self, o: float, h: float, l: float, c: float):
        self.opens.append(o)
        self.highs.append(h)
        self.lows.append(l)
        self.closes.append(c)

    def _log_returns(self):
        c = np.array(self.closes)
        return np.log(c[1:] / c[:-1])

    def rv_cc(self) -> float:
        return float(np.std(self._log_returns()) * _SQ252)

    def rv_parkinson(self) -> float:
        h, l = np.array(self.highs), np.array(self.lows)
        return float(np.sqrt(np.mean(np.log(h / l) ** 2 / (4 * np.log(2))) * 252))

    def rv_rogers_satchell(self) -> float:
        o = np.array(self.opens)
        h, l = np.array(self.highs), np.array(self.lows)
        c = np.array(list(self.closes)[-self.w:])
        lho, llo, lco = np.log(h / o), np.log(l / o), np.log(c / o)
        return float(np.sqrt(np.mean(lho * (lho - lco) + llo * (llo - lco)) * 252))

    def rv_yang_zhang(self) -> float:
        c_arr = np.array(self.closes)
        o = np.array(self.opens)
        h, l = np.array(self.highs), np.array(self.lows)
        c_prev = c_arr[:-1]
        w = self.w
        k = 0.34 / (1.34 + (w + 1) / (w - 1))
        loc_ = np.log(o / c_prev[-len(o):])
        lco = np.log(c_arr[1:][-len(o):] / o)
        lho, llo = np.log(h / o), np.log(l / o)
        rs = lho * (lho - lco) + llo * (llo - lco)
        v = np.var(loc_) + k * np.var(lco) + (1 - k) * np.mean(rs)
        return float(np.sqrt(max(v, 0) * 252))


    def rv_bipower(self) -> float:
        lr = self._log_returns()
        mu1 = np.sqrt(2 / np.pi)
        bp = np.mean(np.abs(lr[1:]) * np.abs(lr[:-1])) / (mu1 ** 2)
        return float(np.sqrt(bp * 252))

    def all_rv(self) -> Dict[str, float]:
        """Return dict with all 5 estimators, or empty dict if not ready."""
        if not self.ready:
            return {}
        return {
            "rv_cc": self.rv_cc(),
            "rv_pk": self.rv_parkinson(),
            "rv_rs": self.rv_rogers_satchell(),
            "rv_yz": self.rv_yang_zhang(),
            "rv_bp": self.rv_bipower(),
        }


# ═══════════════════════════════════════════════════════════════════════
#  HESTON STOCHASTIC-VOLATILITY MODEL
# ═══════════════════════════════════════════════════════════════════════
#  dS/S  = (r − q) dt + √v dW₁
#  dv    = κ(θ − v) dt + σ_v √v dW₂
#  corr(dW₁, dW₂) = ρ
#
#  Parameters: v0  (instantaneous variance)
#              κ   (mean-reversion speed)
#              θ   (long-run variance)
#              σᵥ  (vol-of-vol)
#              ρ   (Wiener correlation, typically negative for equities)
# ═══════════════════════════════════════════════════════════════════════
class HestonModel:

    def __init__(self):
        self.v0: float       = 0.04
        self.kappa: float    = 2.0
        self.theta: float    = 0.04
        self.sigma_v: float  = 0.3
        self.rho: float      = -0.7
        self._fitted: bool   = False

    # ─── characteristic function (Albrecher / "little Heston trap") ──
    @staticmethod
    def _cf(u, T, v0, kappa, theta, sigma_v, rho):
        """
        CF of the centred log-return E[exp(iu·(ln S_T/S − (r−q)T))].
        Branch-cut-safe formulation from Albrecher et al.
        """
        iu  = 1j * u
        sv2 = sigma_v * sigma_v
        beta = kappa - rho * sigma_v * iu
        d    = np.sqrt(beta * beta + sv2 * (iu + u * u))
        g    = (beta - d) / (beta + d)
        edt  = np.exp(-d * T)

        C = (kappa * theta / sv2) * (
            (beta - d) * T - 2.0 * np.log((1.0 - g * edt) / (1.0 - g))
        )
        D = ((beta - d) / sv2) * (1.0 - edt) / (1.0 - g * edt)
        return np.exp(C + D * v0)

    # ─── Gil-Pelaez option pricing ──────────────────────────────────
    def price_call(self, S: float, K: float, T: float,
                   r: float, q: float = 0.0) -> float:
        v0, kap, th, sv, rho_ = (
            self.v0, self.kappa, self.theta, self.sigma_v, self.rho,
        )
        x = math.log(S / K) + (r - q) * T

        def integrand_P1(phi):
            g = self._cf(phi - 1j, T, v0, kap, th, sv, rho_)
            return np.real(np.exp(1j * phi * x) * g / (1j * phi))

        def integrand_P2(phi):
            g = self._cf(phi, T, v0, kap, th, sv, rho_)
            return np.real(np.exp(1j * phi * x) * g / (1j * phi))

        P1 = 0.5 + (1.0 / np.pi) * quad(integrand_P1, 1e-8, 200, limit=200)[0]
        P2 = 0.5 + (1.0 / np.pi) * quad(integrand_P2, 1e-8, 200, limit=200)[0]
        return S * math.exp(-q * T) * P1 - K * math.exp(-r * T) * P2

    def price_put(self, S: float, K: float, T: float,
                  r: float, q: float = 0.0) -> float:
        c = self.price_call(S, K, T, r, q)
        return c - S * math.exp(-q * T) + K * math.exp(-r * T)

    def price(self, S: float, K: float, T: float,
              r: float, q: float = 0.0, otype: str = "call") -> float:
        return self.price_call(S, K, T, r, q) if otype == "call" \
               else self.price_put(S, K, T, r, q)

    # ═══════════════════════════════════════════════════════════════
    #  CALIBRATION  MODE 1 — options chain cross-section
    # ═══════════════════════════════════════════════════════════════
    def calibrate_from_chain(
        self,
        strikes: np.ndarray,
        expiries_T: np.ndarray,
        mid_prices: np.ndarray,
        otypes: list,
        ivs: np.ndarray,
        S: float,
        r: float,
        q: float = 0.0,
    ) -> dict:
        """
        Fit (v0, κ, θ, σᵥ, ρ) to market prices via vega-weighted
        least-squares.  Falls back to differential evolution when
        L-BFGS-B converges poorly.

        Parameters
        ----------
        strikes, expiries_T, mid_prices : 1-D arrays
        otypes : list of 'call' / 'put'
        ivs    : 1-D array of Black-Scholes implied vols (for vega weights)
        S, r, q: spot, risk-free rate, dividend yield
        """
        Ks   = np.asarray(strikes, dtype=np.float64)
        Ts   = np.asarray(expiries_T, dtype=np.float64)
        mids = np.asarray(mid_prices, dtype=np.float64)
        iv_arr = np.asarray(ivs, dtype=np.float64)

        # vega weights
        vegas = np.array([
            max(bs_vega(S, K, T, r, max(iv, 0.05)), 0.01)
            for K, T, iv in zip(Ks, Ts, iv_arr)
        ])
        w = vegas / vegas.sum()

        def obj(p):
            v0_, kap_, th_, sv_, rho_ = p
            old = (self.v0, self.kappa, self.theta, self.sigma_v, self.rho)
            self.v0, self.kappa, self.theta, self.sigma_v, self.rho = (
                v0_, kap_, th_, sv_, rho_,
            )
            try:
                model_px = np.array([
                    self.price(S, K, T, r, q, ot)
                    for K, T, ot in zip(Ks, Ts, otypes)
                ])
            except Exception:
                self.v0, self.kappa, self.theta, self.sigma_v, self.rho = old
                return 1e12
            self.v0, self.kappa, self.theta, self.sigma_v, self.rho = old
            return float(np.sum((model_px - mids) ** 2 * w))

        bounds = [
            (1e-4, 1.0),     # v0
            (0.1,  30.0),    # kappa
            (1e-4, 1.0),     # theta
            (0.05, 2.0),     # sigma_v
            (-0.99, 0.0),    # rho
        ]
        x0 = [
            max(float(iv_arr.mean()) ** 2, 0.005),
            3.0,
            max(float(iv_arr.mean()) ** 2, 0.005),
            0.4,
            -0.65,
        ]

        best = minimize(obj, x0, bounds=bounds, method="L-BFGS-B",
                        options={"maxiter": 200, "ftol": 1e-10})

        # DE fallback for poor local fit
        if best.fun > 0.1 * len(Ks):
            try:
                de = differential_evolution(obj, bounds, maxiter=60, seed=42,
                                            tol=1e-8, polish=True)
                if de.fun < best.fun:
                    best = de
            except Exception:
                pass

        self.v0, self.kappa, self.theta, self.sigma_v, self.rho = best.x
        self._fitted = True
        return self._summary()

    # ═══════════════════════════════════════════════════════════════
    #  CALIBRATION  MODE 2 — returns + variance proxy  (backtest)
    # ═══════════════════════════════════════════════════════════════
    def calibrate_from_returns(
        self,
        returns: np.ndarray,
        var_proxy: np.ndarray,
        dt: float = 1.0 / 252,
    ) -> dict:
        """
        Approximate MLE with Euler discretisation of the Heston SDE.

        Parameters
        ----------
        returns   : 1-D array of log returns
        var_proxy : 1-D array of annualised variance,
                    e.g. (VIX / 100)².  Length ≥ len(returns) + 1.
        dt        : time step (default 1/252)
        """
        rt  = np.asarray(returns, dtype=np.float64)
        vp  = np.asarray(var_proxy, dtype=np.float64)
        vt  = np.maximum(vp[:-1], 1e-8)
        vt1 = np.maximum(vp[1:],  1e-8)
        N   = min(len(rt), len(vt))
        rt, vt, vt1 = rt[:N], vt[:N], vt1[:N]

        _ln2pi = np.log(2.0 * np.pi)
        rt2 = rt ** 2

        def nll(p):
            kap, th, sv, rho_ = p
            var_r   = vt * dt
            ll_r    = -0.5 * (_ln2pi + np.log(var_r) + rt2 / var_r)

            mu_v    = vt + kap * (th - vt) * dt
            var_v   = np.maximum(sv ** 2 * vt * dt, 1e-14)
            resid_v = vt1 - mu_v
            ll_v    = -0.5 * (_ln2pi + np.log(var_v) + resid_v ** 2 / var_v)

            rho_eff = np.clip(
                rho_ * sv * vt * dt / np.sqrt(var_r * var_v + 1e-30),
                -0.999, 0.999,
            )
            z1    = rt / np.sqrt(var_r)
            z2    = resid_v / np.sqrt(var_v)
            denom = np.maximum(1.0 - rho_eff ** 2, 1e-6)
            ll_corr = (
                -0.5 * np.log(denom)
                + rho_eff * z1 * z2 / denom
                - 0.5 * rho_eff ** 2 * (z1 ** 2 + z2 ** 2) / denom
            )
            total = np.sum(ll_r) + np.sum(ll_v) + np.sum(ll_corr)
            return -total if np.isfinite(total) else 1e15

        bounds = [
            (0.1,  30.0),     # kappa
            (1e-4, 0.50),     # theta
            (0.05, 2.0),      # sigma_v
            (-0.99, -0.01),   # rho
        ]
        seeds = [
            [2.0,  0.04, 0.30, -0.70],
            [5.0,  0.06, 0.50, -0.50],
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
            raise RuntimeError("Heston returns-calibration failed")

        self.kappa, self.theta, self.sigma_v, self.rho = best.x
        self.v0 = float(vp[-1])
        self._fitted = True
        return self._summary()

    # ─── summary dict ──────────────────────────────────────────────
    def _summary(self) -> dict:
        return dict(
            v0=round(self.v0, 6),
            kappa=round(self.kappa, 4),
            theta=round(self.theta, 6),
            sigma_v=round(self.sigma_v, 4),
            rho=round(self.rho, 4),
            feller=2 * self.kappa * self.theta > self.sigma_v ** 2,
        )

    # ═══════════════════════════════════════════════════════════════
    #  FORECAST  — Andersen QE Monte-Carlo
    # ═══════════════════════════════════════════════════════════════
    def forecast_density(
        self, horizon: int = 5, n_sims: int = 3000, seed: int = 0,
    ) -> np.ndarray:
        """
        Simulate variance paths forward *horizon* trading days using the
        Quadratic-Exponential (QE) scheme (Andersen 2008).

        Returns
        -------
        1-D array of length n_sims: annualised realised vol per path.
        """
        rng = np.random.default_rng(seed)
        dt  = 1.0 / 252.0
        kap, th, sv = self.kappa, self.theta, self.sigma_v
        v0  = max(self.v0, 1e-8)
        exp_kdt = math.exp(-kap * dt)
        psi_c   = 1.5     # QE switching threshold

        out = np.empty(n_sims)
        for i in range(n_sims):
            v = v0
            cum_var = 0.0
            for _ in range(horizon):
                # ── QE step for variance ──
                m  = th + (v - th) * exp_kdt
                s2 = max(
                    v * sv ** 2 * exp_kdt / kap * (1 - exp_kdt)
                    + th * sv ** 2 / (2 * kap) * (1 - exp_kdt) ** 2,
                    1e-14,
                )
                psi = s2 / (m * m) if m > 1e-12 else 10.0

                if psi <= psi_c:
                    b2 = 2.0 / psi - 1.0 + math.sqrt(2.0 / psi) * math.sqrt(
                        max(2.0 / psi - 1.0, 0.0)
                    )
                    a_ = m / (1.0 + b2)
                    v_next = a_ * (math.sqrt(b2) + rng.standard_normal()) ** 2
                else:
                    p_    = (psi - 1.0) / (psi + 1.0)
                    beta_ = (1.0 - p_) / max(m, 1e-12)
                    u_    = rng.random()
                    v_next = (
                        0.0 if u_ <= p_
                        else math.log((1.0 - p_) / max(1.0 - u_, 1e-12)) / beta_
                    )

                v_next = max(v_next, 0.0)
                cum_var += max(v, 0.0) * dt
                v = v_next

            out[i] = math.sqrt(max(cum_var / (horizon * dt), 0.0))
        return out


# ═══════════════════════════════════════════════════════════════════════
#  REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════
def detect_regime(
    v0: float, theta: float, rv_cc: float, rv_panel_median: float,
) -> tuple:
    """
    Lightweight turbulence indicator from Heston state + RV.

    Returns
    -------
    (regime, prob_turb) where regime ∈ {0, 1}.
    """
    vr   = v0 / max(theta, 1e-8)
    rv_z = ((rv_cc - rv_panel_median) / max(rv_panel_median, 1e-4)
            if rv_panel_median > 0 else 0)
    raw  = 0.5 * (vr - 1.0) + 0.3 * rv_z
    prob = 1.0 / (1.0 + math.exp(-2.0 * raw))
    return int(prob > 0.5), float(np.clip(prob, 0, 1))


# ═══════════════════════════════════════════════════════════════════════
#  SIGNAL  (multi-gate, sell-only)
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class Signal:
    trade: str          # SELL_STRADDLE / SELL_SMALL / NO_TRADE
    conf: float
    vrp_pct: float
    iv: float
    fc_mean: float
    fc_p95: float
    fc_skew: float
    fc_kurt: float
    rv_cons: float
    regime: int
    prob_turb: float
    reason: str


def make_signal(
    density: np.ndarray,
    iv: float,
    rv_dict: Dict[str, float],
    regime: int,
    prob_turb: float,
    vrp_strong: float = 0.10,
    vrp_weak: float   = 0.05,
    kurt_max: float   = 4.0,
    skew_max: float   = 0.5,
) -> Signal:
    """
    Score the forecast density against IV and realised-vol panel.

    Gates
    -----
    G1  VRP breadth      — implied above forecast mean
    G2  IV > RV panel    — implied above all 5 RV estimators
    G3  Tail containment — p95 of density below IV
    G4  Kurtosis cap     — forecast tails not too fat
    G5  Skew cap         — forecast not strongly right-skewed
    G6  Regime penalty   — dampen score in turbulent regime
    """
    den = density
    dm  = float(np.mean(den))
    p90 = float(np.percentile(den, 90))
    p95 = float(np.percentile(den, 95))
    dsk = float(sp_skew(den))
    dku = float(sp_kurtosis(den))

    rvs = [rv_dict["rv_cc"], rv_dict["rv_pk"], rv_dict["rv_rs"],
           rv_dict["rv_yz"], rv_dict["rv_bp"]]
    rvc  = float(np.median(rvs))
    vrp  = iv - dm
    vpct = vrp / iv if iv > 0 else 0

    reasons: list = []
    sc = 0.0

    # G1 — VRP breadth
    if vpct > vrp_strong:
        sc += 0.30;  reasons.append(f"VRP={vpct:+.1%}")
    elif vpct > vrp_weak:
        sc += 0.15;  reasons.append(f"VRP(w)={vpct:+.1%}")

    # G2 — IV above RV consensus
    nb = sum(iv > r for r in rvs)
    if nb == 5:
        sc += 0.30;  reasons.append("IV>5RV")
    elif nb >= 4:
        sc += 0.15;  reasons.append(f"IV>{nb}RV")

    # G3 — Tail containment
    if p95 < iv:
        sc += 0.25;  reasons.append("p95<IV")
    elif p90 < iv:
        sc += 0.10;  reasons.append("p90<IV")

    # G4 — Kurtosis
    if dku < kurt_max:
        sc += 0.10;  reasons.append(f"k={dku:.1f}")

    # G5 — Skew
    if dsk < skew_max:
        sc += 0.05;  reasons.append(f"sk={dsk:.1f}")

    # G6 — Regime penalty
    if regime == 1 and prob_turb > 0.70:
        sc *= 0.60;  reasons.append("TURB")

    if sc >= 0.55:
        trade = "SELL_STRADDLE"
    elif sc >= 0.40:
        trade = "SELL_SMALL"
    else:
        trade = "NO_TRADE"

    return Signal(
        trade=trade, conf=round(min(sc, 1.0), 4),
        vrp_pct=round(vpct, 5), iv=round(iv, 5),
        fc_mean=round(dm, 5), fc_p95=round(p95, 5),
        fc_skew=round(dsk, 4), fc_kurt=round(dku, 4),
        rv_cons=round(rvc, 5), regime=regime,
        prob_turb=round(prob_turb, 4), reason=" | ".join(reasons),
    )


# ═══════════════════════════════════════════════════════════════════════
#  HALF-KELLY SIZER
# ═══════════════════════════════════════════════════════════════════════
def compute_n_contracts(
    sig: Signal, spot: float, capital: float,
    dte: float, r: float, max_risk: float = 0.04,
) -> int:
    """
    Half-Kelly position size with tail-risk and kurtosis dampeners.

    Returns the number of straddle contracts to sell.
    """
    if sig.trade == "NO_TRADE":
        return 0

    edge = max(sig.vrp_pct, 0.0)
    kh   = edge * sig.conf * 0.50        # half-Kelly fraction

    # tail-risk dampener
    tr = sig.fc_p95 / sig.iv if sig.iv > 0 else 1.0
    td = 0.25 if tr > 0.95 else (0.55 if tr > 0.85 else 1.0)

    # kurtosis dampener
    kd = max(0.25, 1.0 - sig.fc_kurt / 15.0)

    frac = kh * td * kd
    if sig.trade == "SELL_SMALL":
        frac *= 0.50

    risk = capital * min(frac, max_risk)
    prem = straddle_px(spot, spot, max(dte, 0.5) / 252, r, sig.iv)
    c1   = prem * 100
    if c1 < 0.05:
        return 0

    n = max(int(risk / c1), 0)
    n = min(n, max(int(capital * 0.05 / c1), 0))
    return n
"""
MS-GARCH(1,1) Volatility Selling Strategy
=========================================================
2-regime (calm / turbulent) Markov-Switching GARCH(1,1)
5-day MC forecast density  ×  2-DTE ATM short straddles
VIX9D as IV proxy 
Half-Kelly + tail-risk gate

Robustness suite
────────────────
• Slippage & transaction cost sweep (with per-leg bid-ask)
• Parameter sensitivity stress test

No-Lookahead Protocol
─────────────────────
• GARCH fitted on returns[…t−1]
• RV panel uses OHLC[…t−1]
• IV = VIX9D close[t−1]
• Entry = Open[t]   Exit = Close[t]
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
from dataclasses import dataclass
import warnings, os, time, copy

warnings.filterwarnings("ignore")
_cdf = stats.norm.cdf
_pdf = stats.norm.pdf
_SQ252 = np.sqrt(252)
_ISQ2PI = 1.0 / np.sqrt(2 * np.pi)

# Output directory — sits next to this script, works on any OS
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# ═══════════════════════════════════════════════════════════════
#  Helper BLACK-SCHOLES
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
    """BS gamma — same for call and put."""
    T = max(T, 1e-10)
    d1, _ = _d1d2(S, K, T, r, v)
    return _pdf(d1) / (S * v * np.sqrt(T))


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
        lco2 = np.log(c / o)
        rs = lho * (lho - lco2) + llo * (llo - lco2)
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


def build_rv_panel(spy, window_size = 22):
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
#  MS-GARCH(1,1)  —  2 regimes (calm / turbulent)
#  Hamilton filter,  3 restarts × 300 iter
# ═══════════════════════════════════════════════════════════════
class MSGARCH:
    """
    Params: [p00, p11, w0, a0, b0, w1, a1, b1]  (8)
    Regime 0 = calm   h_t = w0 + a0*e2 + b0*h
    Regime 1 = turb   h_t = w1 + a1*e2 + b1*h
    """

    def __init__(self, returns: np.ndarray):
        self.r = np.ascontiguousarray(returns, dtype=np.float64)
        self.r2 = self.r ** 2
        self.T = len(returns)
        self.params = None
        self.xi_T = None  # filtered regime probs at T
        self._hT = None   # conditional variances at T

    def _nll(self, p):
        p00, p11 = p[0], p[1]
        w0, a0, b0 = p[2], p[3], p[4]
        w1, a1, b1 = p[5], p[6], p[7]
        P0_0 = p00
        P0_1 = 1 - p11
        P1_0 = 1 - p00
        P1_1 = p11

        r2 = self.r2
        T = self.T
        h0 = h1 = np.var(self.r[: min(30, T)])
        xi0 = xi1 = 0.5
        ll = 0.0

        for t in range(1, T):
            e2 = r2[t - 1]
            h0 = w0 + a0 * e2 + b0 * h0
            h1 = w1 + a1 * e2 + b1 * h1
            if h0 < 1e-12:
                h0 = 1e-12
            if h1 < 1e-12:
                h1 = 1e-12

            rt2 = r2[t]
            f0 = _ISQ2PI / (h0 ** 0.5) * np.exp(-0.5 * rt2 / h0)
            f1 = _ISQ2PI / (h1 ** 0.5) * np.exp(-0.5 * rt2 / h1)

            xp0 = P0_0 * xi0 + P0_1 * xi1
            xp1 = P1_0 * xi0 + P1_1 * xi1
            lk = xp0 * f0 + xp1 * f1
            if lk <= 0 or lk != lk:
                return 1e12
            ll += np.log(lk)
            inv = 1.0 / lk
            xi0 = xp0 * f0 * inv
            xi1 = xp1 * f1 * inv

        return -ll

    def fit(self):
        bnd = [
            (0.50, 0.999), (0.50, 0.999),                  # p00, p11
            (1e-8, 0.005), (0.01, 0.20), (0.70, 0.99),     # calm
            (1e-6, 0.020), (0.05, 0.50), (0.40, 0.97),     # turb
        ]
        seeds = [
            [0.97, 0.92, 5e-6, 0.04, 0.93, 1e-4, 0.14, 0.80],
            [0.95, 0.88, 1e-5, 0.06, 0.90, 2e-4, 0.20, 0.70],
            [0.98, 0.85, 2e-6, 0.03, 0.95, 5e-5, 0.10, 0.82],
        ]
        best = None
        for s in seeds:
            try:
                r = minimize(
                    self._nll, s, bounds=bnd, method="L-BFGS-B",
                    options={"maxiter": 300, "ftol": 1e-10},
                )
                if best is None or r.fun < best.fun:
                    best = r
            except Exception:
                continue
        if best is None or not np.isfinite(best.fun):
            raise RuntimeError("MS-GARCH fit failed")
        self.params = best.x
        self._filter()
        return self._summary()

    def _filter(self):
        p = self.params
        p00, p11 = p[0], p[1]
        w0, a0, b0 = p[2], p[3], p[4]
        w1, a1, b1 = p[5], p[6], p[7]
        P0_0 = p00
        P0_1 = 1 - p11
        P1_0 = 1 - p00
        P1_1 = p11

        h0 = h1 = np.var(self.r[: min(30, self.T)])
        xi0 = xi1 = 0.5
        for t in range(1, self.T):
            e2 = self.r2[t - 1]
            h0 = w0 + a0 * e2 + b0 * h0
            h1 = w1 + a1 * e2 + b1 * h1
            if h0 < 1e-12:
                h0 = 1e-12
            if h1 < 1e-12:
                h1 = 1e-12
            f0 = _ISQ2PI / (h0 ** 0.5) * np.exp(-0.5 * self.r2[t] / h0)
            f1 = _ISQ2PI / (h1 ** 0.5) * np.exp(-0.5 * self.r2[t] / h1)
            xp0 = P0_0 * xi0 + P0_1 * xi1
            xp1 = P1_0 * xi0 + P1_1 * xi1
            lk = xp0 * f0 + xp1 * f1 + 1e-15
            inv = 1.0 / lk
            xi0 = xp0 * f0 * inv
            xi1 = xp1 * f1 * inv
        self.xi_T = np.array([xi0, xi1])
        self._hT = np.array([h0, h1])

    def _summary(self):
        p = self.params
        return {
            "p00": p[0], "p11": p[1],
            "calm": (p[2], p[3], p[4]),
            "turb": (p[5], p[6], p[7]),
            "regime": int(self.xi_T[1] > 0.5),
            "prob_turb": float(self.xi_T[1]),
        }

    def forecast_density(self, horizon=5, n_sims=3000, seed=0):
        rng = np.random.default_rng(seed)
        p = self.params
        p00, p11 = p[0], p[1]
        w0, a0, b0 = p[2], p[3], p[4]
        w1, a1, b1 = p[5], p[6], p[7]
        P = np.array([[p00, 1 - p11], [1 - p00, p11]])

        z = rng.standard_normal((n_sims, horizon))
        out = np.empty(n_sims)
        xi = self.xi_T.copy()

        for i in range(n_sims):
            reg = int(rng.random() > xi[0])
            h = float(self._hT[reg])
            cv = 0.0
            for d in range(horizon):
                reg = int(rng.random() > P[0, reg])
                e2 = z[i, d] ** 2 * h
                cv += e2
                if reg == 0:
                    h = w0 + a0 * e2 + b0 * h
                else:
                    h = w1 + a1 * e2 + b1 * h
                if h < 1e-12:
                    h = 1e-12
            out[i] = np.sqrt(cv / horizon * 252)
        return out

    @property
    def regime(self):
        return int(self.xi_T[1] > 0.5)

    @property
    def prob_turb(self):
        return float(self.xi_T[1])


# ═══════════════════════════════════════════════════════════════
# SIGNAL  (sell-only, multi-gate)
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

    # G6 — Regime: penalise selling into a turbulent regime
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
# HALF-KELLY + TAIL GATE
# ═══════════════════════════════════════════════════════════════
def size_pos(sig, spot, capital, dte, r, max_risk=0.04):
    if sig.trade == "NO_TRADE":
        return 0, 0.0

    edge = max(sig.vrp_pct, 0.0)
    kh = edge * sig.conf * 0.50

    # tail-risk dampener
    tr = sig.d_p95 / sig.iv if sig.iv > 0 else 1.0
    if tr > 0.95:
        td = 0.25
    elif tr > 0.85:
        td = 0.55
    else:
        td = 1.0

    # kurtosis dampener
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
# P&L  (short straddle, BS reprice + gamma adjustment)
# ═══════════════════════════════════════════════════════════════
_PARKINSON_SCALE = 1.0 / (4.0 * np.log(2))   # ≈ 0.3607


def calc_pnl(
    trade, n, si, so, hi, lo, ivi, ivo, dte, r=0.05,
    slip_per_contract=0.0, spread_per_leg=0.00, max_loss_mult=2,
):
    """
    Gamma-adjusted short-straddle P&L with intraday max-loss stop.

    Parameters
    ----------
    si, so : float
        Spot at entry (Open) and exit (Close).
    hi, lo : float
        Intraday High and Low — used for gamma cost and stop-loss check.
    ivi, ivo : float
        Implied vol at entry and exit.
    spread_per_leg : float
        Half-spread per option leg in $ (default $0.05 ≈ tight ATM SPY).
        Applied on BOTH entry and exit for both legs → 4 × spread_per_leg
        per round-trip per contract.
    max_loss_mult : float
        Close position if intraday loss exceeds this multiple of premium
        collected.  E.g. 2.0 → stop if loss > 2× premium.
        Set to 0 or inf to disable.
    """
    if trade == "NO_TRADE" or n == 0:
        return 0.0

    K = si 
    Ti = dte / 252
    pi = straddle_px(si, K, Ti, r, ivi)
    total_premium = pi * 100 * n   # total $ premium collected

    # ── MAX-LOSS STOP CHECK ──────────────────────────────────
    # Evaluate straddle value at worst intraday spot (hi or lo,
    # whichever hurts a short straddle more).  Use same Ti for a
    # conservative check — the stop would fire intraday before
    # theta has decayed much.
    if max_loss_mult > 0 and np.isfinite(max_loss_mult) and total_premium > 0:
        # Straddle value rises when spot moves away from strike
        worst_spot_hi = straddle_px(hi, K, Ti, r, ivi)
        worst_spot_lo = straddle_px(lo, K, Ti, r, ivi)
        worst_val = max(worst_spot_hi, worst_spot_lo)
        intraday_loss = (worst_val - pi) * 100 * n
        max_allowed = max_loss_mult * total_premium

        if intraday_loss > max_allowed:
            spread_cost = 4.0 * spread_per_leg * 100 * n
            slip_cost = slip_per_contract * n
            return round(-max_allowed - spread_cost - slip_cost, 2)

    # ── NORMAL CLOSE-TO-CLOSE P&L ────────────────────────────
    # Proper remaining DTE after 1 trading day
    To = max((dte - 1.0), 0.05) / 252

    # Mark-to-market P&L (short position: entry − exit)
    po = straddle_px(so, K, To, r, ivo)
    mtm = (pi - po) * 100 * n

    # ── GAMMA COST (Parkinson-scaled) ────────────────────────
    # The BS reprice only captures the open→close net move.
    # Intraday path-dependence creates additional gamma losses.
    # Parkinson: E[range²] = 4·ln(2) · σ² for GBM
    net_move_sq = (so - si) ** 2
    range_var = (hi - lo) ** 2 * _PARKINSON_SCALE  
    excess_var = max(range_var - net_move_sq, 0.0)

    if excess_var > 0 and Ti > 1e-6:
        gam = bs_gamma(si, K, Ti, r, ivi)
        # call γ + put γ ≈ 2γ ATM
        straddle_gamma = 2.0 * gam  
        gamma_cost = 0.5 * straddle_gamma * excess_var * 100 * n
    else:
        gamma_cost = 0.0

    # ── TRANSACTION COSTS ────────────────────────────────────
    # Bid-ask: 2 legs × 2 crosses × spread_per_leg
    spread_cost = 4.0 * spread_per_leg * 100 * n
    slip_cost = slip_per_contract * n

    pnl = mtm - gamma_cost - spread_cost - slip_cost
    return round(pnl, 2)


# ═══════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════
def load_data(years: int = None, start: str = None, end: str = None):
    if years is not None:
        end_time = end if end else datetime.now().strftime("%Y-%m-%d")
        start_time = (
            datetime.strptime(end_time, "%Y-%m-%d") - timedelta(days=years * 365)
        ).strftime("%Y-%m-%d")
    else:
        start_time = start
        end_time = end

    if start_time is None:
        raise ValueError("Provide either 'years' or 'start' and 'end' argument.")

    import yfinance as yf

    print("  Fetching SPY …")
    spy = yf.download(
        "SPY", start=start_time, end=end_time, progress=False, auto_adjust=True
    )
    print("  Fetching VIX9D …")
    v9d = yf.download(
        "^VIX9D", start=start_time, end=end_time, progress=False, auto_adjust=True
    )
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


# ═══════════════════════════════════════════════════════════════
# CORE BACKTEST ENGINE
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
    # signal thresholds for stress testing
    vrp_strong: float = 0.10
    vrp_weak: float = 0.05
    kurt_max: float = 4.0
    skew_max: float = 0.5


def run_engine(spy, v9d, rvp, cfg, start_idx=None, end_idx=None, quiet=False):
    """Core backtest loop. Returns DataFrame of daily results."""
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
        # ── periodic refit ──
        if (t - last_fit) >= cfg.refit or mdl is None:
            r_win = ret[t - cfg.lookback : t]
            try:
                mdl = MSGARCH(r_win)
                mdl.fit()
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

        sig = make_signal(
            den, iv,
            float(rv["rv_cc"]), float(rv["rv_pk"]),
            float(rv["rv_rs"]), float(rv["rv_yz"]), float(rv["rv_bp"]),
            mdl.regime, mdl.prob_turb,
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
            "pnl": pnl, "capital": round(cap, 2), "reason": sig.reason,
        })

        step = t - s
        if not quiet and step % 250 == 0:
            el = time.time() - t0
            print(
                f"    [{step / tot * 100:5.1f}%] {idx[t].date()} "
                f"cap=${cap:>14,.2f} ({el:.0f}s)"
            )

    if not quiet:
        print(f"\n  Done {time.time() - t0:.1f}s")
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).set_index("date")


# ═══════════════════════════════════════════════════════════════
# METRICS + REPORT
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
  Period / Days       : {m.get('years', 0):.1f}y  ({m['n_days']:,} days)
  Capital             : ${cfg.capital:>14,.2f}  ->  ${m['final_cap']:>14,.2f}
  Return / CAGR       : {m['return_pct']:+.2f}%  /  {m['cagr']:+.2f}%

  Total P&L           : ${m['total_pnl']:>+14,.2f}
  Max Drawdown        : ${m['max_dd']:>+14,.2f}  ({m['max_dd_pct']:+.1f}%)
  Sharpe              :  {m['sharpe']:+.3f}
  Sortino             :  {m['sortino']:+.3f}

  Trades              : {m['n_trades']:,}
  Win Rate            :  {m['win_rate']:.1f}%
  Avg Win / Loss      : ${m['avg_win']:>+12,.2f}  /  ${m['avg_loss']:>+12,.2f}
  Profit Factor       :  {m['profit_factor']:.2f}

  Bid-Ask / Leg       : ${cfg.spread_per_leg:.2f}
  IV Haircut          :  {cfg.iv_haircut:.0%}
  Max-Loss Stop       :  {cfg.max_loss_mult:.1f}× premium
{'=' * 72}"""


# ═══════════════════════════════════════════════════════════════
# SLIPPAGE & TRANSACTION COST STRESS TEST
# ═══════════════════════════════════════════════════════════════
def stress_slippage(spy, v9d, rvp, base_cfg):
    """Sweep slippage and bid-ask spread."""
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


# ═══════════════════════════════════════════════════════════════
# PARAMETER SENSITIVITY STRESS TEST
# ═══════════════════════════════════════════════════════════════
def stress_params(spy, v9d, rvp, base_cfg):
    """Perturb key signal & sizing parameters."""
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
# CHARTs
# ═══════════════════════════════════════════════════════════════
def make_chart(df, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(
            5, 1, figsize=(16, 17), sharex=True,
            gridspec_kw={"height_ratios": [3, 1.2, 1.2, 1, 1]},
        )
        fig.suptitle(
            "MS-GARCH(1,1) Vol-Sell",
            fontsize=13, weight="bold", y=0.98,
        )

        # 1) equity curve
        cum = df["pnl"].cumsum() + 1_000_000
        ax[0].fill_between(
            cum.index, cum, 1e6, where=cum >= 1e6, alpha=0.2, color="#2E7D32"
        )
        ax[0].fill_between(
            cum.index, cum, 1e6, where=cum < 1e6, alpha=0.2, color="#C62828"
        )
        ax[0].plot(cum.index, cum, color="#1B5E20", lw=1.2)
        ax[0].axhline(1e6, color="grey", lw=0.5, ls="--")
        ax[0].set_ylabel("Portfolio ($)")
        ax[0].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
        )
        ax[0].grid(alpha=0.2)
        ax[0].set_title("Equity Curve", fontsize=10, loc="left")

        # 2) VRP
        ax[1].bar(
            df.index, df["vrp_pct"], width=1, alpha=0.5,
            color=np.where(df["vrp_pct"] > 0, "#4CAF50", "#F44336"),
        )
        act = df[df["trade"] != "NO_TRADE"]
        if len(act):
            ax[1].scatter(
                act.index, act["vrp_pct"], s=8, color="#0D47A1",
                zorder=5, label="Trade",
            )
        ax[1].axhline(0, color="grey", lw=0.5)
        ax[1].set_ylabel("VRP%")
        ax[1].legend(fontsize=7, loc="upper right")
        ax[1].grid(alpha=0.2)

        # 3) Forecast vs IV vs RV
        ax[2].plot(
            df.index, df["fc_mean"], label="GARCH Forecast",
            color="#FF9800", lw=0.9,
        )
        ax[2].plot(
            df.index, df["iv_in"], label="IV (VIX9D adj.)",
            color="#7B1FA2", lw=0.9, alpha=0.8,
        )
        ax[2].plot(
            df.index, df["rv_cons"], label="RV Consensus",
            color="#00796B", lw=0.7, alpha=0.7,
        )
        ax[2].set_ylabel("Ann. Vol")
        ax[2].legend(fontsize=7, loc="upper right")
        ax[2].grid(alpha=0.2)

        # 4) Regime
        ax[3].fill_between(
            df.index, df["prob_turb"], alpha=0.5, color="#E53935"
        )
        ax[3].axhline(0.5, color="grey", lw=0.5, ls="--")
        ax[3].set_ylabel("P(Turbulent)")
        ax[3].set_ylim(0, 1)
        ax[3].grid(alpha=0.2)

        # 5) daily pnl
        col = np.where(df["pnl"] > 0, "#2E7D32", "#C62828")
        ax[4].bar(df.index, df["pnl"], width=1, color=col, alpha=0.6)
        ax[4].axhline(0, color="grey", lw=0.5)
        ax[4].set_ylabel("Daily P&L ($)")
        ax[4].grid(alpha=0.2)
        ax[4].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()
        p = os.path.join(out_dir, "equity_curve.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Chart -> {p}")
    except Exception as e:
        print(f"  (chart skipped: {e})")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main(stress_test: bool):
    cfg = Cfg(max_loss_mult=np.inf)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("-> Loading market data")
    spy, v9d = load_data(years=5)
    rvp = build_rv_panel(spy)

    print("\n-> Running full backtest")
    df = run_engine(spy, v9d, rvp, cfg)
    if df.empty:
        print(" No results — check data or model convergence.")
        return

    m = calc_metrics(df, cfg.capital)
    rpt = format_report(m, "FULL BACKTEST RESULTS", cfg)
    print(rpt)

    if stress_test == True:
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
    print(f"  Slip   -> {slip_path}")

    param_path = os.path.join(OUTPUT_DIR, "stress_params.csv")
    param_df.to_csv(param_path, index=False)
    print(f"  Params -> {param_path}")

    full_text = "\n\n".join([
        rpt,
        "\nSLIPPAGE & BID-ASK STRESS TEST\n" + slip_df.to_string(index=False),
        "\nPARAMETER SENSITIVITY STRESS TEST\n" + param_df.to_string(index=False),
    ])
    report_path = os.path.join(OUTPUT_DIR, "full_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"  Report -> {report_path}")

    make_chart(df, OUTPUT_DIR)

    print("\n  All done.\n")


if __name__ == "__main__":
    main(stress_test=False)
import numpy as np
from scipy.stats import t as scipy_t
from collections import defaultdict
import math


class RegimeDistFilter:
    """
    Regime-conditional Laplace + Student-t mixture for short-horizon
    return filtering on ~10-min intervals.

    f(x) = w * Laplace(x; loc, b) + (1-w) * StudentT(x; loc, s, nu)

    Laplace captures normal microstructure (variance-gamma subordination).
    Student-t captures stressed tails (parameter uncertainty / jumps).
    Weight w shifts by regime: calm -> Laplace, stressed -> Student-t.
    """

    def __init__(self, horizon_min=10, intra_window=40, carry_window=60):
        self.horizon_min = horizon_min
        self.intra_window = intra_window
        self.carry_window = carry_window

        self.intra_rets = []
        self.carry_rets = []
        self.last_px = None
        self.last_tm = None

        # Regime thresholds
        self.vol_calm_hi = 0.13
        self.vol_stress_lo = 0.25
        self.kurt_calm_hi = 4.0
        self.kurt_stress_lo = 7.0
        self.vix_calm_hi = 18.0
        self.vix_stress_lo = 28.0

        # Mixture weights per regime [w = Laplace weight]
        self.w_map = {"calm": 0.75, "normal": 0.50, "stressed": 0.25}

        # Fitted state
        self.regime = "normal"
        self.w_lap = 0.50
        self.lap_loc = 0.0
        self.lap_b = 1e-4
        self.t_df = 10.0
        self.t_loc = 0.0
        self.t_scale = 1e-4
        self.ann_vol = 0.0
        self.ex_kurt = 3.0
        self.fitted = False

        # Hedge-filter percentile thresholds
        self.pct_calm = 0.70
        self.pct_normal = 0.65
        self.pct_stress = 0.55

        # Entry-filter: reject if realised 95th tail / IV-implied > cap
        self.tail_ratio_cap = 2.0

        # Diagnostics
        self.regime_counts = defaultdict(int)
        self.hedge_decisions = {"signal": 0, "noise": 0, "fallback": 0}

    # ── Sampling ────────────────────────────────────────────────

    def update(self, price, time, vix=None):
        if price <= 0:
            return
        if self.last_px is None:
            self.last_px = price
            self.last_tm = time
            return
        elapsed = (time - self.last_tm).total_seconds() / 60.0
        if elapsed < self.horizon_min:
            return
        if self.last_px > 0 and price > 0:
            self.intra_rets.append(math.log(price / self.last_px))
            if len(self.intra_rets) > self.intra_window:
                self.intra_rets = self.intra_rets[-self.intra_window:]
        self.last_px = price
        self.last_tm = time
        pool = self.carry_rets[-self.carry_window:] + self.intra_rets
        if len(pool) >= 12:
            self._fit(pool, vix)

    def new_day(self):
        combined = self.carry_rets[-self.carry_window:] + self.intra_rets
        self.carry_rets = combined[-self.carry_window:]
        self.intra_rets = []
        self.last_px = None
        self.last_tm = None

    # ── Fitting ─────────────────────────────────────────────────

    def _fit(self, rets, vix=None):
        r = np.array(rets)
        n = len(r)
        if n < 12:
            return
        mu = np.mean(r)
        sig = max(np.std(r, ddof=1), 1e-10)
        ipd = 390.0 / self.horizon_min
        self.ann_vol = sig * math.sqrt(ipd * 252)
        z = (r - mu) / sig
        self.ex_kurt = max(float(np.mean(z ** 4)) - 3.0, 0.0)

        # Regime scoring
        cp, sp = 0, 0
        if self.ann_vol < self.vol_calm_hi: cp += 1
        if self.ann_vol > self.vol_stress_lo: sp += 1
        if self.ex_kurt < self.kurt_calm_hi: cp += 1
        if self.ex_kurt > self.kurt_stress_lo: sp += 1
        if vix is not None:
            if vix < self.vix_calm_hi: cp += 1
            if vix > self.vix_stress_lo: sp += 1

        if sp >= 2:
            self.regime = "stressed"
        elif cp >= 2:
            self.regime = "calm"
        else:
            self.regime = "normal"
        self.w_lap = self.w_map[self.regime]
        self.regime_counts[self.regime] += 1

        # Laplace MLE: loc=median, scale=MAD
        med = float(np.median(r))
        self.lap_loc = med
        self.lap_b = max(float(np.mean(np.abs(r - med))), 1e-10)

        # Student-t MoM: kappa=6/(nu-4) => nu=4+6/kappa
        self.t_loc = mu
        if self.ex_kurt > 0.5:
            self.t_df = min(max(4.0 + 6.0 / self.ex_kurt, 4.1), 60.0)
        else:
            self.t_df = 60.0
        if self.t_df > 2.0:
            self.t_scale = max(sig * math.sqrt((self.t_df - 2.0) / self.t_df), 1e-10)
        else:
            self.t_scale = max(sig, 1e-10)
        self.fitted = True

    # ── Distribution queries ────────────────────────────────────

    def _lap_tail(self, x):
        return math.exp(-x / self.lap_b) if x > 0 else 1.0

    def _t_tail(self, x):
        if x <= 0:
            return 1.0
        try:
            return 2.0 * (1.0 - scipy_t.cdf(x / self.t_scale, self.t_df))
        except Exception:
            return 0.01

    def tail_prob(self, move_abs):
        if not self.fitted:
            return 0.5
        return self.w_lap * self._lap_tail(move_abs) + (1.0 - self.w_lap) * self._t_tail(move_abs)

    def move_percentile(self, move_abs):
        return 1.0 - self.tail_prob(move_abs)

    def quantile_abs(self, p):
        if not self.fitted:
            return 0.01
        target = 1.0 - p
        lo, hi = 0.0, self.lap_b * 15.0
        for _ in range(40):
            mid = (lo + hi) * 0.5
            if self.tail_prob(mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) * 0.5

    # ── Strategy interface ──────────────────────────────────────

    def hedge_signal(self, move_frac):
        if not self.fitted:
            self.hedge_decisions["fallback"] += 1
            ok = abs(move_frac) > 0.01
            return ok, "fallback"
        pct = self.move_percentile(abs(move_frac))
        thresh = {"calm": self.pct_calm, "normal": self.pct_normal,
                  "stressed": self.pct_stress}[self.regime]
        if pct >= thresh:
            self.hedge_decisions["signal"] += 1
            return True, f"SIG p={pct:.2f} r={self.regime}"
        self.hedge_decisions["noise"] += 1
        return False, f"NSE p={pct:.2f} r={self.regime}"

    def entry_ok(self, iv):
        if not self.fitted:
            return True, "not_fitted"
        q95 = self.quantile_abs(0.95)
        iv_mv = (iv / math.sqrt(252.0)) * math.sqrt(self.horizon_min / 390.0)
        if iv_mv < 1e-10:
            return True, "iv_zero"
        ratio = q95 / iv_mv
        return ratio < self.tail_ratio_cap, f"r={ratio:.2f} cap={self.tail_ratio_cap}"

    def dynamic_sigma_mult(self):
        return {"calm": 2.0, "normal": 2.5, "stressed": 3.2}.get(self.regime, 2.5)

    def summary(self):
        return {
            "regime": self.regime, "w_laplace": self.w_lap,
            "ann_vol": self.ann_vol, "ex_kurt": self.ex_kurt,
            "lap_b": self.lap_b, "t_df": self.t_df, "t_scale": self.t_scale,
            "n_intra": len(self.intra_rets), "n_carry": len(self.carry_rets),
            "hedge_signal": self.hedge_decisions["signal"],
            "hedge_noise": self.hedge_decisions["noise"],
            "hedge_fallback": self.hedge_decisions["fallback"],
            "regime_counts": dict(self.regime_counts),
        }
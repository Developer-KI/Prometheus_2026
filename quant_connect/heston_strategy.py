# region imports
from AlgorithmImports import *
import numpy as np
from collections import deque
from typing import Optional
import math

from heston_model import (
    HestonModel,
    RVolPanel,
    Signal,
    bs_call,
    bs_put,
    bs_vega,
    straddle_px,
    implied_vol,
    detect_regime,
    make_signal,
    compute_n_contracts,
)
# endregion


# ═══════════════════════════════════════════════════════════════════════
#  HESTON STOCHASTIC-VOL   ×   SHORT STRADDLE
# ═══════════════════════════════════════════════════════════════════════
#
#  Calibrates a Heston model from the historical options chain,
#  runs Andersen-QE Monte Carlo to build a 5-day vol forecast density,
#  compares it against implied vol + realised-vol panel to generate a
#  sell signal with half-Kelly sizing and tail-risk gating.
#
#  Execution: sell ATM short straddle on the nearest ~2 DTE expiry,
#  hold to expiry or close at max-loss stop.
#
# ═══════════════════════════════════════════════════════════════════════


class HestonVolSell(QCAlgorithm):

    # ── configuration ────────────────────────────────────────────────
    REFIT_DAYS        = 5        # recalibrate Heston every N trading days
    FC_HORIZON        = 5        # MC forecast horizon (days)
    MC_SIMS           = 3000     # Monte-Carlo paths
    TARGET_DTE_MIN    = 1        # min DTE for straddle entry
    TARGET_DTE_MAX    = 5        # max DTE for straddle entry
    MAX_RISK_FRAC     = 0.04     # max portfolio fraction at risk per trade
    MAX_LOSS_MULT     = 2.0      # close if loss > N× premium collected
    VRP_STRONG        = 0.10
    VRP_WEAK          = 0.05
    KURT_MAX          = 4.0
    SKEW_MAX          = 0.5
    MONEYNESS_BAND    = 0.15     # chain calibration filter ±15% from ATM
    IV_HAIRCUT        = 0.02     # shave IV before signal generation
    RV_WINDOW         = 22
    LOOKBACK_BARS     = 504      # for returns-based fallback calibration
    CHAIN_MIN_OPTIONS = 8        # min options needed for chain calibration

    # ─────────────────────────────────────────────────────────────────
    #  INITIALISE
    # ─────────────────────────────────────────────────────────────────
    def initialize(self):
        self.set_start_date(2021, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(1_000_000)

        # ── equity ──
        self.equity = self.add_equity("SPY", Resolution.DAILY)
        self.equity.set_data_normalization_mode(DataNormalizationMode.RAW)
        self.spy = self.equity.symbol

        # ── options universe ──
        option = self.add_option("SPY", Resolution.DAILY)
        option.set_filter(
            lambda u: u.include_weeklys()
                       .strikes(-10, 10)
                       .expiration(self.TARGET_DTE_MIN, self.TARGET_DTE_MAX + 7)
        )
        self.option_symbol = option.symbol

        # ── VIX as variance proxy for fallback calibration ──
        try:
            self.vix_symbol = self.add_index("VIX", Resolution.DAILY).symbol
        except Exception:
            try:
                self.vix_symbol = self.add_data(
                    CBOEVix, "VIX", Resolution.DAILY
                ).symbol
            except Exception:
                self.vix_symbol = None
                self.log("WARN: VIX feed unavailable — chain-only calibration")

        self.rfr = 0.05

        # ── model instances (from heston_model.py) ──
        self.heston   = HestonModel()
        self.rv_panel = RVolPanel(window=self.RV_WINDOW)

        # ── strategy state ──
        self.density: Optional[np.ndarray] = None
        self.last_signal: Optional[Signal] = None
        self._days_since_fit = self.REFIT_DAYS + 1
        self._bar_count = 0

        # ── rolling buffers for fallback calibration ──
        self._ret_buf = deque(maxlen=self.LOOKBACK_BARS)
        self._var_buf = deque(maxlen=self.LOOKBACK_BARS)

        # ── position tracking ──
        self._open_call: Optional[Symbol] = None
        self._open_put:  Optional[Symbol] = None
        self._entry_premium: float = 0.0
        self._entry_time: Optional[datetime] = None

        # ── daily scheduled event ──
        self.schedule.on(
            self.date_rules.every_day(self.spy),
            self.time_rules.after_market_open(self.spy, 30),
            self._on_market_open,
        )

        # warm-up so RV panel + return buffer are populated
        self.set_warm_up(timedelta(days=self.LOOKBACK_BARS + 60))

    # ═══════════════════════════════════════════════════════════════
    #  DAILY LOGIC
    # ═══════════════════════════════════════════════════════════════
    def _on_market_open(self):
        if self.is_warming_up:
            return

        spot = self.securities[self.spy].price
        if spot <= 0:
            return

        # ── feed RV panel ──
        bar = self.securities[self.spy]
        if bar.open > 0 and bar.high > 0:
            self.rv_panel.update(
                float(bar.open), float(bar.high),
                float(bar.low),  float(bar.close),
            )

        # ── feed returns / variance buffers ──
        hist = self.history(self.spy, 2, Resolution.DAILY)
        if len(hist) >= 2:
            c1, c0 = float(hist["close"].iloc[-1]), float(hist["close"].iloc[-2])
            if c0 > 0 and c1 > 0:
                self._ret_buf.append(math.log(c1 / c0))

        if self.vix_symbol is not None:
            vix_sec = self.securities.get(self.vix_symbol)
            if vix_sec is not None and vix_sec.price > 0:
                self._var_buf.append((float(vix_sec.price) / 100.0) ** 2)

        self._bar_count += 1

        # ── max-loss stop check ──
        if self._has_position():
            self._check_stop_loss(spot)

        # ── periodic Heston refit ──
        self._days_since_fit += 1
        if self._days_since_fit >= self.REFIT_DAYS:
            self._calibrate_heston(spot)

        if self.density is None or not self.rv_panel.ready:
            return

        # ── generate signal ──
        rv = self.rv_panel.all_rv()
        if not rv:
            return

        raw_iv = math.sqrt(max(self.heston.v0, 1e-6))
        iv = max(raw_iv - self.IV_HAIRCUT, 0.01)

        rv_med = np.median(list(rv.values()))
        regime, prob_turb = detect_regime(
            self.heston.v0, self.heston.theta, rv["rv_cc"], rv_med,
        )

        sig = make_signal(
            self.density, iv, rv, regime, prob_turb,
            self.VRP_STRONG, self.VRP_WEAK, self.KURT_MAX, self.SKEW_MAX,
        )
        self.last_signal = sig

        self.log(
            f"SIG: {sig.trade}  conf={sig.conf:.2f}  "
            f"VRP={sig.vrp_pct:+.1%}  IV={sig.iv:.3f}  "
            f"fc_mean={sig.fc_mean:.3f}  fc_p95={sig.fc_p95:.3f}  "
            f"regime={'TURB' if regime else 'CALM'}({prob_turb:.0%})  "
            f"\u03ba={self.heston.kappa:.1f} \u03b8={self.heston.theta:.4f} "
            f"\u03c3\u1d65={self.heston.sigma_v:.2f} \u03c1={self.heston.rho:.2f}  "
            f"| {sig.reason}"
        )

        # ── execute ──
        if sig.trade == "NO_TRADE" or self._has_position():
            return
        self._enter_straddle(sig, spot)

    # ═══════════════════════════════════════════════════════════════
    #  HESTON CALIBRATION
    # ═══════════════════════════════════════════════════════════════
    def _calibrate_heston(self, spot: float):
        """Try chain calibration first; fall back to returns + VIX."""
        success = False

        # ── attempt 1: options chain ──
        try:
            chain_data = self._extract_chain(spot)
            if chain_data is not None and len(chain_data["strikes"]) >= self.CHAIN_MIN_OPTIONS:
                summary = self.heston.calibrate_from_chain(
                    chain_data["strikes"], chain_data["Ts"],
                    chain_data["mids"], chain_data["otypes"],
                    chain_data["ivs"], spot, self.rfr,
                )
                self.log(f"HESTON-CHAIN: {summary}")
                success = True
        except Exception as e:
            self.log(f"Chain calibration failed: {e}")

        # ── attempt 2: returns + VIX fallback ──
        if not success:
            try:
                if len(self._ret_buf) >= 100 and len(self._var_buf) >= 100:
                    summary = self.heston.calibrate_from_returns(
                        np.array(self._ret_buf),
                        np.array(self._var_buf),
                    )
                    self.log(f"HESTON-RET: {summary}")
                    success = True
            except Exception as e:
                self.log(f"Returns calibration failed: {e}")

        if not success:
            return

        # ── MC forecast ──
        try:
            seed = int(self.time.strftime("%Y%m%d"))
            self.density = self.heston.forecast_density(
                self.FC_HORIZON, self.MC_SIMS, seed=seed,
            )
            self._days_since_fit = 0
        except Exception as e:
            self.log(f"MC forecast failed: {e}")

    # ─── chain extraction ───────────────────────────────────────────
    def _extract_chain(self, spot: float) -> Optional[dict]:
        """
        Pull the current options chain via QC's OptionChainProvider,
        filter to near-ATM / reasonable DTE, return arrays for calibration.
        """
        contracts = self.option_chain_provider.get_option_contract_list(
            self.spy, self.time,
        )
        if not contracts:
            return None

        today = self.time.date()
        lo_k  = spot * (1 - self.MONEYNESS_BAND)
        hi_k  = spot * (1 + self.MONEYNESS_BAND)

        strikes, Ts, mids, otypes, ivs = [], [], [], [], []

        for symbol in contracts:
            K   = float(symbol.id.strike_price)
            exp = symbol.id.date.date()
            dte = (exp - today).days
            if dte < 2 or dte > 90:
                continue
            if K < lo_k or K > hi_k:
                continue

            T  = dte / 365.0
            ot = "call" if symbol.id.option_right == OptionRight.CALL else "put"

            # get market data for this contract
            sec = self.securities.get(symbol)
            if sec is None:
                try:
                    self.add_option_contract(symbol, Resolution.DAILY)
                    sec = self.securities.get(symbol)
                except Exception:
                    continue
            if sec is None:
                continue

            bid = float(sec.bid_price) if sec.bid_price > 0 else 0
            ask = float(sec.ask_price) if sec.ask_price > 0 else 0
            mid = 0.5 * (bid + ask) if bid > 0 and ask > 0 else float(sec.price)
            if mid < 0.10:
                continue

            try:
                iv = implied_vol(mid, spot, K, T, self.rfr, ot)
                if iv < 0.02 or iv > 3.0:
                    continue
            except Exception:
                iv = 0.20

            strikes.append(K)
            Ts.append(T)
            mids.append(mid)
            otypes.append(ot)
            ivs.append(iv)

        if len(strikes) < self.CHAIN_MIN_OPTIONS:
            return None

        return dict(
            strikes=np.array(strikes), Ts=np.array(Ts),
            mids=np.array(mids), otypes=otypes, ivs=np.array(ivs),
        )

    # ═══════════════════════════════════════════════════════════════
    #  TRADE EXECUTION
    # ═══════════════════════════════════════════════════════════════
    def _enter_straddle(self, sig: Signal, spot: float):
        """Find the best ATM straddle and sell it."""
        chain = self.current_slice.option_chains.get(self.option_symbol)
        if chain is None:
            return

        # find expiry closest to ~2 DTE target
        now = self.time.date()
        best_exp, best_dte = None, 999
        for contract in chain:
            dte = (contract.expiry.date() - now).days
            if self.TARGET_DTE_MIN <= dte <= self.TARGET_DTE_MAX + 3:
                if abs(dte - 2) < abs(best_dte - 2):
                    best_exp = contract.expiry
                    best_dte = dte
        if best_exp is None:
            return

        # filter to chosen expiry
        exp_contracts = [c for c in chain if c.expiry == best_exp]
        calls = [c for c in exp_contracts if c.right == OptionRight.CALL]
        puts  = [c for c in exp_contracts if c.right == OptionRight.PUT]
        if not calls or not puts:
            return

        # ATM strike
        atm_call = min(calls, key=lambda c: abs(float(c.strike) - spot))
        K = float(atm_call.strike)
        atm_put = min(puts, key=lambda c: abs(float(c.strike) - K))

        # position sizing
        capital = float(self.portfolio.total_portfolio_value)
        n = compute_n_contracts(
            sig, spot, capital, max(best_dte, 0.5), self.rfr, self.MAX_RISK_FRAC,
        )
        if n <= 0:
            return

        # estimated premium for tracking
        call_mid = (
            0.5 * (float(atm_call.bid_price) + float(atm_call.ask_price))
            if atm_call.bid_price > 0
            else float(atm_call.last_price)
        )
        put_mid = (
            0.5 * (float(atm_put.bid_price) + float(atm_put.ask_price))
            if atm_put.bid_price > 0
            else float(atm_put.last_price)
        )
        est_premium = (call_mid + put_mid) * 100 * n

        # sell straddle
        self.market_order(atm_call.symbol, -n)
        self.market_order(atm_put.symbol, -n)

        self._open_call     = atm_call.symbol
        self._open_put      = atm_put.symbol
        self._entry_premium = est_premium
        self._entry_time    = self.time

        self.log(
            f"SELL STRADDLE: {n}x K={K:.0f} DTE={best_dte} "
            f"est_prem=${est_premium:,.0f}  sig={sig.trade}({sig.conf:.2f})"
        )

    # ═══════════════════════════════════════════════════════════════
    #  POSITION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    def _has_position(self) -> bool:
        return self._open_call is not None or self._open_put is not None

    def _check_stop_loss(self, spot: float):
        """Close if unrealised loss > MAX_LOSS_MULT × premium collected."""
        if self._entry_premium <= 0:
            return

        unreal = 0.0
        for sym in [self._open_call, self._open_put]:
            if sym is not None and self.portfolio[sym].invested:
                unreal += float(self.portfolio[sym].unrealized_profit)

        loss = -unreal
        max_allowed = self.MAX_LOSS_MULT * self._entry_premium
        if loss > max_allowed:
            self.log(
                f"STOP-LOSS: loss=${loss:,.0f} > "
                f"{self.MAX_LOSS_MULT}x prem=${self._entry_premium:,.0f}"
            )
            self._close_position("STOP_LOSS")

    def _close_position(self, reason: str = ""):
        """Liquidate all option legs and reset tracking."""
        for sym in [self._open_call, self._open_put]:
            if sym is not None and self.portfolio[sym].invested:
                self.liquidate(sym, tag=reason)

        pnl = 0.0
        for sym in [self._open_call, self._open_put]:
            if sym is not None:
                pnl += float(self.portfolio[sym].last_trade_profit)
        self.log(f"CLOSE ({reason}): P&L=${pnl:,.2f}")

        self._open_call     = None
        self._open_put      = None
        self._entry_premium = 0.0
        self._entry_time    = None

    # ─── event handlers ─────────────────────────────────────────────
    def on_order_event(self, order_event):
        """Handle assignment / exercise / expiry."""
        if order_event.status != OrderStatus.FILLED:
            return

        sym = order_event.symbol

        # option leg closed externally (exercise / expiry)
        if sym.security_type == SecurityType.OPTION:
            if sym in [self._open_call, self._open_put]:
                if not self.portfolio[sym].invested:
                    if sym == self._open_call:
                        self._open_call = None
                    elif sym == self._open_put:
                        self._open_put = None
                    if self._open_call is None and self._open_put is None:
                        self._entry_premium = 0.0
                        self._entry_time    = None

        # if assigned, liquidate underlying shares immediately
        if sym.security_type == SecurityType.EQUITY and sym == self.spy:
            qty = self.portfolio[self.spy].quantity
            if qty != 0:
                self.log(f"ASSIGNMENT: liquidating {qty} SPY shares")
                self.market_order(self.spy, -qty, tag="assignment_unwind")

    def on_securities_changed(self, changes):
        """Clean up tracking when option contracts are delisted."""
        for sec in changes.removed_securities:
            if sec.symbol == self._open_call:
                self._open_call = None
            if sec.symbol == self._open_put:
                self._open_put = None
        if self._open_call is None and self._open_put is None:
            self._entry_premium = 0.0

    def on_end_of_algorithm(self):
        if self._has_position():
            self._close_position("EOD")
        self.log(f"FINAL: ${self.portfolio.total_portfolio_value:,.2f}")


# ─────────────────────────────────────────────────────────────────────
#  CBOE VIX custom data (fallback if add_index unavailable)
# ─────────────────────────────────────────────────────────────────────
class CBOEVix(PythonData):
    """Reads VIX history from CBOE public CSV."""

    def get_source(self, config, date, is_live):
        return SubscriptionDataSource(
            "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
            SubscriptionTransportMedium.REMOTE_FILE,
        )

    def reader(self, config, line, date, is_live):
        if not line.strip() or line.upper().startswith("DATE"):
            return None
        try:
            cols = line.split(",")
            data = CBOEVix()
            data.symbol = config.symbol
            data.time = datetime.strptime(cols[0].strip(), "%m/%d/%Y")
            data.value = float(cols[4])
            return data
        except Exception:
            return None
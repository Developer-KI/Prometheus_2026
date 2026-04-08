# region imports
from AlgorithmImports import *
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
# endregion

class MSGARCHVolatilitySelling(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2021, 1, 1)
        self.set_end_date(2024, 3, 1)
        self.set_cash(1000000)
        
        # GARCH & Model Parameters
        self.lookback = int(self.get_parameter("lookback", 504))
        self.refit_days = int(self.get_parameter("refit_days", 5))
        self.mc_sims = int(self.get_parameter("mc_sims", 1000))
        self.fc_horizon = int(self.get_parameter("fc_horizon", 5))
        
        # Option Selection Parameters
        self.dte_target = int(self.get_parameter("dte_target", 2))
        self.iv_haircut = float(self.get_parameter("iv_haircut", 0.00))
        
        # Risk & Stop Loss Parameters
        self.max_risk = float(self.get_parameter("max_risk", 0.04))
        self.stop_loss_multiplier = float(self.get_parameter("stop_loss_mult", 2.0))
        
        # Signal Threshold Parameters
        self.vrp_strong = float(self.get_parameter("vrp_strong", 0.10))
        self.vrp_weak = float(self.get_parameter("vrp_weak", 0.05))
        self.kurt_max = float(self.get_parameter("kurt_limit", 4.0))
        self.skew_max = float(self.get_parameter("skew_limit", 0.5))

        # --- Assets ---
        self.spy = self.add_equity("SPY", Resolution.MINUTE)
        self.spy.set_data_normalization_mode(DataNormalizationMode.RAW)
        self.vix9d = self.add_index("VIX9D", Resolution.DAILY).symbol
        self.option = self.add_option("SPY", Resolution.MINUTE)
        self.option.set_filter(-2, +2, 1, 4) 
        
        # --- State Variables ---
        self.mdl = None
        self.den = None
        self.days_count = 0
        self.active_straddle_symbols = []
        self.entry_premium_per_unit = 0

        # --- Scheduling ---
        self.schedule.on(self.date_rules.every_day("SPY"), 
                         self.time_rules.after_market_open("SPY", 30), 
                         self.process_strategy)
        
        self.schedule.on(self.date_rules.every_day("SPY"), 
                         self.time_rules.before_market_close("SPY", 15), 
                         self.close_positions)

        self.set_warm_up(self.lookback, Resolution.DAILY)

    def on_data(self, data):
        """
        Monitors the stop loss intraday (minute-by-minute).
        """
        if not self.active_straddle_symbols:
            return

        # Ensure we have data for both legs
        s1, s2 = self.active_straddle_symbols[0], self.active_straddle_symbols[1]
        if data.contains_key(s1) and data.contains_key(s2):
            # Current cost to buy back the straddle
            current_straddle_price = self.securities[s1].price + self.securities[s2].price
            
            # Stop Loss Check
            stop_threshold = self.entry_premium_per_unit * self.stop_loss_multiplier
            
            if current_straddle_price >= stop_threshold:
                self.log(f"STOP LOSS TRIGGERED: Current {current_straddle_price:.2f} >= Limit {stop_threshold:.2f}")
                self.liquidate()
                self.active_straddle_symbols = []

    def process_strategy(self):
        if self.is_warming_up: return
        if self.days_count % self.refit_days == 0:
            self.update_model()
        
        self.days_count += 1
        if self.mdl is None or self.den is None: return

        vix_history = self.history(self.vix9d, 1, Resolution.DAILY)
        if vix_history.empty: return
        iv = float(vix_history.iloc[-1]['close']) / 100.0

        rv_panel = self.get_rv_panel()
        if rv_panel is None: return

        sig = self.make_signal(iv, rv_panel)
        if sig['trade'] != "NO_TRADE":
            self.execute_trade(sig)

    def execute_trade(self, sig):
        chain = self.option_chain_provider.get_option_contract_list(self.spy.symbol, self.time)
        if not chain: return

        expiry = sorted(chain, key=lambda x: abs((x.id.date - self.time).days - self.dte_target))[0].id.date
        contracts = [x for x in chain if x.id.date == expiry]
        atm_strike = sorted(contracts, key=lambda x: abs(x.id.strike_price - self.spy.price))[0].id.strike_price
        
        put = [x for x in contracts if x.id.strike_price == atm_strike and x.id.option_right == OptionRight.PUT]
        call = [x for x in contracts if x.id.strike_price == atm_strike and x.id.option_right == OptionRight.CALL]

        if not put or not call: return
        
        p_con = self.add_option_contract(put[0])
        c_con = self.add_option_contract(call[0])

        # Sizing
        capital = self.portfolio.total_portfolio_value
        frac = max(sig['vrp_pct'], 0.0) * sig['conf'] * 0.50
        if sig['trade'] == "SELL_SMALL": frac *= 0.50
        
        risk_usd = capital * min(frac, self.max_risk)
        self.entry_premium_per_unit = p_con.price + c_con.price
        
        if self.entry_premium_per_unit > 0:
            qty = int(risk_usd / (self.entry_premium_per_unit * 100))
            if qty > 0:
                self.sell(c_con.symbol, qty)
                self.sell(p_con.symbol, qty)
                # Store symbols for the intraday stop-loss monitor
                self.active_straddle_symbols = [c_con.symbol, p_con.symbol]

    def close_positions(self):
        if self.portfolio.invested:
            self.liquidate()
        self.active_straddle_symbols = []

    def get_rv_panel(self):
        hist = self.history(self.spy.symbol, 30, Resolution.DAILY)
        if len(hist) < 25: return None
        c = hist['close']
        rv_cc = np.log(c / c.shift(1)).std() * np.sqrt(252)
        rv_pk = np.sqrt((1/(4*np.log(2))) * (np.log(hist['high']/hist['low'])**2).mean() * 252)
        return {"cc": rv_cc, "pk": rv_pk}

    def make_signal(self, iv, rv):
        dm = float(np.mean(self.den))
        p95 = float(np.percentile(self.den, 95))
        dku = float(stats.kurtosis(self.den))
        dsk = float(stats.skew(self.den))
        
        vrp_pct = (iv - dm) / iv if iv > 0 else 0
        score = 0.0
        if vrp_pct > self.vrp_strong: score += 0.30
        elif vrp_pct > self.vrp_weak: score += 0.15
        if iv > rv['cc'] and iv > rv['pk']: score += 0.30
        if p95 < iv: score += 0.25
        if dku < self.kurt_max: score += 0.10
        if dsk < self.skew_max: score += 0.05
        
        if self.mdl.regime == 1 and self.mdl.prob_turb > 0.70: score *= 0.60

        trade = "NO_TRADE"
        if score >= 0.50: trade = "SELL_STRADDLE"
        elif score >= 0.35: trade = "SELL_SMALL"
        return {"trade": trade, "vrp_pct": vrp_pct, "conf": score}

    def update_model(self):
        history = self.history(self.spy.symbol, self.lookback + 1, Resolution.DAILY)
        if history.empty: return
        returns = np.diff(np.log(history['close'].values))
        try:
            self.mdl = MSGARCH_Model(returns)
            self.mdl.fit()
            self.den = self.mdl.forecast_density(self.fc_horizon, self.mc_sims)
        except: pass

class MSGARCH_Model:
    def __init__(self, returns):
        self.r = returns.astype(np.float64)
        self.r2 = self.r ** 2
        self.T = len(returns)
        self.params = None
        self.xi_T = np.array([0.5, 0.5])
        self._hT = np.array([0.0001, 0.0001])

    def _nll(self, p):
        p00, p11, w0, a0, b0, w1, a1, b1 = p
        P = np.array([[p00, 1-p11], [1-p00, p11]])
        h0 = h1 = np.var(self.r[:30])
        xi0, xi1 = 0.5, 0.5
        ll = 0.0
        for t in range(1, self.T):
            e2 = self.r2[t-1]
            h0 = np.maximum(w0 + a0 * e2 + b0 * h0, 1e-10)
            h1 = np.maximum(w1 + a1 * e2 + b1 * h1, 1e-10)
            f0 = (1/np.sqrt(2*np.pi*h0)) * np.exp(-0.5*self.r2[t]/h0)
            f1 = (1/np.sqrt(2*np.pi*h1)) * np.exp(-0.5*self.r2[t]/h1)
            xp0, xp1 = P[0,0]*xi0 + P[0,1]*xi1, P[1,0]*xi0 + P[1,1]*xi1
            lk = xp0*f0 + xp1*f1
            if lk <= 0 or not np.isfinite(lk): return 1e12
            ll += np.log(lk)
            xi0, xi1 = xp0*f0/lk, xp1*f1/lk
        return -ll

    def fit(self):
        bnd = [(0.5, 0.99)]*2 + [(1e-7, 1e-3), (0.01, 0.15), (0.7, 0.98), (1e-6, 1e-2), (0.05, 0.4), (0.4, 0.9)]
        init = [0.95, 0.90, 1e-6, 0.05, 0.90, 1e-5, 0.15, 0.70]
        res = minimize(self._nll, init, bounds=bnd, method='L-BFGS-B')
        self.params = res.x
        self._filter()

    def _filter(self):
        p00, p11, w0, a0, b0, w1, a1, b1 = self.params
        h0 = h1 = np.var(self.r[:30])
        xi0, xi1 = 0.5, 0.5
        for t in range(1, self.T):
            e2 = self.r2[t-1]
            h0, h1 = w0 + a0*e2 + b0*h0, w1 + a1*e2 + b1*h1
            f0, f1 = (1/np.sqrt(2*np.pi*h0))*np.exp(-0.5*self.r2[t]/h0), (1/np.sqrt(2*np.pi*h1))*np.exp(-0.5*self.r2[t]/h1)
            xp0, xp1 = p00*xi0 + (1-p11)*xi1, (1-p00)*xi0 + p11*xi1
            lk = xp0*f0 + xp1*f1 + 1e-15
            xi0, xi1 = xp0*f0/lk, xp1*f1/lk
        self.xi_T, self._hT = np.array([xi0, xi1]), np.array([h0, h1])

    def forecast_density(self, horizon, n_sims):
        p00, p11, w0, a0, b0, w1, a1, b1 = self.params
        P = np.array([[p00, 1-p11], [1-p00, p11]])
        rng = np.random.default_rng(42)
        vols = []
        for _ in range(n_sims):
            reg = 0 if rng.random() < self.xi_T[0] else 1
            h, cum_var = self._hT[reg], 0
            for _ in range(horizon):
                reg = 0 if rng.random() < P[0, reg] else 1
                e2 = (rng.standard_normal()**2) * h
                cum_var += e2
                if reg == 0: h = w0 + a0*e2 + b0*h
                else: h = w1 + a1*e2 + b1*h
            vols.append(np.sqrt(cum_var / horizon * 252))
        return np.array(vols)

    @property
    def regime(self): return int(self.xi_T[1] > 0.5)
    @property
    def prob_turb(self): return float(self.xi_T[1])
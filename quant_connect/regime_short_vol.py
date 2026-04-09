from AlgorithmImports import *
from collections import defaultdict
import numpy as np
from datetime import timedelta
from regime_filter import RegimeDistFilter


class ShortStraddleEnhanced(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2026, 2, 28)
        self.set_cash(1000000)

        self.underlying = self.add_equity("QQQ", Resolution.MINUTE).symbol
        option = self.add_option("QQQ", Resolution.MINUTE)
        option.set_filter(lambda u: u.strikes(-5, 5).expiration(0, 0))
        self.option_symbol = option.symbol

        self.vix_sym = self.add_data(CBOE, "VIX", Resolution.DAILY).symbol
        self.vix3m_sym = self.add_data(CBOE, "VIX3M", Resolution.DAILY).symbol
        self.vvix_sym = self.add_data(CBOE, "VVIX", Resolution.DAILY).symbol

        self.min_iv = 0.12
        self.pt = 1.00; self.sl = 1.00
        self.hband = 0.01; self.eod_mins = 15
        self.hedge_mode = "full"
        self.risk_pct = 0.05; self.qty_cap = 20
        self.init_d_thresh = 25; self.min_htrade = 10
        self.max_entry_d = 100
        self.weak_on = True; self.w_hr = 12; self.w_mn = 30; self.w_pct = 0.25
        self.ew_s = 45; self.ew_e = 120

        self.ts_on = True;   self.min_vr = 0.85
        self.vrp_on = True;  self.min_vrp = 0.002; self.rv_lb = 20
        self.on_on = True;   self.max_or = 0.01
        self.vv_on = True;   self.max_vv = 120.0
        self.macro_on = True
        self.mom_on = True;  self.max_mm = 0.008
        self.sig_on = True;  self.sig_m = 2.5
        self.har_on = True;  self.har_b = 1.05
        self.regime_on = True

        self.rdf = RegimeDistFilter(horizon_min=10, intra_window=40, carry_window=60)

        self.closes = RollingWindow[float](30)
        self.prev_close = None; self.today_open = None
        self.ocap = False; self.p935 = None
        self.straddle = None; self.pend = None
        self.hshr = 0; self.done = False; self.lhpx = None
        self.tlog = []; self.tid = 0; self.sk_d = 0
        self.skips = defaultdict(int)
        self.macro_dates = self._build_macro()

        self.schedule.on(self.date_rules.every_day(self.underlying),
            self.time_rules.after_market_open(self.underlying, 1), self.rday)
        self.schedule.on(self.date_rules.every_day(self.underlying),
            self.time_rules.before_market_close(self.underlying, self.eod_mins), self.feod)
        self.schedule.on(self.date_rules.every_day(self.underlying),
            self.time_rules.before_market_close(self.underlying, 1), self.cday)

    def _build_macro(self):
        dates = set()
        fomc = [
            "2020-01-29","2020-03-03","2020-03-15","2020-04-29","2020-06-10",
            "2020-07-29","2020-09-16","2020-11-05","2020-12-16",
            "2021-01-27","2021-03-17","2021-04-28","2021-06-16",
            "2021-07-28","2021-09-22","2021-11-03","2021-12-15",
            "2022-01-26","2022-03-16","2022-05-04","2022-06-15",
            "2022-07-27","2022-09-21","2022-11-02","2022-12-14",
            "2023-02-01","2023-03-22","2023-05-03","2023-06-14",
            "2023-07-26","2023-09-20","2023-11-01","2023-12-13",
            "2024-01-31","2024-03-20","2024-05-01","2024-06-12",
            "2024-07-31","2024-09-18","2024-11-07","2024-12-18",
            "2025-01-29","2025-03-19","2025-05-07","2025-06-18",
            "2025-07-30","2025-09-17","2025-11-05","2025-12-17",
            "2026-01-28","2026-03-18","2026-04-29","2026-06-17",
            "2026-07-29","2026-09-16","2026-11-04","2026-12-16"]
        cpi = [
            "2020-01-14","2020-02-13","2020-03-11","2020-04-10","2020-05-12",
            "2020-06-10","2020-07-14","2020-08-12","2020-09-11","2020-10-13",
            "2020-11-12","2020-12-10",
            "2021-01-13","2021-02-10","2021-03-10","2021-04-13","2021-05-12",
            "2021-06-10","2021-07-13","2021-08-11","2021-09-14","2021-10-13",
            "2021-11-10","2021-12-10",
            "2022-01-12","2022-02-10","2022-03-10","2022-04-12","2022-05-11",
            "2022-06-10","2022-07-13","2022-08-10","2022-09-13","2022-10-13",
            "2022-11-10","2022-12-13",
            "2023-01-12","2023-02-14","2023-03-14","2023-04-12","2023-05-10",
            "2023-06-13","2023-07-12","2023-08-10","2023-09-13","2023-10-12",
            "2023-11-14","2023-12-12",
            "2024-01-11","2024-02-13","2024-03-12","2024-04-10","2024-05-15",
            "2024-06-12","2024-07-11","2024-08-14","2024-09-11","2024-10-10",
            "2024-11-13","2024-12-11",
            "2025-01-15","2025-02-12","2025-03-12","2025-04-10","2025-05-13",
            "2025-06-11","2025-07-15","2025-08-12","2025-09-10","2025-10-14",
            "2025-11-12","2025-12-10",
            "2026-01-13","2026-02-11","2026-03-11","2026-04-14","2026-05-12",
            "2026-06-10","2026-07-14","2026-08-12","2026-09-15","2026-10-13",
            "2026-11-12","2026-12-10"]
        nfp = []
        for y in range(2020, 2027):
            for m in range(1, 13):
                d = datetime(y, m, 1)
                while d.weekday() != 4: d += timedelta(days=1)
                nfp.append(d.strftime("%Y-%m-%d"))
        for d in fomc + cpi + nfp:
            try: dates.add(datetime.strptime(d, "%Y-%m-%d").date())
            except: pass
        return dates

    def _cboe(self, s):
        if self.securities.contains_key(s):
            v = self.securities[s]
            if v and v.price > 0: return v.price
        return None

    def mid(self, c): return 0.5 * (c.bid_price + c.ask_price)

    def rv_calc(self, n=None):
        n = n or self.rv_lb
        if self.closes.count < n + 1: return None
        px = [self.closes[i] for i in range(n + 1)]; px.reverse()
        lr = [np.log(px[i]/px[i-1]) for i in range(1, len(px))
              if px[i-1] > 0 and px[i] > 0]
        return np.std(lr) * np.sqrt(252) if len(lr) >= n - 1 else None

    def har_rv(self):
        if self.closes.count < 23: return None
        px = [self.closes[i] for i in range(23)]; px.reverse()
        lr = [np.log(px[i]/px[i-1]) for i in range(1, len(px))
              if px[i-1] > 0 and px[i] > 0]
        if len(lr) < 22: return None
        rd = abs(lr[-1]) * np.sqrt(252)
        rw = np.std(lr[-5:]) * np.sqrt(252)
        rm = np.std(lr[-22:]) * np.sqrt(252)
        return max(0.01 + 0.35*rd + 0.30*rw + 0.25*rm, 0.01)

    def oret(self):
        if not self.prev_close or self.prev_close <= 0: return None
        p = self.securities[self.underlying].price
        return (p - self.prev_close) / self.prev_close if p > 0 else None

    def isig(self):
        if not self.straddle: return None
        iv = self.straddle["iv"]
        if iv <= 0: return None
        cn = self.time.hour * 60 + self.time.minute
        en = self.straddle["t_in"].hour * 60 + self.straddle["t_in"].minute
        f = max((cn - en) / 390.0, 0.001)
        return (iv / np.sqrt(252)) * np.sqrt(f) * self.securities[self.underlying].price

    def chk_signals(self, iv):
        if self.macro_on and self.time.date() in self.macro_dates:
            return False, "macro"
        if self.ts_on:
            vx, v3 = self._cboe(self.vix_sym), self._cboe(self.vix3m_sym)
            if vx and v3 and v3 > 0 and vx / v3 > 1.0 / self.min_vr:
                return False, f"bkwd={vx/v3:.3f}"
        if self.vv_on:
            vv = self._cboe(self.vvix_sym)
            if vv and vv > self.max_vv: return False, f"vvix={vv:.1f}"
        if self.on_on:
            o = self.oret()
            if o is not None and abs(o) > self.max_or: return False, f"oret={o:.4f}"
        if self.vrp_on:
            r = self.rv_calc()
            if r is not None and iv**2 - r**2 < self.min_vrp:
                return False, f"vrp={iv**2-r**2:.5f}"
        if self.har_on:
            h = self.har_rv()
            if h and iv < h * self.har_b: return False, f"har_iv={iv:.3f}_h={h:.3f}"
        if self.mom_on and self.p935:
            p = self.securities[self.underlying].price
            if self.p935 > 0 and p > 0:
                mv = abs(p - self.p935) / self.p935
                if mv > self.max_mm: return False, f"mom={mv:.4f}"
        if self.regime_on:
            rok, rinfo = self.rdf.entry_ok(iv)
            if not rok: return False, f"regime_tail={rinfo}"
        return True, None

    def rday(self):
        self.done = False; self.lhpx = None; self.ocap = False; self.p935 = None
        self.rdf.new_day()

    def cday(self):
        p = self.securities[self.underlying].price
        if p > 0: self.prev_close = p; self.closes.add(p)

    def on_data(self, data: Slice):
        px = self.securities[self.underlying].price
        if px > 0:
            self.rdf.update(px, self.time, vix=self._cboe(self.vix_sym))
        if not self.ocap and self.time.hour == 9 and self.time.minute >= 31:
            if px > 0: self.today_open = px; self.ocap = True
        if self.p935 is None and self.time.hour == 9 and self.time.minute >= 35:
            if px > 0: self.p935 = px
        if (self.time.hour == 9 and self.ew_s <= self.time.minute < self.ew_e
                and not self.straddle and not self.pend and not self.done):
            self.try_entry(data); self.done = True
        if self.pend: self.finalize()
        if not self.straddle: return
        if self.hedge_mode == "full" and self.lhpx: self.hedge_chk()
        self.exit_chk()

    def try_entry(self, data: Slice):
        if not data.option_chains.contains_key(self.option_symbol): return
        chain = data.option_chains[self.option_symbol]
        if not chain: return
        pair = self.find_atm(chain)
        if not pair: return
        call, put = pair
        civ, piv = call.implied_volatility, put.implied_volatility
        iv = (civ + piv) / 2.0
        if iv < self.min_iv: self.skips["iv_low"] += 1; return
        ok, why = self.chk_signals(iv)
        if not ok: self.skips[why] += 1; return
        cm, pm = self.mid(call), self.mid(put)
        sp = cm + pm
        if sp <= 0: return
        pv = self.portfolio.total_portfolio_value
        qty = max(1, min(int(pv * self.risk_pct / (sp * 100)), self.qty_cap))
        ed = -(call.greeks.delta + put.greeks.delta) * qty * 100
        if abs(ed) > self.max_entry_d: self.sk_d += 1; return
        t = self.tid
        vx = self._cboe(self.vix_sym); v3 = self._cboe(self.vix3m_sym)
        vv = self._cboe(self.vvix_sym); rv = self.rv_calc()
        hr = self.har_rv(); ov = self.oret(); rs = self.rdf.summary()
        self.market_order(call.symbol, -qty, tag=f"SC-{t}")
        self.market_order(put.symbol, -qty, tag=f"SP-{t}")
        self.pend = {
            "id": t, "call": call.symbol, "put": put.symbol,
            "qty": qty, "expiry": call.expiry, "strike": call.strike,
            "t_in": self.time, "iv": iv, "civ": civ, "piv": piv,
            "skew": piv - civ,
            "uin": self.securities[self.underlying].price,
            "dte": (call.expiry.date() - self.time.date()).days,
            "est_px": sp, "est_d": ed,
            "vix": vx, "vix3m": v3, "vvix": vv, "rv": rv, "har": hr,
            "vrp": (iv**2 - rv**2) if rv else None, "oret": ov,
            "regime": rs["regime"], "w_lap": rs["w_laplace"],
            "rdf_vol": rs["ann_vol"], "rdf_kurt": rs["ex_kurt"],
        }
        self.tid += 1

    def find_atm(self, chain):
        upx = self.securities[self.underlying].price
        if upx <= 0: return None
        val = [c for c in chain if c.greeks and c.bid_price > 0 and c.ask_price > 0]
        if not val: return None
        for exp in sorted(set(c.expiry for c in val)):
            se = [c for c in val if c.expiry == exp]
            stk = sorted(set(c.strike for c in se))
            if not stk: continue
            atm = min(stk, key=lambda k: abs(k - upx))
            ca = [c for c in se if c.right == OptionRight.CALL and c.strike == atm]
            pu = [c for c in se if c.right == OptionRight.PUT and c.strike == atm]
            if ca and pu: return ca[0], pu[0]
        return None

    def finalize(self):
        cp = self.portfolio[self.pend["call"]]
        pp = self.portfolio[self.pend["put"]]
        q = self.pend["qty"]
        if cp.quantity != -q or pp.quantity != -q: return
        credit = (cp.average_price + pp.average_price) * q * 100
        od = self._od(self.pend["call"], self.pend["put"], q)
        self.straddle = {
            "id": self.pend["id"],
            "call": self.pend["call"], "put": self.pend["put"],
            "qty": q, "strike": self.pend["strike"],
            "expiry": self.pend["expiry"], "t_in": self.pend["t_in"],
            "iv": self.pend["iv"], "civ": self.pend["civ"], "piv": self.pend["piv"],
            "skew": self.pend["skew"],
            "uin": self.pend["uin"], "dte": self.pend["dte"],
            "credit": credit, "init_d": od, "est_d": self.pend["est_d"],
            "hdg": 0, "hdg_sig": 0, "hdg_nse": 0,
            "vix": self.pend["vix"], "vix3m": self.pend["vix3m"],
            "vvix": self.pend["vvix"], "rv": self.pend["rv"],
            "har": self.pend["har"], "vrp": self.pend["vrp"],
            "oret": self.pend["oret"],
            "regime": self.pend.get("regime"), "w_lap": self.pend.get("w_lap"),
            "rdf_vol": self.pend.get("rdf_vol"), "rdf_kurt": self.pend.get("rdf_kurt"),
        }
        self.pend = None
        if self.hedge_mode in ["initial_only", "full"] and abs(od) > self.init_d_thresh:
            tgt = int(round(-od))
            if tgt != 0:
                self.market_order(self.underlying, tgt, tag=f"IH-{self.straddle['id']}")
                self.hshr = tgt
        self.lhpx = self.securities[self.underlying].price

    def _od(self, csym=None, psym=None, qty=None):
        if csym is None:
            if not self.straddle: return 0.0
            csym, psym, qty = self.straddle["call"], self.straddle["put"], self.straddle["qty"]
        chain = self.current_slice.option_chains.get(self.option_symbol) if self.current_slice else None
        if not chain: return 0.0
        d = 0.0
        for c in chain:
            if c.greeks is None: continue
            if c.symbol == csym or c.symbol == psym:
                d += -c.greeks.delta * qty * 100
        return d

    def hedge_chk(self):
        if not self.straddle: return
        px = self.securities[self.underlying].price
        if px <= 0 or not self.lhpx: return
        move_frac = (px - self.lhpx) / self.lhpx
        if self.regime_on:
            should_hedge, _ = self.rdf.hedge_signal(move_frac)
            if not should_hedge: return
        else:
            if abs(move_frac) < self.hband: return
        cp = self.portfolio[self.straddle["call"]]
        pp = self.portfolio[self.straddle["put"]]
        if not cp.invested or not pp.invested: return
        od = self._od()
        tgt = int(round(-od))
        trade = tgt - self.hshr
        if abs(trade) >= self.min_htrade:
            self.market_order(self.underlying, trade, tag=f"DH-{self.straddle['id']}")
            self.hshr = tgt; self.lhpx = px
            self.straddle["hdg"] += 1; self.straddle["hdg_sig"] += 1
        else:
            self.straddle["hdg_nse"] += 1

    def opnl(self):
        if not self.straddle: return 0.0
        return (self.portfolio[self.straddle["call"]].unrealized_profit +
                self.portfolio[self.straddle["put"]].unrealized_profit)

    def hpnl(self): return self.portfolio[self.underlying].unrealized_profit
    def tpnl(self): return self.opnl() + self.hpnl() if self.straddle else 0.0

    def exit_chk(self):
        if not self.straddle: return
        cp = self.portfolio[self.straddle["call"]]
        pp = self.portfolio[self.straddle["put"]]
        if not cp.invested and not pp.invested: self.clear(); return
        cr = self.straddle["credit"]
        if cr <= 0: return
        pct = self.tpnl() / cr
        reason = None
        if self.sig_on: reason = self._sig_stop()
        if not reason and pct >= self.pt: reason = f"PT {pct:.1%}"
        elif not reason and pct <= -self.sl: reason = f"SL {pct:.1%}"
        elif not reason and self._weak(): reason = f"WE {pct:.1%}"
        if reason: self.close(reason)

    def _sig_stop(self):
        if not self.straddle: return None
        p = self.securities[self.underlying].price
        e = self.straddle["uin"]
        if e <= 0 or p <= 0: return None
        s = self.isig()
        if not s or s <= 0: return None
        mult = self.rdf.dynamic_sigma_mult() if (self.regime_on and self.rdf.fitted) else self.sig_m
        mv = abs(p - e) / s
        return f"Sig {mv:.1f}s/{mult:.1f} r={self.rdf.regime}" if mv >= mult else None

    def _weak(self):
        if not self.straddle or not self.weak_on: return False
        cm = self.time.hour * 60 + self.time.minute
        if cm < self.w_hr * 60 + self.w_mn: return False
        cr = self.straddle["credit"]
        return cr > 0 and self.tpnl() / cr < self.w_pct

    def feod(self):
        if self.straddle or self.pend: self.close("EOD")

    def close(self, reason):
        if self.pend and not self.straddle:
            self.liquidate(tag=reason); self.pend = None; self.clear(); return
        if not self.straddle: return
        op = self.opnl(); hp = self.hpnl(); tp = op + hp
        ep = self.securities[self.underlying].price
        mn = (self.time - self.straddle["t_in"]).total_seconds() / 60.0
        self.liquidate(self.straddle["call"], tag=reason)
        self.liquidate(self.straddle["put"], tag=reason)
        if self.portfolio[self.underlying].invested:
            self.liquidate(self.underlying, tag=f"FH-{self.straddle['id']}")
        rs = self.rdf.summary()
        self.tlog.append({
            "id": self.straddle["id"],
            "t_in": str(self.straddle["t_in"]), "t_out": str(self.time),
            "dte": self.straddle["dte"], "strike": self.straddle["strike"],
            "uin": self.straddle["uin"], "uout": ep,
            "qty": self.straddle["qty"], "credit": self.straddle["credit"],
            "iv": self.straddle["iv"], "civ": self.straddle["civ"],
            "piv": self.straddle["piv"], "skew": self.straddle["skew"],
            "init_d": self.straddle["init_d"], "est_d": self.straddle["est_d"],
            "hdg": self.straddle["hdg"],
            "hdg_sig": self.straddle["hdg_sig"], "hdg_nse": self.straddle["hdg_nse"],
            "mins": mn, "op": op, "hp": hp, "pnl": tp, "reason": reason,
            "vix": self.straddle.get("vix"), "vix3m": self.straddle.get("vix3m"),
            "vvix": self.straddle.get("vvix"), "rv": self.straddle.get("rv"),
            "har": self.straddle.get("har"), "vrp": self.straddle.get("vrp"),
            "oret": self.straddle.get("oret"),
            "regime_in": self.straddle.get("regime"), "w_lap_in": self.straddle.get("w_lap"),
            "rdf_vol_in": self.straddle.get("rdf_vol"), "rdf_kurt_in": self.straddle.get("rdf_kurt"),
            "regime_out": rs["regime"], "w_lap_out": rs["w_laplace"],
            "rdf_vol_out": rs["ann_vol"], "rdf_kurt_out": rs["ex_kurt"],
        })
        self.clear()

    def clear(self):
        self.straddle = None; self.pend = None
        self.hshr = 0; self.lhpx = None

    def on_order_event(self, e): pass
    def on_assignment_order_event(self, e):
        self.liquidate(tag="Assign"); self.clear()

    def on_end_of_algorithm(self):
        if not self.tlog: self.log("NO TRADES"); return
        T = self.tlog
        W = [t for t in T if t["pnl"] > 0]
        L = [t for t in T if t["pnl"] <= 0]
        a = lambda v: sum(v)/len(v) if v else 0.0
        sa = lambda lst, k: a([t[k] for t in lst if t.get(k) is not None])
        tp = sum(t["pnl"] for t in T)

        self.log("=" * 70)
        self.log("SIGNAL FILTER SUMMARY")
        total_sk = sum(self.skips.values())
        self.log(f"Total skipped: {total_sk} | skipped_high_delta: {self.sk_d}")
        for k, v in sorted(self.skips.items(), key=lambda x: -x[1]):
            self.log(f"  {k}: {v} ({v/max(total_sk,1):.1%})")

        self.log("=" * 70)
        self.log(
            f"OVERVIEW | mode={self.hedge_mode} | trades={len(T)} | "
            f"W={len(W)} L={len(L)} wr={len(W)/len(T):.2%} | "
            f"total={tp:.2f} avg={a([t['pnl'] for t in T]):.2f} | "
            f"avg_opt={a([t['op'] for t in T]):.2f} avg_hdg={a([t['hp'] for t in T]):.2f} | "
            f"avg_hdg_count={a([t['hdg'] for t in T]):.2f}")
        self.log(
            f"W/L | avg_win={a([t['pnl'] for t in W]):.2f} "
            f"avg_loss={a([t['pnl'] for t in L]):.2f} | "
            f"win_hdg={a([t['hdg'] for t in W]):.2f} loss_hdg={a([t['hdg'] for t in L]):.2f} | "
            f"win_d={a([abs(t['init_d']) for t in W]):.2f} "
            f"loss_d={a([abs(t['init_d']) for t in L]):.2f}")
        self.log(
            f"SIGNALS ALL | vix={sa(T,'vix'):.2f} vvix={sa(T,'vvix'):.2f} "
            f"rv={sa(T,'rv'):.4f} vrp={sa(T,'vrp'):.5f} "
            f"har={sa(T,'har'):.4f} oret={sa(T,'oret'):.5f}")
        self.log(
            f"SIGNALS W  | vix={sa(W,'vix'):.2f} vvix={sa(W,'vvix'):.2f} "
            f"rv={sa(W,'rv'):.4f} vrp={sa(W,'vrp'):.5f} oret={sa(W,'oret'):.5f}")
        self.log(
            f"SIGNALS L  | vix={sa(L,'vix'):.2f} vvix={sa(L,'vvix'):.2f} "
            f"rv={sa(L,'rv'):.4f} vrp={sa(L,'vrp'):.5f} oret={sa(L,'oret'):.5f}")
        self.log(f"WEAK EXIT | on={self.weak_on} time={self.w_hr:02d}:{self.w_mn:02d} pct={self.w_pct:.2f}")

        zh = [t for t in T if t["hdg"] == 0]
        oh = [t for t in T if t["hdg"] >= 1]
        self.log(
            f"HEDGE | zero={len(zh)} avg={a([t['pnl'] for t in zh]):.2f} | "
            f"1+={len(oh)} avg={a([t['pnl'] for t in oh]):.2f}")
        by_hc = defaultdict(list); by_ex = defaultdict(list)
        for t in T: by_hc[t["hdg"]].append(t); by_ex[t["reason"]].append(t)
        for hc in sorted(by_hc):
            b = by_hc[hc]; w = [t for t in b if t["pnl"] > 0]
            self.log(
                f"  HDG={hc} | n={len(b)} wr={len(w)/len(b):.2%} "
                f"avg={a([t['pnl'] for t in b]):.2f} "
                f"opt={a([t['op'] for t in b]):.2f} hdg={a([t['hp'] for t in b]):.2f}")
        for r in sorted(by_ex):
            b = by_ex[r]
            self.log(
                f"  EXIT {r} | n={len(b)} avg={a([t['pnl'] for t in b]):.2f} "
                f"opt={a([t['op'] for t in b]):.2f} hdg={a([t['hp'] for t in b]):.2f}")

        rs = self.rdf.summary()
        self.log("=" * 70)
        self.log("REGIME DISTRIBUTION FILTER")
        self.log(f"  Hedge: sig={rs['hedge_signal']} nse={rs['hedge_noise']} fb={rs['hedge_fallback']}")
        self.log(f"  Regimes: {rs['regime_counts']}")
        self.log(f"  Final: r={rs['regime']} w={rs['w_laplace']:.2f} "
                 f"vol={rs['ann_vol']:.4f} kurt={rs['ex_kurt']:.2f} "
                 f"lap_b={rs['lap_b']:.6f} t_df={rs['t_df']:.1f}")
        by_rin = defaultdict(list)
        for t in T: by_rin[t.get("regime_in", "unk")].append(t)
        for reg in sorted(by_rin):
            b = by_rin[reg]; wb = [t for t in b if t["pnl"] > 0]
            self.log(
                f"  IN={reg} | n={len(b)} wr={len(wb)/max(len(b),1):.2%} "
                f"avg={a([t['pnl'] for t in b]):.2f} hdg={a([t['hdg'] for t in b]):.1f} "
                f"iv={sa(b,'iv'):.3f} w_lap={sa(b,'w_lap_in'):.2f}")
        self.log(
            f"  Per-trade: hdg_sig={a([t['hdg_sig'] for t in T]):.2f} "
            f"hdg_nse={a([t['hdg_nse'] for t in T]):.2f}")
        self.log("=" * 70)
        self.log(
            f"PARAMS | window={self.ew_s}-{self.ew_e} min_iv={self.min_iv} "
            f"min_vrp={self.min_vrp} max_oret={self.max_or} max_vvix={self.max_vv} "
            f"min_vr={self.min_vr} sig_m={self.sig_m} har_b={self.har_b} max_mom={self.max_mm}")
        self.log(
            f"REGIME | on={self.regime_on} hz={self.rdf.horizon_min}m "
            f"iw={self.rdf.intra_window} cw={self.rdf.carry_window} "
            f"pc={self.rdf.pct_calm}/{self.rdf.pct_normal}/{self.rdf.pct_stress} "
            f"tc={self.rdf.tail_ratio_cap}")
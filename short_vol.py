from AlgorithmImports import *
from collections import defaultdict


class ShortStraddleOnly(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2026, 2, 28)
        self.set_cash(1000000)

        self.underlying = self.add_equity("QQQ", Resolution.MINUTE).symbol

        option = self.add_option("QQQ", Resolution.MINUTE)
        option.set_filter(lambda u: u.strikes(-5, 5).expiration(0, 0))
        self.option_symbol = option.symbol

        # Strategy parameters
        self.min_iv_short = 0.12
        self.short_profit_target = 1.00
        self.short_stop_loss = 1.00
        self.short_hedge_band = 0.005
        self.eod_close_mins = 15

        # Hedge mode
        self.hedge_mode = "full"

        # Sizing
        self.risk_pct = 0.005
        self.qty_cap = 10
        self.initial_delta_hedge_threshold = 25
        self.min_hedge_trade = 5

        # Entry filter
        self.max_entry_abs_delta = 100

        # Time based weak-trade exit
        self.enable_weak_trade_exit = True
        self.weak_exit_hour = 12
        self.weak_exit_minute = 30
        self.weak_exit_pnl_pct = 0.25

        # Trade state
        self.straddle = None
        self.pending_entry = None
        self.hedge_shares = 0
        self.entry_done_today = False
        self.last_hedge_price = None

        # Diagnostics
        self.trade_log = []
        self.current_trade_id = 0
        self.skipped_high_delta = 0

        self.schedule.on(
            self.date_rules.every_day(self.underlying),
            self.time_rules.after_market_open(self.underlying, 1),
            self.reset_daily_flag
        )

        self.schedule.on(
            self.date_rules.every_day(self.underlying),
            self.time_rules.before_market_close(self.underlying, self.eod_close_mins),
            self.force_close_all
        )

    def reset_daily_flag(self):
        self.entry_done_today = False
        self.last_hedge_price = None

    def on_data(self, slice: Slice):
        if (
            self.time.hour == 9
            and 35 <= self.time.minute < 40
            and self.straddle is None
            and self.pending_entry is None
            and not self.entry_done_today
        ):
            self.check_entry(slice)
            self.entry_done_today = True

        if self.pending_entry is not None:
            self.try_finalize_entry()

        if self.straddle is None:
            return

        if self.hedge_mode == "full" and self.last_hedge_price is not None:
            self.threshold_hedge_check()

        self.check_intraday_exit()

    def check_entry(self, slice: Slice):
        if not slice.option_chains.contains_key(self.option_symbol):
            return

        chain = slice.option_chains[self.option_symbol]
        if not chain:
            return

        pair = self.select_true_atm_straddle(chain)
        if pair is None:
            return

        call, put = pair

        call_iv = call.implied_volatility
        put_iv = put.implied_volatility
        implied_vol = (call_iv + put_iv) / 2.0
        skew = put_iv - call_iv
        underlying_price = self.securities[self.underlying].price
        dte = (call.expiry.date() - self.time.date()).days

        if implied_vol < self.min_iv_short:
            return

        call_mid = self.mid_price(call)
        put_mid = self.mid_price(put)
        est_straddle_price = call_mid + put_mid

        if est_straddle_price <= 0:
            return

        portfolio_value = self.portfolio.total_portfolio_value
        qty = int((portfolio_value * self.risk_pct) / (est_straddle_price * 100))
        qty = max(1, min(qty, self.qty_cap))

        # Pre-trade delta filter
        est_options_delta = -(call.greeks.delta + put.greeks.delta) * qty * 100
        if abs(est_options_delta) > self.max_entry_abs_delta:
            self.skipped_high_delta += 1
            return

        trade_id = self.current_trade_id

        self.market_order(call.symbol, -qty, tag=f"ShortCall-{trade_id}")
        self.market_order(put.symbol, -qty, tag=f"ShortPut-{trade_id}")

        self.pending_entry = {
            "trade_id": trade_id,
            "call": call.symbol,
            "put": put.symbol,
            "qty": qty,
            "expiry": call.expiry,
            "strike": call.strike,
            "entry_time": self.time,
            "implied_vol": implied_vol,
            "call_iv": call_iv,
            "put_iv": put_iv,
            "skew": skew,
            "underlying_entry": underlying_price,
            "dte": dte,
            "est_straddle_price": est_straddle_price,
            "est_initial_delta": est_options_delta
        }

        self.current_trade_id += 1

    def select_true_atm_straddle(self, chain):
        underlying_price = self.securities[self.underlying].price
        if underlying_price <= 0:
            return None

        valid = [
            c for c in chain
            if c.greeks is not None
            and c.bid_price > 0
            and c.ask_price > 0
        ]

        if not valid:
            return None

        expiries = sorted(set(c.expiry for c in valid))

        for expiry in expiries:
            same_expiry = [c for c in valid if c.expiry == expiry]
            strikes = sorted(set(c.strike for c in same_expiry))

            if not strikes:
                continue

            atm_strike = min(strikes, key=lambda k: abs(k - underlying_price))

            call_candidates = [
                c for c in same_expiry
                if c.right == OptionRight.CALL and c.strike == atm_strike
            ]
            put_candidates = [
                c for c in same_expiry
                if c.right == OptionRight.PUT and c.strike == atm_strike
            ]

            if call_candidates and put_candidates:
                return call_candidates[0], put_candidates[0]

        return None

    def try_finalize_entry(self):
        call_pos = self.portfolio[self.pending_entry["call"]]
        put_pos = self.portfolio[self.pending_entry["put"]]
        qty = self.pending_entry["qty"]

        if call_pos.quantity != -qty or put_pos.quantity != -qty:
            return

        call_entry = call_pos.average_price
        put_entry = put_pos.average_price
        entry_credit = (call_entry + put_entry) * qty * 100

        options_delta = self.get_pending_options_delta(
            self.pending_entry["call"],
            self.pending_entry["put"],
            qty
        )

        self.straddle = {
            "trade_id": self.pending_entry["trade_id"],
            "call": self.pending_entry["call"],
            "put": self.pending_entry["put"],
            "qty": qty,
            "strike": self.pending_entry["strike"],
            "expiry": self.pending_entry["expiry"],
            "entry_time": self.pending_entry["entry_time"],
            "implied_vol": self.pending_entry["implied_vol"],
            "call_iv": self.pending_entry["call_iv"],
            "put_iv": self.pending_entry["put_iv"],
            "skew": self.pending_entry["skew"],
            "underlying_entry": self.pending_entry["underlying_entry"],
            "dte": self.pending_entry["dte"],
            "entry_credit": entry_credit,
            "initial_options_delta": options_delta,
            "estimated_entry_delta": self.pending_entry["est_initial_delta"],
            "hedge_count": 0
        }

        self.pending_entry = None

        if self.hedge_mode in ["initial_only", "full"]:
            if abs(options_delta) > self.initial_delta_hedge_threshold:
                target_hedge = int(round(-options_delta))
                if target_hedge != 0:
                    self.market_order(
                        self.underlying,
                        target_hedge,
                        tag=f"InitialHedge-{self.straddle['trade_id']}"
                    )
                    self.hedge_shares = target_hedge

        self.last_hedge_price = self.securities[self.underlying].price

    def get_pending_options_delta(self, call_symbol, put_symbol, qty):
        chain = self.current_slice.option_chains.get(self.option_symbol) if self.current_slice else None
        if not chain:
            return 0.0

        total_delta = 0.0
        for c in chain:
            if c.greeks is None:
                continue
            if c.symbol == call_symbol:
                total_delta += -c.greeks.delta * qty * 100
            elif c.symbol == put_symbol:
                total_delta += -c.greeks.delta * qty * 100

        return total_delta

    def get_options_delta(self):
        if self.straddle is None:
            return 0.0

        chain = self.current_slice.option_chains.get(self.option_symbol) if self.current_slice else None
        if not chain:
            return 0.0

        qty = self.straddle["qty"]
        total_delta = 0.0

        for c in chain:
            if c.greeks is None:
                continue
            if c.symbol == self.straddle["call"]:
                total_delta += -c.greeks.delta * qty * 100
            elif c.symbol == self.straddle["put"]:
                total_delta += -c.greeks.delta * qty * 100

        return total_delta

    def threshold_hedge_check(self):
        if self.straddle is None:
            return

        current_price = self.securities[self.underlying].price
        if current_price <= 0 or self.last_hedge_price is None:
            return

        price_move = abs(current_price - self.last_hedge_price) / self.last_hedge_price
        if price_move < self.short_hedge_band:
            return

        call_pos = self.portfolio[self.straddle["call"]]
        put_pos = self.portfolio[self.straddle["put"]]

        if not call_pos.invested or not put_pos.invested:
            return

        options_delta = self.get_options_delta()
        target_hedge = int(round(-options_delta))
        qty_to_trade = target_hedge - self.hedge_shares

        if abs(qty_to_trade) >= self.min_hedge_trade:
            self.market_order(
                self.underlying,
                qty_to_trade,
                tag=f"DeltaHedge-{self.straddle['trade_id']}"
            )
            self.hedge_shares = target_hedge
            self.last_hedge_price = current_price
            self.straddle["hedge_count"] += 1

    def get_option_legs_pnl(self):
        if self.straddle is None:
            return 0.0
        return (
            self.portfolio[self.straddle["call"]].unrealized_profit +
            self.portfolio[self.straddle["put"]].unrealized_profit
        )

    def get_hedge_pnl(self):
        return self.portfolio[self.underlying].unrealized_profit

    def get_total_trade_pnl(self):
        if self.straddle is None:
            return 0.0
        return self.get_option_legs_pnl() + self.get_hedge_pnl()

    def should_exit_weak_trade(self):
        if self.straddle is None or not self.enable_weak_trade_exit:
            return False

        current_minutes = self.time.hour * 60 + self.time.minute
        weak_exit_minutes = self.weak_exit_hour * 60 + self.weak_exit_minute

        if current_minutes < weak_exit_minutes:
            return False

        entry_credit = self.straddle["entry_credit"]
        if entry_credit <= 0:
            return False

        total_trade_pnl = self.get_total_trade_pnl()
        pnl_pct_of_credit = total_trade_pnl / entry_credit

        return pnl_pct_of_credit < self.weak_exit_pnl_pct

    def check_intraday_exit(self):
        if self.straddle is None:
            return

        call_pos = self.portfolio[self.straddle["call"]]
        put_pos = self.portfolio[self.straddle["put"]]

        if not call_pos.invested and not put_pos.invested:
            self.clear_trade_state()
            return

        entry_credit = self.straddle["entry_credit"]
        if entry_credit <= 0:
            return

        total_trade_pnl = self.get_total_trade_pnl()
        pnl_pct_of_credit = total_trade_pnl / entry_credit

        exit_reason = None

        if pnl_pct_of_credit >= self.short_profit_target:
            exit_reason = f"ProfitTarget {pnl_pct_of_credit:.1%}"
        elif pnl_pct_of_credit <= -self.short_stop_loss:
            exit_reason = f"StopLoss {pnl_pct_of_credit:.1%}"
        elif self.should_exit_weak_trade():
            exit_reason = f"WeakTradeExit {pnl_pct_of_credit:.1%}"

        if exit_reason:
            self.close_position(exit_reason)

    def force_close_all(self):
        if self.straddle is not None or self.pending_entry is not None:
            self.close_position("EOD-Flat")

    def close_position(self, reason):
        if self.pending_entry is not None and self.straddle is None:
            self.liquidate(tag=reason)
            self.pending_entry = None
            self.clear_trade_state()
            return

        if self.straddle is None:
            return

        option_pnl = self.get_option_legs_pnl()
        hedge_pnl = self.get_hedge_pnl()
        total_trade_pnl = option_pnl + hedge_pnl
        exit_spot = self.securities[self.underlying].price
        holding_minutes = (self.time - self.straddle["entry_time"]).total_seconds() / 60.0

        self.liquidate(self.straddle["call"], tag=reason)
        self.liquidate(self.straddle["put"], tag=reason)

        if self.portfolio[self.underlying].invested:
            self.liquidate(self.underlying, tag=f"FlattenHedge-{self.straddle['trade_id']}")

        trade_summary = {
            "trade_id": self.straddle["trade_id"],
            "entry_time": str(self.straddle["entry_time"]),
            "exit_time": str(self.time),
            "dte": self.straddle["dte"],
            "strike": self.straddle["strike"],
            "underlying_entry": self.straddle["underlying_entry"],
            "underlying_exit": exit_spot,
            "qty": self.straddle["qty"],
            "entry_credit": self.straddle["entry_credit"],
            "avg_iv": self.straddle["implied_vol"],
            "call_iv": self.straddle["call_iv"],
            "put_iv": self.straddle["put_iv"],
            "skew": self.straddle["skew"],
            "initial_options_delta": self.straddle["initial_options_delta"],
            "estimated_entry_delta": self.straddle["estimated_entry_delta"],
            "hedge_count": self.straddle["hedge_count"],
            "holding_minutes": holding_minutes,
            "option_pnl": option_pnl,
            "hedge_pnl": hedge_pnl,
            "total_trade_pnl": total_trade_pnl,
            "exit_reason": reason
        }

        self.trade_log.append(trade_summary)
        self.clear_trade_state()

    def clear_trade_state(self):
        self.straddle = None
        self.pending_entry = None
        self.hedge_shares = 0
        self.last_hedge_price = None

    def mid_price(self, contract):
        return 0.5 * (contract.bid_price + contract.ask_price)

    def on_order_event(self, order_event: OrderEvent):
        pass

    def on_assignment_order_event(self, assignment_event: OrderEvent):
        self.liquidate(tag="AssignmentExit")
        self.clear_trade_state()

    def on_end_of_algorithm(self):
        self.print_final_summary()

    def print_final_summary(self):
        if not self.trade_log:
            self.log("FINAL SUMMARY | no trades logged")
            return

        trades = self.trade_log
        winners = [t for t in trades if t["total_trade_pnl"] > 0]
        losers = [t for t in trades if t["total_trade_pnl"] <= 0]

        def avg(values):
            return sum(values) / len(values) if values else 0.0

        self.log(
            f"FINAL SUMMARY | hedge_mode={self.hedge_mode} | trades={len(trades)} | "
            f"skipped_high_delta={self.skipped_high_delta} | "
            f"winners={len(winners)} | losers={len(losers)} | "
            f"win_rate={len(winners)/len(trades):.2%} | "
            f"avg_total_pnl={avg([t['total_trade_pnl'] for t in trades]):.2f} | "
            f"avg_option_pnl={avg([t['option_pnl'] for t in trades]):.2f} | "
            f"avg_hedge_pnl={avg([t['hedge_pnl'] for t in trades]):.2f} | "
            f"avg_hedges={avg([t['hedge_count'] for t in trades]):.2f}"
        )

        self.log(
            f"WINNERS VS LOSERS | "
            f"avg_win={avg([t['total_trade_pnl'] for t in winners]):.2f} | "
            f"avg_loss={avg([t['total_trade_pnl'] for t in losers]):.2f} | "
            f"avg_win_hedges={avg([t['hedge_count'] for t in winners]):.2f} | "
            f"avg_loss_hedges={avg([t['hedge_count'] for t in losers]):.2f} | "
            f"avg_win_init_delta={avg([abs(t['initial_options_delta']) for t in winners]):.2f} | "
            f"avg_loss_init_delta={avg([abs(t['initial_options_delta']) for t in losers]):.2f}"
        )

        self.log(
            f"WEAK EXIT PARAMETERS | enabled={self.enable_weak_trade_exit} | "
            f"weak_exit_time={self.weak_exit_hour:02d}:{self.weak_exit_minute:02d} | "
            f"weak_exit_pnl_pct={self.weak_exit_pnl_pct:.2f}"
        )

        zero_hedge = [t for t in trades if t["hedge_count"] == 0]
        one_plus_hedge = [t for t in trades if t["hedge_count"] >= 1]

        self.log(
            f"HEDGE USAGE | zero_hedge_trades={len(zero_hedge)} | "
            f"one_plus_hedge_trades={len(one_plus_hedge)} | "
            f"avg_zero_hedge_pnl={avg([t['total_trade_pnl'] for t in zero_hedge]):.2f} | "
            f"avg_one_plus_hedge_pnl={avg([t['total_trade_pnl'] for t in one_plus_hedge]):.2f}"
        )

        by_hedge_count = defaultdict(list)
        by_exit_reason = defaultdict(list)

        for t in trades:
            by_hedge_count[t["hedge_count"]].append(t)
            by_exit_reason[t["exit_reason"]].append(t)

        for hc in sorted(by_hedge_count.keys()):
            bucket = by_hedge_count[hc]
            self.log(
                f"BY HEDGE COUNT | hedges={hc} | count={len(bucket)} | "
                f"win_rate={len([t for t in bucket if t['total_trade_pnl'] > 0]) / len(bucket):.2%} | "
                f"avg_total_pnl={avg([t['total_trade_pnl'] for t in bucket]):.2f} | "
                f"avg_option_pnl={avg([t['option_pnl'] for t in bucket]):.2f} | "
                f"avg_hedge_pnl={avg([t['hedge_pnl'] for t in bucket]):.2f}"
            )

        for reason in sorted(by_exit_reason.keys()):
            bucket = by_exit_reason[reason]
            self.log(
                f"BY EXIT REASON | reason={reason} | count={len(bucket)} | "
                f"avg_total_pnl={avg([t['total_trade_pnl'] for t in bucket]):.2f} | "
                f"avg_option_pnl={avg([t['option_pnl'] for t in bucket]):.2f} | "
                f"avg_hedge_pnl={avg([t['hedge_pnl'] for t in bucket]):.2f}"
            )
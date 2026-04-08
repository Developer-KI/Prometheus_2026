# =============================================================================
# Intraday Option Straddle Momentum Strategy
# Based on: Muravyev, Da, Goyenko & Zhang — "Intraday Option Return: A Tale
#           of Two Momentum" (University of Notre Dame, 2025)
#
# KEY INSIGHT:
#   - Morning Momentum (9:35–10:00 AM): Yesterday's morning straddle winners
#     continue winning this morning. Driven by under-reaction to overnight
#     volatility news. NO reversal → fundamental signal.
#   - Afternoon Momentum (3:30–4:00 PM): Yesterday's afternoon straddle winners
#     continue winning this afternoon. Driven by persistent OMM inventory
#     pressure. Reversal follows → transitory signal.
#
# STRATEGY:
#   Each day, compute ATM straddle returns for the first 30 min (morning) and
#   last 30 min (afternoon) for a liquid universe. Sort into quintiles based on
#   yesterday's returns. Go long top quintile, short bottom quintile in the
#   same window the next day. Unwind after 25 min.
#
# UNIVERSE: Top ~50 most liquid S&P 500 option names (by open interest).
#           Rebalanced monthly.
#
# NOTE: Does not work well
# =============================================================================

from AlgorithmImports import *

class IntradayOptionMomentum(QCAlgorithm):

    def Initialize(self):
        # ── Backtest period & cash ──────────────────────────────────────────
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(1_000_000)

        # ── Parameters ──────────────────────────────────────────────────────
        self.num_stocks = 30            # Universe size (liquid names)
        self.quintile_size = 6          # ~top/bottom 20% of 30
        self.morning_enabled = True     # Toggle morning momentum leg
        self.afternoon_enabled = True   # Toggle afternoon momentum leg

        # ATM delta band (|delta| between 0.375 and 0.625 per paper)
        self.delta_lo = 0.375
        self.delta_hi = 0.625

        # Days to expiry window (30-180 per paper)
        self.dte_lo = 30
        self.dte_hi = 180

        # ── Time windows (Eastern) ──────────────────────────────────────────
        # Morning window: 9:35 → 10:00  (paper skips first 5 min)
        # Afternoon window: 15:30 → 16:00
        self.morning_open = time(9, 35)
        self.morning_close = time(10, 0)
        self.afternoon_open = time(15, 30)
        self.afternoon_close = time(15, 59)  # close before market close

        # ── Data structures ─────────────────────────────────────────────────
        # Track straddle mid-prices at window boundaries
        self.morning_entry_prices = {}    # symbol → mid at 9:35
        self.morning_exit_prices = {}     # symbol → mid at 10:00
        self.afternoon_entry_prices = {}  # symbol → mid at 15:30
        self.afternoon_exit_prices = {}   # symbol → mid at 16:00

        # Yesterday's returns for signal generation
        self.prev_morning_returns = {}    # symbol → return
        self.prev_afternoon_returns = {}  # symbol → return

        # Today's returns (accumulated during the day, stored at EOD)
        self.today_morning_returns = {}
        self.today_afternoon_returns = {}

        # Track active option contracts per underlying
        self.option_contracts = {}  # equity symbol → list of (call, put) pairs

        # ── Universe selection ──────────────────────────────────────────────
        self.UniverseSettings.Resolution = Resolution.Minute
        self.AddUniverse(self.CoarseSelection)

        self.active_symbols = []
        self.options_added = set()

        # ── Scheduling ──────────────────────────────────────────────────────
        # Morning momentum entry & exit
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(9, 35),
            self.MorningEntry
        )
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(10, 0),
            self.MorningExit
        )

        # Afternoon momentum entry & exit
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(15, 30),
            self.AfternoonEntry
        )
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(15, 58),
            self.AfternoonExit
        )

        # End-of-day: store today's returns as signals for tomorrow
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(15, 59),
            self.EndOfDay
        )

        # Warm-up
        self.SetWarmUp(5, Resolution.Daily)

        # Benchmark
        self.SetBenchmark("SPY")

        # Track performance
        self.morning_pnl = 0
        self.afternoon_pnl = 0
        self.trade_count = 0

    # ════════════════════════════════════════════════════════════════════════
    # UNIVERSE SELECTION
    # ════════════════════════════════════════════════════════════════════════
    def CoarseSelection(self, coarse):
        """Select top liquid, large-cap stocks (S&P 500 proxy)."""
        if self.Time.day != 1:
            return Universe.Unchanged

        filtered = [x for x in coarse
                    if x.HasFundamentalData
                    and x.Price > 20
                    and x.DollarVolume > 10_000_000]

        sorted_by_volume = sorted(filtered,
                                  key=lambda x: x.DollarVolume,
                                  reverse=True)

        symbols = [x.Symbol for x in sorted_by_volume[:self.num_stocks]]
        self.active_symbols = symbols
        return symbols

    def OnSecuritiesChanged(self, changes):
        """Add option chains for new universe members."""
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol.SecurityType == SecurityType.Equity:
                if symbol not in self.options_added:
                    option = self.AddOption(symbol.Value, Resolution.Minute)
                    option.SetFilter(lambda u: u
                        .IncludeWeeklys()
                        .Strikes(-5, 5)
                        .Expiration(self.dte_lo, self.dte_hi))
                    self.options_added.add(symbol)

        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.options_added:
                self.options_added.discard(symbol)

    # ════════════════════════════════════════════════════════════════════════
    # STRADDLE HELPERS
    # ════════════════════════════════════════════════════════════════════════
    def GetATMStraddle(self, equity_symbol):
        """
        Find the best ATM straddle for a given equity.
        Returns (call_symbol, put_symbol, straddle_mid_price) or None.

        ATM defined as |delta| in [0.375, 0.625] per the paper.
        We pick the strike closest to the current stock price within
        the DTE window.
        """
        chain = self.CurrentSlice.OptionChains
        if chain is None:
            return None

        # Find the option chain for this equity
        for kvp in chain:
            option_chain = kvp.Value
            underlying = option_chain.Underlying
            if underlying is None:
                continue

            underlying_symbol = underlying.Symbol
            # Match by underlying ticker
            if underlying_symbol.Value != equity_symbol.Value:
                continue

            stock_price = underlying.Price
            if stock_price <= 0:
                continue

            # Filter contracts by DTE
            contracts = [c for c in option_chain
                        if self.dte_lo <= (c.Expiry - self.Time).days <= self.dte_hi
                        and c.BidPrice > 0.50
                        and c.AskPrice > 0.50
                        and (c.AskPrice - c.BidPrice) < 3.0]  # Spread filter

            if not contracts:
                continue

            # Group by strike
            strikes = set(c.Strike for c in contracts)
            best_straddle = None
            best_distance = float('inf')

            for strike in strikes:
                calls = [c for c in contracts
                        if c.Right == OptionRight.Call and c.Strike == strike]
                puts = [c for c in contracts
                       if c.Right == OptionRight.Put and c.Strike == strike]

                if not calls or not puts:
                    continue

                # Pick the contract pair with the closest expiry (most liquid)
                call = min(calls, key=lambda c: (c.Expiry - self.Time).days)
                put = min(puts, key=lambda c: (c.Expiry - self.Time).days)

                # Check ATM-ness via delta or distance to stock price
                distance = abs(strike - stock_price)
                if distance < best_distance:
                    best_distance = distance
                    call_mid = (call.BidPrice + call.AskPrice) / 2
                    put_mid = (put.BidPrice + put.AskPrice) / 2
                    straddle_mid = call_mid + put_mid

                    if straddle_mid > 0:
                        best_straddle = (call.Symbol, put.Symbol, straddle_mid)

            if best_straddle:
                return best_straddle

        return None

    def GetStraddleMidPrice(self, call_symbol, put_symbol):
        """Get current mid-price of a straddle."""
        call_data = self.Securities.get(call_symbol)
        put_data = self.Securities.get(put_symbol)

        if call_data is None or put_data is None:
            return None

        call_bid = call_data.BidPrice
        call_ask = call_data.AskPrice
        put_bid = put_data.BidPrice
        put_ask = put_data.AskPrice

        if call_bid <= 0 or call_ask <= 0 or put_bid <= 0 or put_ask <= 0:
            return None

        return (call_bid + call_ask) / 2 + (put_bid + put_ask) / 2

    # ════════════════════════════════════════════════════════════════════════
    # MORNING MOMENTUM
    # ════════════════════════════════════════════════════════════════════════
    def MorningEntry(self):
        """
        At 9:35 AM: Record straddle prices AND enter positions based on
        yesterday's morning return signal.
        """
        if self.IsWarmingUp:
            return

        # ── Step 1: Snapshot straddle prices at 9:35 for today's signal ────
        self.morning_entry_prices = {}
        straddle_map = {}  # symbol → (call, put)

        for symbol in self.active_symbols:
            result = self.GetATMStraddle(symbol)
            if result:
                call_sym, put_sym, mid = result
                self.morning_entry_prices[symbol] = mid
                straddle_map[symbol] = (call_sym, put_sym)

        # ── Step 2: Trade on yesterday's signal ────────────────────────────
        if not self.morning_enabled or not self.prev_morning_returns:
            return

        # Sort by previous morning return
        sorted_returns = sorted(self.prev_morning_returns.items(),
                                key=lambda x: x[1])

        n = min(self.quintile_size, len(sorted_returns) // 5)
        if n < 1:
            return

        losers = [s for s, r in sorted_returns[:n]]    # Short these
        winners = [s for s, r in sorted_returns[-n:]]   # Long these

        # Equal-weight allocation per leg
        capital_per_leg = self.Portfolio.TotalPortfolioValue * 0.15
        capital_per_name = capital_per_leg / n if n > 0 else 0

        for symbol in winners:
            if symbol in straddle_map:
                call_sym, put_sym = straddle_map[symbol]
                mid = self.morning_entry_prices.get(symbol, 0)
                if mid > 0:
                    qty = max(1, int(capital_per_name / (mid * 100)))
                    self.MarketOrder(call_sym, qty, tag="MornMom_Long_Call")
                    self.MarketOrder(put_sym, qty, tag="MornMom_Long_Put")
                    self.trade_count += 2

        for symbol in losers:
            if symbol in straddle_map:
                call_sym, put_sym = straddle_map[symbol]
                mid = self.morning_entry_prices.get(symbol, 0)
                if mid > 0:
                    qty = max(1, int(capital_per_name / (mid * 100)))
                    self.MarketOrder(call_sym, -qty, tag="MornMom_Short_Call")
                    self.MarketOrder(put_sym, -qty, tag="MornMom_Short_Put")
                    self.trade_count += 2

    def MorningExit(self):
        """
        At 10:00 AM: Unwind all morning momentum positions and record
        today's morning return for tomorrow's signal.
        """
        if self.IsWarmingUp:
            return

        # ── Record today's morning returns ─────────────────────────────────
        self.today_morning_returns = {}
        for symbol in self.active_symbols:
            if symbol in self.morning_entry_prices:
                entry_mid = self.morning_entry_prices[symbol]
                result = self.GetATMStraddle(symbol)
                if result:
                    _, _, exit_mid = result
                    if entry_mid > 0:
                        ret = (exit_mid - entry_mid) / entry_mid
                        self.today_morning_returns[symbol] = ret

        # ── Liquidate morning positions ────────────────────────────────────
        for kvp in self.Portfolio:
            holding = kvp.Value
            if holding.Invested and holding.Symbol.SecurityType == SecurityType.Option:
                tag = ""
                if hasattr(holding, 'Tag'):
                    tag = holding.Tag
                # Liquidate all option positions (morning + any residual)
                self.Liquidate(holding.Symbol, tag="MornMom_Exit")

    # ════════════════════════════════════════════════════════════════════════
    # AFTERNOON MOMENTUM
    # ════════════════════════════════════════════════════════════════════════
    def AfternoonEntry(self):
        """
        At 3:30 PM: Record straddle prices AND enter positions based on
        yesterday's afternoon return signal.
        """
        if self.IsWarmingUp:
            return

        # ── Step 1: Snapshot straddle prices ───────────────────────────────
        self.afternoon_entry_prices = {}
        straddle_map = {}

        for symbol in self.active_symbols:
            result = self.GetATMStraddle(symbol)
            if result:
                call_sym, put_sym, mid = result
                self.afternoon_entry_prices[symbol] = mid
                straddle_map[symbol] = (call_sym, put_sym)

        # ── Step 2: Trade on yesterday's signal ────────────────────────────
        if not self.afternoon_enabled or not self.prev_afternoon_returns:
            return

        sorted_returns = sorted(self.prev_afternoon_returns.items(),
                                key=lambda x: x[1])

        n = min(self.quintile_size, len(sorted_returns) // 5)
        if n < 1:
            return

        losers = [s for s, r in sorted_returns[:n]]
        winners = [s for s, r in sorted_returns[-n:]]

        # Smaller allocation for afternoon (weaker signal per paper)
        capital_per_leg = self.Portfolio.TotalPortfolioValue * 0.08
        capital_per_name = capital_per_leg / n if n > 0 else 0

        for symbol in winners:
            if symbol in straddle_map:
                call_sym, put_sym = straddle_map[symbol]
                mid = self.afternoon_entry_prices.get(symbol, 0)
                if mid > 0:
                    qty = max(1, int(capital_per_name / (mid * 100)))
                    self.MarketOrder(call_sym, qty, tag="AftMom_Long_Call")
                    self.MarketOrder(put_sym, qty, tag="AftMom_Long_Put")
                    self.trade_count += 2

        for symbol in losers:
            if symbol in straddle_map:
                call_sym, put_sym = straddle_map[symbol]
                mid = self.afternoon_entry_prices.get(symbol, 0)
                if mid > 0:
                    qty = max(1, int(capital_per_name / (mid * 100)))
                    self.MarketOrder(call_sym, -qty, tag="AftMom_Short_Call")
                    self.MarketOrder(put_sym, -qty, tag="AftMom_Short_Put")
                    self.trade_count += 2

    def AfternoonExit(self):
        """
        At 3:58 PM: Unwind all afternoon momentum positions and record
        today's afternoon return.
        """
        if self.IsWarmingUp:
            return

        # ── Record today's afternoon returns ───────────────────────────────
        self.today_afternoon_returns = {}
        for symbol in self.active_symbols:
            if symbol in self.afternoon_entry_prices:
                entry_mid = self.afternoon_entry_prices[symbol]
                result = self.GetATMStraddle(symbol)
                if result:
                    _, _, exit_mid = result
                    if entry_mid > 0:
                        ret = (exit_mid - entry_mid) / entry_mid
                        self.today_afternoon_returns[symbol] = ret

        # ── Liquidate afternoon positions ──────────────────────────────────
        for kvp in self.Portfolio:
            holding = kvp.Value
            if holding.Invested and holding.Symbol.SecurityType == SecurityType.Option:
                self.Liquidate(holding.Symbol, tag="AftMom_Exit")

    # ════════════════════════════════════════════════════════════════════════
    # END OF DAY
    # ════════════════════════════════════════════════════════════════════════
    def EndOfDay(self):
        """Store today's returns as tomorrow's signal. Ensure flat."""
        if self.IsWarmingUp:
            return

        # Roll signals forward
        if self.today_morning_returns:
            self.prev_morning_returns = dict(self.today_morning_returns)
        if self.today_afternoon_returns:
            self.prev_afternoon_returns = dict(self.today_afternoon_returns)

        # Safety: liquidate any residual positions
        self.Liquidate(tag="EOD_Cleanup")

        # Log daily stats
        if self.Time.day == 1:
            self.Log(f"[{self.Time.date()}] Portfolio: ${self.Portfolio.TotalPortfolioValue:,.0f} "
                     f"| Trades: {self.trade_count} "
                     f"| Morning signals: {len(self.prev_morning_returns)} "
                     f"| Afternoon signals: {len(self.prev_afternoon_returns)}")

    # ════════════════════════════════════════════════════════════════════════
    # RISK MANAGEMENT
    # ════════════════════════════════════════════════════════════════════════
    def OnOrderEvent(self, orderEvent):
        """Track fills for logging."""
        if orderEvent.Status == OrderStatus.Filled:
            pass  # Could add detailed trade logging here

    def OnEndOfAlgorithm(self):
        self.Log(f"=== FINAL RESULTS ===")
        self.Log(f"Total Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        self.Log(f"Total Trades: {self.trade_count}")
        self.Log(f"Morning signals tracked: {len(self.prev_morning_returns)}")
        self.Log(f"Afternoon signals tracked: {len(self.prev_afternoon_returns)}")
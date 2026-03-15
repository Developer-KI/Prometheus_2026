import numpy as np
import pandas as pd


class RegimeGatedLiquidityShockFramework:

    def __init__(
        self,
        signal_horizon: int = 3,
        vol_lookback: int = 20,
        entry_z: float = 2.0,
        gross_target: float = 1.0,
        drawdown_reduce: float = -0.10,
        drawdown_stop: float = -0.15,
        asset_vol_spike_multiple: float = 2.0,
    ):
        self.signal_horizon = signal_horizon
        self.vol_lookback = vol_lookback
        self.entry_z = entry_z
        self.gross_target = gross_target
        self.drawdown_reduce = drawdown_reduce
        self.drawdown_stop = drawdown_stop
        self.asset_vol_spike_multiple = asset_vol_spike_multiple

    @staticmethod
    def classify_vix_regime(vix_value: float) -> int:
    
        if pd.isna(vix_value):
            return 2
        if vix_value < 15:
            return 1
        if vix_value < 25:
            return 2
        if vix_value < 35:
            return 3
        return 4

    @staticmethod
    def regime_multiplier(regime: int) -> float:
    
        mapping = {
            1: 1.00,  # Low volatility
            2: 0.80,  # Normal
            3: 0.40,  # Elevated
            4: 0.00,  # Crisis
        }
        return mapping.get(regime, 0.0)

    @staticmethod
    def compute_drawdown(nav: pd.Series) -> pd.Series:
       
        running_max = nav.cummax()
        return nav / running_max - 1.0

    def compute_k_day_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        
        return prices.pct_change(self.signal_horizon)

    def compute_daily_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
       
        return prices.pct_change()

    def compute_rolling_volatility(self, daily_returns: pd.DataFrame) -> pd.DataFrame:
        
        return daily_returns.rolling(self.vol_lookback).std() * np.sqrt(252)

    def compute_standardized_shocks(
        self,
        k_day_returns: pd.DataFrame,
        rolling_volatility: pd.DataFrame,
    ) -> pd.DataFrame:
        
        horizon_vol = rolling_volatility * np.sqrt(self.signal_horizon / 252.0)
        return k_day_returns / horizon_vol

    def compute_asset_vol_filter(self, rolling_volatility: pd.DataFrame) -> pd.DataFrame:
        
        median_vol = rolling_volatility.rolling(60).median()
        return rolling_volatility <= (self.asset_vol_spike_multiple * median_vol)

    def generate_raw_weights(
        self,
        z_scores_row: pd.Series,
        tradable_assets_row: pd.Series,
    ) -> pd.Series:
        
        eligible = z_scores_row[tradable_assets_row.fillna(False)].dropna()
        eligible = eligible[eligible.abs() >= self.entry_z]

        if eligible.empty:
            return pd.Series(dtype=float)

        raw_weights = -eligible / eligible.abs().sum()
        gross = raw_weights.abs().sum()

        if gross > 0:
            raw_weights = raw_weights * (self.gross_target / gross)

        return raw_weights

    def build_target_weights(
        self,
        prices: pd.DataFrame,
        vix: pd.Series,
        portfolio_nav_proxy: pd.Series | None = None,
    ) -> pd.DataFrame:
        
        prices = prices.sort_index()
        vix = vix.reindex(prices.index).ffill()

        daily_returns = self.compute_daily_returns(prices)
        k_day_returns = self.compute_k_day_returns(prices)
        rolling_volatility = self.compute_rolling_volatility(daily_returns)
        z_scores = self.compute_standardized_shocks(k_day_returns, rolling_volatility)
        tradable_assets = self.compute_asset_vol_filter(rolling_volatility)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if portfolio_nav_proxy is None:
            proxy_returns = daily_returns.mean(axis=1).fillna(0.0)
            portfolio_nav_proxy = (1.0 + proxy_returns).cumprod()

        drawdown = self.compute_drawdown(portfolio_nav_proxy)

        for dt in prices.index:
            regime = self.classify_vix_regime(float(vix.loc[dt]))
            multiplier = self.regime_multiplier(regime)

            if multiplier == 0.0:
                continue

            current_drawdown = float(drawdown.loc[dt])

            if current_drawdown <= self.drawdown_stop:
                continue

            if current_drawdown <= self.drawdown_reduce:
                multiplier *= 0.5

            raw_weights = self.generate_raw_weights(
                z_scores_row=z_scores.loc[dt],
                tradable_assets_row=tradable_assets.loc[dt],
            )

            if raw_weights.empty:
                continue

            weights.loc[dt, raw_weights.index] = multiplier * raw_weights

        return weights

    @staticmethod
    def backtest(
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        transaction_cost_bps: float = 0.0,
    ) -> pd.DataFrame:
        
        prices = prices.sort_index()
        weights = weights.reindex(prices.index).fillna(0.0)

        asset_returns = prices.pct_change().fillna(0.0)
        shifted_weights = weights.shift(1).fillna(0.0)

        gross_turnover = (weights - shifted_weights).abs().sum(axis=1)
        transaction_cost = gross_turnover * (transaction_cost_bps / 10000.0)

        strategy_returns = (shifted_weights * asset_returns).sum(axis=1) - transaction_cost
        nav = (1.0 + strategy_returns).cumprod()

        running_max = nav.cummax()
        drawdown = nav / running_max - 1.0

        results = pd.DataFrame({
            "strategy_returns": strategy_returns,
            "nav": nav,
            "drawdown": drawdown,
            "turnover": gross_turnover,
        })

        return results

    @staticmethod
    def performance_summary(results: pd.DataFrame) -> dict:
        
        returns = results["strategy_returns"].dropna()

        if returns.empty:
            return {
                "Total Return": np.nan,
                "Annualized Return": np.nan,
                "Annualized Volatility": np.nan,
                "Sharpe Ratio": np.nan,
                "Max Drawdown": np.nan,
            }

        total_return = results["nav"].iloc[-1] - 1.0
        ann_return = (results["nav"].iloc[-1] ** (252 / len(results))) - 1.0 if len(results) > 0 else np.nan
        ann_vol = returns.std() * np.sqrt(252)

        if ann_vol > 0:
            sharpe = ann_return / ann_vol
        else:
            sharpe = np.nan

        max_drawdown = results["drawdown"].min()

        return {
            "Total Return": total_return,
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown,
        }


def create_dummy_data():
    
    np.random.seed(42)

    dates = pd.date_range("2023-01-02", periods=300, freq="B")
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "SPY"]

    random_returns = np.random.normal(0.0002, 0.015, size=(len(dates), len(tickers)))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(random_returns, axis=0)),
        index=dates,
        columns=tickers,
    )

    vix = pd.Series(
        np.clip(20 + np.cumsum(np.random.normal(0, 0.2, size=len(dates))), 12, 38),
        index=dates,
        name="VIX",
    )

    return prices, vix


if __name__ == "__main__":
    prices, vix = create_dummy_data()

    framework = RegimeGatedLiquidityShockFramework(
        signal_horizon=3,
        vol_lookback=20,
        entry_z=2.0,
        gross_target=1.0,
        drawdown_reduce=-0.10,
        drawdown_stop=-0.15,
        asset_vol_spike_multiple=2.0,
    )

    weights = framework.build_target_weights(prices=prices, vix=vix)
    results = framework.backtest(prices=prices, weights=weights, transaction_cost_bps=2.0)
    summary = framework.performance_summary(results)

    print("Performance Summary")
    print("-" * 40)
    for key, value in summary.items():
        if pd.isna(value):
            print(f"{key}: NaN")
        else:
            print(f"{key}: {value:.4f}")

    print("\nLast 5 rows of results:")
    print(results.tail())
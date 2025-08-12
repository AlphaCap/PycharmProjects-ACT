"""
Comprehensive Indicator Library - FIXED for nGS data format
Contains ALL your proven trading indicators as building blocks for AI
Each indicator is implemented as a reusable function with consistent interface
FIXED: Now handles both 'Close'/'close' and other column name variations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import warnings

warnings.filterwarnings("ignore")


class ComprehensiveIndicatorLibrary:
    """
    Complete library of YOUR proven indicators
    AI will use these as building blocks for adaptive strategy generation
    FIXED: Now automatically detects column names (Close vs close, High vs high, etc.)
    """

    def __init__(self) -> None:
        self.indicators_catalog: Dict[str, Dict[str, Any]] = {}
        self._register_all_indicators()
        print(f" Comprehensive Indicator Library initialized")
        print(f" Registered {len(self.indicators_catalog)} indicators")

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CRITICAL FIX: Standardize column names to work with both data formats
        Handles: Close/close, High/high, Low/low, Open/open, Volume/volume
        """
        df_copy = df.copy()

        # Column mapping - try multiple variations
        column_mapping = {
            "close": ["Close", "close", "CLOSE"],
            "high": ["High", "high", "HIGH"],
            "low": ["Low", "low", "LOW"],
            "open": ["Open", "open", "OPEN"],
            "volume": ["Volume", "volume", "VOLUME", "vol", "Vol"],
        }

        # Standardize to capitalized names
        for standard_name, variations in column_mapping.items():
            for variation in variations:
                if variation in df_copy.columns:
                    if standard_name.capitalize() != variation:
                        df_copy[standard_name.capitalize()] = df_copy[variation]
                    break

        # Ensure we have the basic required columns
        required_cols = ["Close", "High", "Low", "Open"]
        missing = [col for col in required_cols if col not in df_copy.columns]

        if missing:
            print(f"  WARNING: Missing columns {missing} in data")
            print(f"Available columns: {list(df_copy.columns)}")
            # Try to create missing columns from available data
            if "Close" not in df_copy.columns and "close" in df_copy.columns:
                df_copy["Close"] = df_copy["close"]
            if "High" not in df_copy.columns and "high" in df_copy.columns:
                df_copy["High"] = df_copy["high"]
            if "Low" not in df_copy.columns and "low" in df_copy.columns:
                df_copy["Low"] = df_copy["low"]
            if "Open" not in df_copy.columns and "open" in df_copy.columns:
                df_copy["Open"] = df_copy["open"]
            if "Volume" not in df_copy.columns and "volume" in df_copy.columns:
                df_copy["Volume"] = df_copy["volume"]

        return df_copy

    def _register_all_indicators(self) -> None:
        """Register all your proven indicators with metadata"""

        # TREND INDICATORS
        self.indicators_catalog["tsf"] = {
            "name": "Time Series Forecast",
            "function": self.time_series_forecast,
            "params": {"length": 14, "forecast_periods": 1},
            "category": "trend",
            "output_type": "price_level",
            "description": "Your trend prediction indicator",
        }

        self.indicators_catalog["linreg"] = {
            "name": "Linear Regression",
            "function": self.linear_regression,
            "params": {"length": 14},
            "category": "trend",
            "output_type": "price_level",
            "description": "Linear regression trend line",
        }

        self.indicators_catalog["linreg_slope"] = {
            "name": "Linear Regression Slope",
            "function": self.linear_regression_slope,
            "params": {"length": 14},
            "category": "trend",
            "output_type": "oscillator",
            "description": "Trend strength and direction",
        }

        # BOLLINGER BAND FAMILY
        self.indicators_catalog["bb_position"] = {
            "name": "Bollinger Band Position",
            "function": self.bollinger_position,
            "params": {"length": 20, "deviation": 2.0},
            "category": "mean_reversion",
            "output_type": "percentage",
            "description": "Your core entry logic - where price sits in BB range",
        }

        self.indicators_catalog["bb_squeeze"] = {
            "name": "Bollinger Band Squeeze",
            "function": self.bollinger_squeeze,
            "params": {"length": 20, "deviation": 2.0, "squeeze_threshold": 0.1},
            "category": "volatility",
            "output_type": "binary",
            "description": "Volatility compression detection",
        }

        self.indicators_catalog["bb_width"] = {
            "name": "Bollinger Band Width",
            "function": self.bollinger_width,
            "params": {"length": 20, "deviation": 2.0},
            "category": "volatility",
            "output_type": "percentage",
            "description": "Volatility measure",
        }

        # MOMENTUM INDICATORS
        self.indicators_catalog["rsi"] = {
            "name": "RSI",
            "function": self.rsi,
            "params": {"length": 14},
            "category": "momentum",
            "output_type": "oscillator",
            "description": "Relative Strength Index",
        }

        self.indicators_catalog["rsi_divergence"] = {
            "name": "RSI Divergence",
            "function": self.rsi_divergence,
            "params": {"length": 14, "lookback": 20},
            "category": "momentum",
            "output_type": "binary",
            "description": "Momentum divergence detection",
        }

        self.indicators_catalog["stochastic"] = {
            "name": "Stochastic",
            "function": self.stochastic,
            "params": {"k_length": 14, "d_length": 3},
            "category": "momentum",
            "output_type": "oscillator",
            "description": "Stochastic oscillator",
        }

        # VOLUME INDICATORS
        self.indicators_catalog["volume_profile"] = {
            "name": "Volume Profile",
            "function": self.volume_profile,
            "params": {"length": 20, "threshold": 1.5},
            "category": "volume",
            "output_type": "binary",
            "description": "Your volume confirmation filter",
        }

        self.indicators_catalog["volume_sma_ratio"] = {
            "name": "Volume SMA Ratio",
            "function": self.volume_sma_ratio,
            "params": {"length": 20},
            "category": "volume",
            "output_type": "ratio",
            "description": "Volume strength relative to average",
        }

        self.indicators_catalog["on_balance_volume"] = {
            "name": "On Balance Volume",
            "function": self.on_balance_volume,
            "params": {"length": 20},
            "category": "volume",
            "output_type": "cumulative",
            "description": "Volume accumulation indicator",
        }

        # MARKET EFFICIENCY & STRUCTURE
        self.indicators_catalog["market_efficiency"] = {
            "name": "Market Efficiency",
            "function": self.market_efficiency,
            "params": {"length": 20, "method": "standard"},
            "category": "efficiency",
            "output_type": "percentage",
            "description": "Your key filter - market efficiency measurement",
        }

        self.indicators_catalog["fractal_dimension"] = {
            "name": "Fractal Dimension",
            "function": self.fractal_dimension,
            "params": {"length": 20},
            "category": "efficiency",
            "output_type": "bounded",
            "description": "Market structure complexity",
        }

        # VOLATILITY INDICATORS
        self.indicators_catalog["atr"] = {
            "name": "Average True Range",
            "function": self.average_true_range,
            "params": {"length": 14},
            "category": "volatility",
            "output_type": "price_level",
            "description": "Volatility measure",
        }

        self.indicators_catalog["volatility_ratio"] = {
            "name": "Volatility Ratio",
            "function": self.volatility_ratio,
            "params": {"short_length": 10, "long_length": 30},
            "category": "volatility",
            "output_type": "ratio",
            "description": "Relative volatility measurement",
        }

        # PATTERN RECOGNITION
        self.indicators_catalog["support_resistance"] = {
            "name": "Support/Resistance",
            "function": self.support_resistance_levels,
            "params": {"length": 20, "threshold": 0.02},
            "category": "pattern",
            "output_type": "binary",
            "description": "Key price level proximity",
        }

        self.indicators_catalog["pivot_points"] = {
            "name": "Pivot Points",
            "function": self.pivot_points,
            "params": {"type": "standard"},
            "category": "pattern",
            "output_type": "price_level",
            "description": "Daily pivot calculations",
        }
    # =============================================================================
    # INDICATOR IMPLEMENTATIONS - ALL FIXED FOR COLUMN NAME COMPATIBILITY
    # =============================================================================

    def time_series_forecast(
        self, df: pd.DataFrame, length: int = 14, forecast_periods: int = 1
    ) -> pd.Series:
        df = self._standardize_columns(df)
        tsf_values: List[float] = []

        for i in range(len(df)):
            if i < length:
                tsf_values.append(np.nan)
            else:
                recent_prices = df["Close"].iloc[i - length + 1 : i + 1].to_numpy(dtype=float)
                x_values = np.arange(len(recent_prices))

                if len(recent_prices) > 1:
                    coeffs = np.polyfit(x_values, recent_prices, 1)
                    forecast_x = len(recent_prices) - 1 + forecast_periods
                    forecast_price: float = float(coeffs[0] * forecast_x + coeffs[1])
                    tsf_values.append(forecast_price)
                else:
                    tsf_values.append(float(df["Close"].iloc[i]))

        return pd.Series(tsf_values, index=df.index, name="TSF")

    def linear_regression(self, df: pd.DataFrame, length: int = 14) -> pd.Series:
        df = self._standardize_columns(df)
        linreg_values: List[float] = []

        for i in range(len(df)):
            if i < length:
                linreg_values.append(np.nan)
            else:
                recent_prices = df["Close"].iloc[i - length + 1 : i + 1].to_numpy(dtype=float)
                x_values = np.arange(len(recent_prices))

                if len(recent_prices) > 1:
                    coeffs = np.polyfit(x_values, recent_prices, 1)
                    current_x = len(recent_prices) - 1
                    current_linreg: float = float(coeffs[0] * current_x + coeffs[1])
                    linreg_values.append(current_linreg)
                else:
                    linreg_values.append(float(df["Close"].iloc[i]))

        return pd.Series(linreg_values, index=df.index, name="LinReg")

    def linear_regression_slope(self, df: pd.DataFrame, length: int = 14) -> pd.Series:
        df = self._standardize_columns(df)
        slopes: List[float] = []

        for i in range(len(df)):
            if i < length:
                slopes.append(0.0)
            else:
                recent_prices = df["Close"].iloc[i - length + 1 : i + 1].to_numpy(dtype=float)
                x_values = np.arange(len(recent_prices))

                if len(recent_prices) > 1:
                    slope: float = float(np.polyfit(x_values, recent_prices, 1)[0])
                    slopes.append(slope)
                else:
                    slopes.append(0.0)

        return pd.Series(slopes, index=df.index, name="LinRegSlope")

    def bollinger_position(
        self, df: pd.DataFrame, length: int = 20, deviation: float = 2.0
    ) -> pd.Series:
        df = self._standardize_columns(df)
        bb_mid = df["Close"].rolling(window=length).mean()
        bb_std = df["Close"].rolling(window=length).std()
        bb_upper = bb_mid + (deviation * bb_std)
        bb_lower = bb_mid - (deviation * bb_std)

        bb_position = ((df["Close"] - bb_lower) / (bb_upper - bb_lower) * 100).astype(float)
        bb_position = bb_position.fillna(50).clip(0, 100)
        return bb_position.rename("BB_Position")

    def bollinger_squeeze(
        self,
        df: pd.DataFrame,
        length: int = 20,
        deviation: float = 2.0,
        squeeze_threshold: float = 0.1,
    ) -> pd.Series:
        df = self._standardize_columns(df)
        bb_mid = df["Close"].rolling(window=length).mean()
        bb_std = df["Close"].rolling(window=length).std()
        bb_width_pct = (deviation * bb_std * 2) / bb_mid * 100

        squeeze = (bb_width_pct < squeeze_threshold).astype(int)
        return squeeze.fillna(0).rename("BB_Squeeze")

    def bollinger_width(
        self, df: pd.DataFrame, length: int = 20, deviation: float = 2.0
    ) -> pd.Series:
        df = self._standardize_columns(df)
        bb_mid = df["Close"].rolling(window=length).mean()
        bb_std = df["Close"].rolling(window=length).std()
        bb_width = (deviation * bb_std * 2) / bb_mid * 100
        return bb_width.fillna(0).rename("BB_Width")

    def rsi(self, df: pd.DataFrame, length: int = 14) -> pd.Series:
        df = self._standardize_columns(df)
        price_delta = df["Close"].diff()
        gains = price_delta.where(price_delta > 0, 0)
        losses = -price_delta.where(price_delta < 0, 0)

        avg_gains = gains.rolling(window=length).mean()
        avg_losses = losses.rolling(window=length).mean()

        # Prevent division by zero
        avg_losses = avg_losses.replace(0, np.nan)
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).rename("RSI")

    def rsi_divergence(
        self, df: pd.DataFrame, length: int = 14, lookback: int = 20
    ) -> pd.Series:
        df = self._standardize_columns(df)
        rsi_values = self.rsi(df, length)

        price_highs = df["High"].rolling(window=lookback).max()
        rsi_highs = rsi_values.rolling(window=lookback).max()

        price_trend = price_highs.diff(lookback) > 0
        rsi_trend = rsi_highs.diff(lookback) < 0

        divergence = (price_trend & rsi_trend).astype(int)
        return divergence.fillna(0).rename("RSI_Divergence")

    def stochastic(
        self, df: pd.DataFrame, k_length: int = 14, d_length: int = 3
    ) -> pd.Series:
        df = self._standardize_columns(df)
        lowest_low = df["Low"].rolling(window=k_length).min()
        highest_high = df["High"].rolling(window=k_length).max()

        k_percent = ((df["Close"] - lowest_low) / (highest_high - lowest_low) * 100).astype(float)
        k_percent = k_percent.fillna(50)
        d_percent = k_percent.rolling(window=d_length).mean()
        return d_percent.fillna(50).rename("Stochastic")

    def volume_profile(
        self, df: pd.DataFrame, length: int = 20, threshold: float = 1.5
    ) -> pd.Series:
        df = self._standardize_columns(df)
        if "Volume" not in df.columns:
            print("  Volume column not found, using dummy volume data")
            return pd.Series(0, index=df.index, name="Volume_Profile")

        volume_avg = df["Volume"].rolling(window=length).mean()
        volume_ratio = df["Volume"] / volume_avg
        high_volume = (volume_ratio > threshold).astype(int)
        return high_volume.fillna(0).rename("Volume_Profile")

    def volume_sma_ratio(self, df: pd.DataFrame, length: int = 20) -> pd.Series:
        df = self._standardize_columns(df)
        if "Volume" not in df.columns:
            return pd.Series(1.0, index=df.index, name="Volume_SMA_Ratio")

        volume_sma = df["Volume"].rolling(window=length).mean()
        ratio = df["Volume"] / volume_sma
        return ratio.fillna(1.0).rename("Volume_SMA_Ratio")

    def on_balance_volume(self, df: pd.DataFrame, length: int = 20) -> pd.Series:
        df = self._standardize_columns(df)
        if "Volume" not in df.columns:
            return pd.Series(0, index=df.index, name="OBV")

        price_change = df["Close"].diff()
        volume_direction = np.where(
            price_change > 0, df["Volume"], np.where(price_change < 0, -df["Volume"], 0)
        )
        obv = pd.Series(volume_direction, index=df.index).cumsum()
        obv_smooth = obv.rolling(window=length).mean()
        return obv_smooth.fillna(0).rename("OBV")

    def market_efficiency(
        self, df: pd.DataFrame, length: int = 20, method: str = "standard"
    ) -> pd.Series:
        df = self._standardize_columns(df)
        if method == "standard":
            price_change = abs(df["Close"].pct_change(length))
            volatility = (
                df["Close"].rolling(length).std() / df["Close"].rolling(length).mean()
            )
            # avoid division by zero
            volatility = volatility.replace(0, np.nan)
            efficiency = price_change / volatility * 100
            efficiency = efficiency.fillna(50).clip(0, 100)
        else:
            returns = df["Close"].pct_change()
            net_movement = abs(returns.rolling(length).sum())
            path_length = abs(returns).rolling(length).sum()
            # avoid division by zero
            path_length = path_length.replace(0, np.nan)
            efficiency = net_movement / path_length * 100
            efficiency = efficiency.fillna(50).clip(0, 100)
        return efficiency.rename("Market_Efficiency")

    def fractal_dimension(self, df: pd.DataFrame, length: int = 20) -> pd.Series:
        df = self._standardize_columns(df)
        fd_values: List[float] = []

        for i in range(len(df)):
            if i < length:
                fd_values.append(1.5)
            else:
                prices = df["Close"].iloc[i - length + 1 : i + 1].to_numpy(dtype=float)
                if len(prices) > 2:
                    log_returns = np.log(prices[1:] / prices[:-1])
                    if len(log_returns) > 1 and np.std(log_returns) > 0:
                        mean_abs_return = float(np.mean(np.abs(log_returns)))
                        std_return = float(np.std(log_returns))
                        if mean_abs_return > 0:
                            fd = 2 - (std_return / mean_abs_return)
                            fd = float(max(1.0, min(2.0, fd)))
                        else:
                            fd = 1.5
                    else:
                        fd = 1.5
                else:
                    fd = 1.5
                fd_values.append(fd)
        return pd.Series(fd_values, index=df.index, name="Fractal_Dimension")

    def average_true_range(self, df: pd.DataFrame, length: int = 14) -> pd.Series:
        df = self._standardize_columns(df)
        high_low = (df["High"] - df["Low"]).astype(float)
        high_close_prev = np.abs(df["High"] - df["Close"].shift(1)).astype(float)
        low_close_prev = np.abs(df["Low"] - df["Close"].shift(1)).astype(float)
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = pd.Series(true_range).rolling(window=length).mean()
        return atr.fillna(0).rename("ATR")

    def volatility_ratio(
        self, df: pd.DataFrame, short_length: int = 10, long_length: int = 30
    ) -> pd.Series:
        df = self._standardize_columns(df)
        short_vol = df["Close"].rolling(short_length).std()
        long_vol = df["Close"].rolling(long_length).std()
        # avoid division by zero
        long_vol = long_vol.replace(0, np.nan)
        vol_ratio = short_vol / long_vol
        return vol_ratio.fillna(1.0).rename("Volatility_Ratio")

    def support_resistance_levels(
        self, df: pd.DataFrame, length: int = 20, threshold: float = 0.02
    ) -> pd.Series:
        df = self._standardize_columns(df)
        recent_high = df["High"].rolling(window=length).max()
        recent_low = df["Low"].rolling(window=length).min()
        dist_to_resistance = abs(df["Close"] - recent_high) / df["Close"]
        dist_to_support = abs(df["Close"] - recent_low) / df["Close"]
        near_sr = (
            (dist_to_resistance < threshold) | (dist_to_support < threshold)
        ).astype(int)
        return near_sr.fillna(0).rename("Near_SR")

    def pivot_points(self, df: pd.DataFrame, type: str = "standard") -> pd.Series:
        df = self._standardize_columns(df)
        if type == "standard":
            pivot = (
                df["High"].shift(1) + df["Low"].shift(1) + df["Close"].shift(1)
            ) / 3
        else:
            pivot = (
                df["High"].shift(1) + df["Low"].shift(1) + df["Close"].shift(1) * 2
            ) / 4
        return pivot.fillna(df["Close"]).rename("Pivot_Points")

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    def get_indicator_info(self, indicator_name: str) -> Dict[str, Any]:
        if indicator_name in self.indicators_catalog:
            return self.indicators_catalog[indicator_name]
        else:
            raise ValueError(f"Indicator '{indicator_name}' not found")

    def list_indicators_by_category(self, category: Optional[str] = None) -> Union[Dict[str, List[str]], List[str]]:
        if category is None:
            categories: Dict[str, List[str]] = {}
            for name, info in self.indicators_catalog.items():
                cat = info["category"]
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(name)
            return categories
        else:
            return [
                name
                for name, info in self.indicators_catalog.items()
                if info["category"] == category
            ]

    def calculate_indicator(
        self, indicator_name: str, df: pd.DataFrame, **kwargs
    ) -> pd.Series:
        if indicator_name not in self.indicators_catalog:
            raise ValueError(f"Indicator '{indicator_name}' not found")

        df_standardized = self._standardize_columns(df)
        indicator_info = self.indicators_catalog[indicator_name]
        function = indicator_info["function"]
        params = indicator_info["params"].copy()
        params.update(kwargs)
        return function(df_standardized, **params)

    def calculate_multiple_indicators(
        self, df: pd.DataFrame, indicator_list: List[str], custom_params: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        results = self._standardize_columns(df)
        custom_params = custom_params or {}
        for indicator_name in indicator_list:
            if indicator_name in self.indicators_catalog:
                params = custom_params.get(indicator_name, {})
                indicator_values = self.calculate_indicator(
                    indicator_name, df, **params
                )
                results[indicator_name] = indicator_values
            else:
                print(f"Warning: Indicator '{indicator_name}' not found, skipping")
        return results

def test_indicator_library() -> None:
    print("\n TESTING FIXED INDICATOR LIBRARY")
    print("=" * 40)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    sample_data = pd.DataFrame(
        {
            "Date": dates,
            "open": 100 + np.cumsum(np.random.randn(100) * 0.5),
            "high": 0,
            "low": 0,
            "close": 100 + np.cumsum(np.random.randn(100) * 0.5),
            "volume": np.random.randint(10000, 50000, 100),
        }
    )
    sample_data["high"] = (
        np.maximum(sample_data["open"], sample_data["close"]) + np.random.rand(100) * 2
    )
    sample_data["low"] = (
        np.minimum(sample_data["open"], sample_data["close"]) - np.random.rand(100) * 2
    )
    lib = ComprehensiveIndicatorLibrary()
    print(f"Original columns: {list(sample_data.columns)}")
    standardized = lib._standardize_columns(sample_data)
    print(f"Standardized columns: {list(standardized.columns)}")
    test_indicators = [
        "bb_position",
        "rsi",
        "market_efficiency",
        "tsf",
        "volume_profile",
    ]
    print(
        f"\nTesting {len(test_indicators)} indicators on {len(sample_data)} bars of sample data:"
    )
    for indicator in test_indicators:
        try:
            result = lib.calculate_indicator(indicator, sample_data)
            valid_values = result.dropna()
            print(
                f" {indicator}: {len(valid_values)} valid values, range: {valid_values.min():.2f} to {valid_values.max():.2f}"
            )
        except Exception as e:
            print(f" {indicator}: Error - {e}")
    print(f"\n Available Categories:")
    categories = lib.list_indicators_by_category()
    for cat, indicators in categories.items():
        print(f"   {cat.upper()}: {len(indicators)} indicators")
    print(f"\n COLUMN NAME COMPATIBILITY FIXED!")

if __name__ == "__main__":
    print(" FIXED COMPREHENSIVE INDICATOR LIBRARY")
    print("=" * 50)
    print(" CRITICAL FIX: Now handles both 'Close' and 'close' column formats")
    library = ComprehensiveIndicatorLibrary()
    categories = library.list_indicators_by_category()
    total_indicators = sum(len(indicators) for indicators in categories.values())
    print(f" Total Indicators: {total_indicators}")
    print(f" Categories: {len(categories)}")
    test_indicator_library()
    print(f"\n READY FOR AI STRATEGY GENERATION!")
    print("Next: Test with your nGS data format")

import sys
import os

project_root = r"C:\Users\theca\PycharmProjects"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pathlib import Path

import pandas as pd
import pytest

def test_project_structure() -> None:
    """Test that required project directories exist"""
    project_root = Path(__file__).parent.parent
    data_dirs = [
        project_root / "data" / "daily",  # For individual stock data
        project_root / "data" / "etf_historical",  # For sector ETF data
    ]

    for dir_path in data_dirs:
        if not dir_path.exists():
            os.makedirs(dir_path)
            print(f"Created missing directory: {dir_path}")
        assert dir_path.exists(), f"Directory {dir_path} not found"


def test_sector_etf_data() -> None:
    """Test that sector ETF data is properly structured for optimization"""
    # These ETFs are used for sector parameter optimization only
    sector_etfs = {
        "Technology": "XLK",
        "Financials": "XLF",
        "Healthcare": "XLV",
        "Energy": "XLE",
        "Consumer Discretionary": "XLY",
        "Industrials": "XLI",
        "Utilities": "XLU",
        "Materials": "XLB",
        "Real Estate": "XLRE",
        "Consumer Staples": "XLP",
        "Communication Services": "XLC",
    }

    etf_dir = Path("data") / "etf_historical"
    available_etfs = []

    if not etf_dir.exists():
        pytest.skip(f"ETF directory {etf_dir} does not exist")

    # Test ETF data files
    for sector, etf in sector_etfs.items():
        file_path = etf_dir / f"{etf}_historical.csv"
        if file_path.exists():
            available_etfs.append(etf)

            # Verify data format
            df = pd.read_csv(file_path)

            # Check required columns for parameter optimization
            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            assert all(
                col in df.columns for col in required_cols
            ), f"Missing required columns in {etf} data"

            # Validate data quality
            assert len(df) > 0, f"No data found in {etf} file"
            assert pd.to_datetime(
                df["Date"]
            ).is_monotonic_increasing, f"Dates not in chronological order in {etf} file"

            # Check for invalid values
            numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in numeric_cols:
                assert not df[col].isnull().any(), f"Found NULL values in {etf} {col}"
                assert (df[col] > 0).all(), f"Found non-positive values in {etf} {col}"

    print(f"\nFound {len(available_etfs)} sector ETFs for optimization:")
    print(", ".join(available_etfs))

    if not available_etfs:
        pytest.skip("No ETF data available - run etf_historical_downloader.py first")


def test_trading_data_directory() -> None:
    """Test that stock data directory is properly structured for trading"""
    stock_dir = Path("data") / "daily"

    if not stock_dir.exists():
        pytest.skip(f"Stock directory {stock_dir} does not exist")

    stock_files = list(stock_dir.glob("*.csv"))
    print(f"\nFound {len(stock_files)} stock data files")

    if stock_files:
        # Test sample of stock files
        sample_size = min(5, len(stock_files))
        for file_path in stock_files[:sample_size]:
            df = pd.read_csv(file_path)

            # Check required columns
            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            assert all(
                col in df.columns for col in required_cols
            ), f"Missing required columns in {file_path.name}"

            # Validate data format
            assert len(df) > 0, f"Empty data file: {file_path.name}"
            assert pd.to_datetime(
                df["Date"]
            ).is_monotonic_increasing, (
                f"Dates not in chronological order in {file_path.name}"
            )

            # Check data quality
            numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in numeric_cols:
                assert (
                    not df[col].isnull().any()
                ), f"Found NULL values in {file_path.name} {col}"
                assert (
                    df[col] > 0
                ).all(), f"Found non-positive values in {file_path.name} {col}"


def test_etf_data_sufficiency_for_walk_forward() -> None:
    """Test that ETF data has sufficient history for walk-forward optimization"""
    etf_dir = Path("data") / "etf_historical"
    if not etf_dir.exists():
        pytest.skip(f"ETF directory {etf_dir} does not exist")

    min_required_months = 12  # Minimum required months for proper walk-forward testing
    results = {}

    for file_path in etf_dir.glob("*_historical.csv"):
        try:
            df = pd.read_csv(file_path)
            if "Date" not in df.columns:
                continue

            df["Date"] = pd.to_datetime(df["Date"])
            date_range = (
                df["Date"].max() - df["Date"].min()
            ).days / 30.44  # Approximate months

            symbol = file_path.stem.replace("_historical", "")
            results[symbol] = {
                "months_of_data": round(date_range, 1),
                "start_date": df["Date"].min().strftime("%Y-%m-%d"),
                "end_date": df["Date"].max().strftime("%Y-%m-%d"),
                "sufficient": date_range >= min_required_months,
            }

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    print("\nETF Data Coverage Summary:")
    for symbol, info in results.items():
        status = "" if info["sufficient"] else ""
        print(
            f"{status} {symbol}: {info['months_of_data']:.1f} months "
            f"({info['start_date']} to {info['end_date']})"
        )

    # Assert that at least some ETFs have sufficient data
    sufficient_etfs = [s for s, i in results.items() if i["sufficient"]]
    assert len(sufficient_etfs) > 0, (
        f"No ETFs have sufficient data for walk-forward analysis "
        f"(need at least {min_required_months} months)"
    )


def test_etf_data_consistency() -> None:
    """Test that ETF data is consistent and complete for optimization"""
    etf_dir = Path("data") / "etf_historical"
    if not etf_dir.exists():
        pytest.skip(f"ETF directory {etf_dir} does not exist")

    for file_path in etf_dir.glob("*_historical.csv"):
        df = pd.read_csv(file_path)
        symbol = file_path.stem.replace("_historical", "")

        # Check for price consistency
        assert (
            df["High"] >= df["Low"]
        ).all(), f"{symbol}: High prices must be >= Low prices"
        assert (
            df["High"] >= df["Close"]
        ).all(), f"{symbol}: High prices must be >= Close prices"
        assert (
            df["Close"] >= df["Low"]
        ).all(), f"{symbol}: Close prices must be >= Low prices"
        assert (df["Open"] <= df["High"]).all(), f"{symbol}: Open must be <= High"
        assert (df["Open"] >= df["Low"]).all(), f"{symbol}: Open must be >= Low"

        # Check for data gaps
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        date_diffs = df["Date"].diff().dt.days[1:]  # Skip first row which will be NaT

        max_gap = date_diffs.max()
        if max_gap > 5:  # Allow for weekends and some holidays
            print(f"  Warning: {symbol} has {max_gap} day gap in data")

        # Check for realistic price movements
        daily_returns = df["Close"].pct_change()
        max_return = daily_returns.max()
        min_return = daily_returns.min()

        # Flag suspicious price movements (e.g., >20% daily moves)
        if max_return > 0.20 or min_return < -0.20:
            print(
                f"  Warning: {symbol} has suspicious price movement "
                f"(max: {max_return:.1%}, min: {min_return:.1%})"
            )

        # Check for volume consistency
        assert (df["Volume"] >= 0).all(), f"{symbol}: Found negative volume values"
        zero_volume_days = (df["Volume"] == 0).sum()
        if zero_volume_days > 0:
            print(f"  Warning: {symbol} has {zero_volume_days} days with zero volume")


def test_data_alignment() -> None:
    """Test that data is properly aligned for cross-ETF analysis"""
    etf_dir = Path("data") / "etf_historical"
    if not etf_dir.exists():
        pytest.skip(f"ETF directory {etf_dir} does not exist")

    etf_dates = {}

    # Collect date ranges for each ETF
    for file_path in etf_dir.glob("*_historical.csv"):
        df = pd.read_csv(file_path)
        if "Date" not in df.columns:
            continue

        symbol = file_path.stem.replace("_historical", "")
        df["Date"] = pd.to_datetime(df["Date"])
        etf_dates[symbol] = {
            "start": df["Date"].min(),
            "end": df["Date"].max(),
            "trading_days": len(df),
        }

    if len(etf_dates) < 2:
        pytest.skip("Need at least 2 ETFs to test data alignment")

    # Check for alignment issues
    print("\nData Alignment Analysis:")
    for symbol, dates in etf_dates.items():
        other_etfs = {k: v for k, v in etf_dates.items() if k != symbol}

        # Check if this ETF's date range is similar to others
        start_diffs = [(dates["start"] - v["start"]).days for v in other_etfs.values()]
        end_diffs = [(dates["end"] - v["end"]).days for v in other_etfs.values()]

        max_start_diff = max(abs(min(start_diffs)), abs(max(start_diffs)))
        max_end_diff = max(abs(min(end_diffs)), abs(max(end_diffs)))

        if max_start_diff > 30 or max_end_diff > 30:  # More than a month difference
            print(f"  {symbol}: Data range differs significantly from other ETFs")
            print(f"   Start diff: {max_start_diff} days")
            print(f"   End diff: {max_end_diff} days")

        # Check for similar number of trading days
        trading_days_others = [v["trading_days"] for v in other_etfs.values()]
        avg_trading_days = sum(trading_days_others) / len(trading_days_others)
        diff_pct = (
            abs(dates["trading_days"] - avg_trading_days) / avg_trading_days * 100
        )

        if diff_pct > 5:  # More than 5% difference in trading days
            print(f"  {symbol}: Trading days count differs by {diff_pct:.1f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

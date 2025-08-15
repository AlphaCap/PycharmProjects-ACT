# verify_sp500_symbols.py
"""
Quick script to verify S&P 500 symbol coverage.

This script checks the existence and content of the S&P 500 symbols file,
ensuring the trading universe is adequately covered.
"""

import os
from typing import List

from data_manager import get_sp500_symbols, verify_sp500_coverage


def main() -> None:
    """
    Verify S&P 500 symbol coverage and display results.

    Raises:
        FileNotFoundError: If the symbols file is missing.
        ValueError: If no symbols are loaded.
    """
    print("S&P 500 Symbol Verification")
    print("=" * 40)

    # Check if symbols file exists
    symbols_file: str = "sp500_symbols.txt"
    if not os.path.exists(symbols_file):
        print("", symbols_file, "not found!")
        print("   You need to create this file with all S&P 500 symbols")
        return

    # Load and verify symbols
    symbols: List[str] = get_sp500_symbols()

    if not symbols:
        print(" No symbols loaded!")
        return

    print(" Loaded", len(symbols), "symbols")

    # Verify coverage
    coverage_ok: bool = verify_sp500_coverage()

    if coverage_ok:
        print(" S&P 500 coverage looks good")
    else:
        print("  S&P 500 coverage may be incomplete")

    # Show sample symbols
    print("\nFirst 10 symbols:", symbols[:10])
    print("Last 10 symbols:", symbols[-10:])

    # Check for common symbols
    common_symbols: List[str] = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "META",
        "NVDA",
        "BRK.B",
        "UNH",
        "JNJ",
    ]
    found_common: List[str] = [s for s in common_symbols if s in symbols]
    missing_common: List[str] = [s for s in common_symbols if s not in symbols]

    print("\nCommon symbols found:", found_common)
    if missing_common:
        print("Missing common symbols:", missing_common)

    print("\nTo ensure all S&P 500 symbols are being scanned:")
    print("1. Verify", symbols_file, "contains all current S&P 500 symbols")
    print("2. Check that your data scanning uses get_sp500_symbols()")
    print("3. Expected count: ~500 symbols (currently:", len(symbols), ")")


if __name__ == "__main__":
    main()

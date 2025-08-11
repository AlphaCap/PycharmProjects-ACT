#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed M/E Ratio Calculator Script
This version handles Unicode encoding issues properly.
"""
import sys


def main():
    # Configure encoding
    sys.stdout.reconfigure(encoding='utf-8')

    # Main function to run the M/E ratio diagnostic
    print("M/E Ratio Calculator Diagnostic Tool")
    print("=" * 50)

def fix_me_calculation():
    filename = "nGS_Revised_Strategy.py"

    try:
        # Try UTF-8 first (most common for Python files)
        print(f"Reading {filename} with UTF-8 encoding...")
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        print(" Successfully read with UTF-8 encoding")

    except UnicodeDecodeError as e:
        print(f" UTF-8 failed: {e}")

        # Try UTF-8 with error handling
        try:
            print("Trying UTF-8 with error replacement...")
            with open(filename, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            print(" Successfully read with UTF-8 (with replacements)")

        except Exception as e:
            print(f" UTF-8 with replacements failed: {e}")

            # Try Latin-1 (can read any byte sequence)
            try:
                print("Trying Latin-1 encoding...")
                with open(filename, "r", encoding="latin-1") as f:
                    content = f.read()
                print(" Successfully read with Latin-1 encoding")

            except Exception as e:
                print(f" All encoding attempts failed: {e}")
                return False

    except FileNotFoundError:
        print(f" File not found: {filename}")
        print("Make sure you're running this script from the correct directory.")
        return False

    except Exception as e:
        print(f" Unexpected error reading file: {e}")
        return False

    print(f"\n File Analysis:")
    print(f"   File size: {len(content):,} characters")
    print(f"   Lines: {len(content.splitlines()):,}")

    # Look for M/E ratio calculation issues
    print(f"\n Analyzing M/E Ratio Calculations...")

    # Find M/E related functions and methods
    me_functions = []
    lines = content.splitlines()

    for i, line in enumerate(lines, 1):
        line_lower = line.lower()
        if (
            "me_ratio" in line_lower
            or "calculate_current_me" in line_lower
            or "calculate_historical_me" in line_lower
            or "dailymeratioCalculator" in line.lower()
        ):
            me_functions.append((i, line.strip()))

    if me_functions:
        print(f"\n Found {len(me_functions)} M/E ratio related lines:")
        for line_num, line_content in me_functions[:10]:  # Show first 10
            print(f"   Line {line_num:4d}: {line_content[:80]}...")
        if len(me_functions) > 10:
            print(f"   ... and {len(me_functions) - 10} more lines")

    # Look for potential calculation errors
    print(f"\n  Checking for Common M/E Calculation Issues:")

    issues_found = []

    # Check for division by cash instead of total equity
    if "total_position_value / self.cash" in content:
        issues_found.append(" Using cash instead of total equity in M/E calculation")

    # Check for missing position updates
    if "update_position" in content:
        update_count = content.count("update_position")
        print(f"   Found {update_count} position updates")

    # Check for missing M/E recording
    if "record_historical_me_ratio" in content:
        record_count = content.count("record_historical_me_ratio")
        print(f"   Found {record_count} M/E recordings")

    # Look for calculation formula
    if "portfolio_equity = " in content:
        print(" Found portfolio equity calculation")
    else:
        issues_found.append(" Portfolio equity calculation not found")

    if "me_ratio = (" in content:
        print(" Found M/E ratio calculation")
    else:
        issues_found.append(" M/E ratio calculation not found")

    # Check for proper M/E rebalancing
    if "me_rebalancing_enabled" in content:
        print(" Found M/E rebalancing system")
    else:
        issues_found.append(" M/E rebalancing system not found")

    if issues_found:
        print(f"\n Issues Found:")
        for issue in issues_found:
            print(f"   {issue}")
    else:
        print(f"\n No obvious M/E calculation issues found")

    # Save a clean UTF-8 version
    try:
        clean_filename = f"{filename}.clean"
        with open(clean_filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\n Saved clean UTF-8 version as: {clean_filename}")

    except Exception as e:
        print(f" Could not save clean version: {e}")

    return True


def check_me_ratio_formula():
    """
    Check if the M/E ratio formula is mathematically correct.
    """
    print(f"\n M/E RATIO FORMULA VERIFICATION")
    print("=" * 50)

    print("Correct M/E Ratio Formula:")
    print("   M/E = (Total Open Position Value) / (Total Portfolio Equity) * 100")
    print("")
    print("Where:")
    print(
        "   Total Open Position Value = Sum of |shares * current_price| for all positions"
    )
    print("   Total Portfolio Equity = Cash + Unrealized P&L + Realized P&L")
    print("   OR")
    print("   Total Portfolio Equity = Initial Capital + Total P&L")
    print("")
    print("Common Errors:")
    print("    Using Cash instead of Portfolio Equity (denominator)")
    print("    Not updating position values with current prices")
    print("    Not including unrealized P&L in equity calculation")
    print("    Counting short positions as negative values")
    print("")
    print("Expected M/E Range:")
    print("   • 0-50%:   Conservative (low leverage)")
    print("   • 50-80%:  Moderate (target range)")
    print("   • 80-100%: Aggressive (high leverage)")
    print("   • >100%:   Dangerous (over-leveraged)")


def main():
    """
    Main function to run the M/E ratio diagnostic.
    """
    print("M/E Ratio Calculator Diagnostic Tool")
    print("=" * 50)

    # Fix the file reading issue first
    if fix_me_calculation():
        # Then check the formula
        check_me_ratio_formula()

        print(f"\n{'='*50}")
        print("Diagnostic completed. Check the analysis above for issues.")
        print("If you found encoding issues, run the Unicode fix tool first.")
    else:
        print(
            " Could not read the strategy file. Please check file path and encoding."
        )


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
Demonstration script showing the difference between production and non-production modes
in the enhanced backtesting system.
"""

import pandas as pd
from ngs_integrated_ai_system import NGSAIBacktestingSystem


def demonstrate_mode_differences():
    """Demonstrate the differences between production and non-production modes"""
    
    print("=" * 70)
    print("nGS BACKTESTING SYSTEM - PRODUCTION vs NON-PRODUCTION MODE DEMO")
    print("=" * 70)
    
    # Sample trade data
    sample_trades = [
        {"profit": 1000.0, "entry_price": 100.0, "shares": 50},
        {"profit": -500.0, "entry_price": 80.0, "shares": 30},
        {"profit": 750.0, "entry_price": 150.0, "shares": 20},
    ]
    
    print(f"\nSample Trade Data ({len(sample_trades)} trades):")
    for i, trade in enumerate(sample_trades, 1):
        print(f"  Trade {i}: Profit=${trade['profit']:.2f}, Price=${trade['entry_price']:.2f}, Shares={trade['shares']}")
    
    original_total_profit = sum(trade["profit"] for trade in sample_trades)
    print(f"\nOriginal Total Profit: ${original_total_profit:.2f}")
    
    # Test Production Mode (default)
    print(f"\n" + "="*50)
    print("PRODUCTION MODE (Default - for Track Record Development)")
    print("="*50)
    
    prod_backtester = NGSAIBacktestingSystem(production_mode=True)
    prod_adjusted = prod_backtester._apply_trading_costs(sample_trades)
    
    prod_total_profit = sum(trade["profit"] for trade in prod_adjusted)
    prod_total_commission = sum(trade["commission"] for trade in prod_adjusted)
    prod_total_slippage = sum(trade["slippage"] for trade in prod_adjusted)
    
    print(f"Adjusted Total Profit: ${prod_total_profit:.2f}")
    print(f"Total Commission: ${prod_total_commission:.2f}")
    print(f"Total Slippage: ${prod_total_slippage:.2f}")
    print(f"Net Impact: ${prod_total_profit - original_total_profit:.2f}")
    print("✅ No trading costs applied - ideal for track record development")
    
    # Test Non-Production Mode
    print(f"\n" + "="*50)
    print("NON-PRODUCTION MODE (for Legacy/Testing)")
    print("="*50)
    
    nonprod_backtester = NGSAIBacktestingSystem(production_mode=False)
    nonprod_adjusted = nonprod_backtester._apply_trading_costs(sample_trades)
    
    nonprod_total_profit = sum(trade["profit"] for trade in nonprod_adjusted)
    nonprod_total_commission = sum(trade["commission"] for trade in nonprod_adjusted)
    nonprod_total_slippage = sum(trade["slippage"] for trade in nonprod_adjusted)
    
    print(f"Adjusted Total Profit: ${nonprod_total_profit:.2f}")
    print(f"Total Commission: ${nonprod_total_commission:.2f}")
    print(f"Total Slippage: ${nonprod_total_slippage:.2f}")
    print(f"Net Impact: ${nonprod_total_profit - original_total_profit:.2f}")
    print("⚠️  Trading costs applied - reduces profits")
    
    # Summary
    print(f"\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    cost_difference = nonprod_total_profit - prod_total_profit
    print(f"Cost Difference: ${abs(cost_difference):.2f}")
    print(f"Production Mode Advantage: ${cost_difference:.2f} higher profit")
    
    print("\n📊 RECOMMENDATION FOR TRACK RECORD DEVELOPMENT:")
    print("   Use Production Mode (default) to bypass brokerage costs")
    print("   This provides clean performance metrics without external cost factors")
    
    print(f"\n" + "="*70)


if __name__ == "__main__":
    demonstrate_mode_differences()
# calculate_daily_me_ratio_fixed.py - Calculate and store daily M/E ratios (Fixed for actual trade format)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def calculate_daily_me_ratios(initial_value=100000):
    """
    Calculate daily M/E ratios from trade history and save as indicator data
    Fixed to handle trade history without exit_reason column
    """
    print("=== CALCULATING DAILY M/E RATIOS ===\n")

    # Check if trade history file exists
    trade_file = "data/trades/trade_history.csv"
    if not os.path.exists(trade_file):
        print(f"ERROR: Trade history file not found: {trade_file}")
        print("Please ensure the file exists with the correct format.")
        print("You can run test_me_ratio_calculation.py to create sample data.")
        return None

    # Load trades
    trades_df = pd.read_csv(trade_file)

    # Check if required columns exist
    required_columns = [
        "symbol",
        "type",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "shares",
        "profit",
    ]
    missing_columns = [col for col in required_columns if col not in trades_df.columns]

    if missing_columns:
        print(
            f"ERROR: Missing required columns in trade_history.csv: {missing_columns}"
        )
        print(f"Required columns: {required_columns}")
        print(f"Found columns: {list(trades_df.columns)}")
        return None

    try:
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
    except Exception as e:
        print(f"ERROR: Failed to parse dates in trade history: {e}")
        print("Please ensure dates are in YYYY-MM-DD format")
        return None

    if len(trades_df) == 0:
        print("ERROR: No trades found in trade history file")
        return None

    # Get date range
    start_date = trades_df["entry_date"].min()
    end_date = trades_df["exit_date"].max()

    print(f"Analyzing period: {start_date.date()} to {end_date.date()}")
    print(f"Total trades in history: {len(trades_df)}")

    # Show trade summary
    print("\nTrade Summary:")
    for _, trade in trades_df.iterrows():
        print(
            f"  {trade['symbol']}: {trade['type']} {trade['shares']} shares, "
            f"{trade['entry_date'].date()} to {trade['exit_date'].date()}, "
            f"${trade['entry_price']:.2f} -> ${trade['exit_price']:.2f}, "
            f"P&L: ${trade['profit']:.2f}"
        )

    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Initialize tracking
    daily_me_data = []
    position_tracker = {}  # symbol -> {shares, entry_price, type}

    # Process each day
    for current_date in date_range:
        current_date_str = current_date.strftime("%Y-%m-%d")

        # Add new positions that start today
        new_entries = trades_df[trades_df["entry_date"].dt.date == current_date.date()]
        for _, trade in new_entries.iterrows():
            symbol = trade["symbol"]
            if symbol not in position_tracker:
                position_tracker[symbol] = {
                    "shares": 0,
                    "entry_prices": [],
                    "total_cost": 0,
                    "type": trade["type"],
                }

            # Add to position
            shares_to_add = (
                trade["shares"] if trade["type"] == "long" else -trade["shares"]
            )
            position_tracker[symbol]["shares"] += shares_to_add
            position_tracker[symbol]["entry_prices"].append(trade["entry_price"])
            position_tracker[symbol]["total_cost"] += (
                abs(shares_to_add) * trade["entry_price"]
            )

            print(
                f"  {current_date_str}: Opening {trade['type']} position in {symbol}: {abs(shares_to_add)} shares @ ${trade['entry_price']:.2f}"
            )

        # Remove positions that exit today
        exits = trades_df[trades_df["exit_date"].dt.date == current_date.date()]
        for _, trade in exits.iterrows():
            symbol = trade["symbol"]
            if symbol in position_tracker:
                shares_to_remove = (
                    trade["shares"] if trade["type"] == "long" else -trade["shares"]
                )
                position_tracker[symbol]["shares"] -= shares_to_remove

                print(
                    f"  {current_date_str}: Closing {trade['type']} position in {symbol}: {abs(shares_to_remove)} shares @ ${trade['exit_price']:.2f}, P&L: ${trade['profit']:.2f}"
                )

                # Remove if position is closed
                if abs(position_tracker[symbol]["shares"]) < 0.01:
                    print(f"    {symbol} position fully closed")
                    del position_tracker[symbol]

        # Calculate current metrics
        total_long_value = 0
        total_short_value = 0
        long_positions = 0
        short_positions = 0

        for symbol, pos_data in position_tracker.items():
            if pos_data["shares"] > 0:
                # Long position
                avg_price = (
                    pos_data["total_cost"] / abs(pos_data["shares"])
                    if pos_data["shares"] != 0
                    else 0
                )
                position_value = abs(pos_data["shares"]) * avg_price
                total_long_value += position_value
                long_positions += 1
            elif pos_data["shares"] < 0:
                # Short position
                avg_price = (
                    pos_data["total_cost"] / abs(pos_data["shares"])
                    if pos_data["shares"] != 0
                    else 0
                )
                position_value = abs(pos_data["shares"]) * avg_price
                total_short_value += position_value
                short_positions += 1

        # Calculate portfolio equity (initial + realized profits to date)
        closed_to_date = trades_df[trades_df["exit_date"] <= current_date]
        cumulative_profit = (
            closed_to_date["profit"].sum() if not closed_to_date.empty else 0
        )
        portfolio_equity = initial_value + cumulative_profit

        # Calculate M/E ratio
        total_position_value = total_long_value + total_short_value
        total_account_value = (
            self.initial_portfolio_value + self.realized_pnl + total_unrealized_pnl
        )
        me_ratio = (
            (total_position_value / total_account_value * 100)
            if total_account_value > 0
            else 0.0
        )

        # Store daily data
        daily_me_data.append(
            {
                "Date": current_date,
                "Portfolio_Equity": portfolio_equity,
                "Long_Value": total_long_value,
                "Short_Value": total_short_value,
                "Total_Position_Value": total_position_value,
                "ME_Ratio": me_ratio,
                "Long_Positions": long_positions,
                "Short_Positions": short_positions,
                "Total_Positions": long_positions + short_positions,
                "Cumulative_Profit": cumulative_profit,
            }
        )

        # Debug output for key dates
        if len(new_entries) > 0 or len(exits) > 0 or current_date.day % 5 == 0:
            print(
                f"  {current_date_str}: Long=${total_long_value:,.0f}, Short=${total_short_value:,.0f}, "
                f"Equity=${portfolio_equity:,.0f}, M/E={me_ratio:.1f}%, "
                f"Positions: {long_positions}L/{short_positions}S"
            )

    # Create DataFrame
    me_df = pd.DataFrame(daily_me_data)

    # Save to CSV
    output_file = "data/me_ratio_history.csv"
    os.makedirs("data", exist_ok=True)
    me_df.to_csv(output_file, index=False)
    print(f"\nSaved M/E ratio history to {output_file}")

    # Calculate statistics
    print("\n=== M/E RATIO STATISTICS ===")
    if len(me_df) > 0:
        print(f"Average M/E Ratio: {me_df['ME_Ratio'].mean():.1f}%")
        print(f"Maximum M/E Ratio: {me_df['ME_Ratio'].max():.1f}%")
        print(f"Minimum M/E Ratio: {me_df['ME_Ratio'].min():.1f}%")
        print(f"Standard Deviation: {me_df['ME_Ratio'].std():.1f}%")
    else:
        print("No data available for statistics")
        return None

    # Position statistics
    print(f"\n=== POSITION STATISTICS ===")
    if len(me_df) > 0:
        print(f"Average Total Positions: {me_df['Total_Positions'].mean():.1f}")
        print(f"Maximum Positions: {me_df['Total_Positions'].max()}")
        print(f"Average Long Positions: {me_df['Long_Positions'].mean():.1f}")
        print(f"Average Short Positions: {me_df['Short_Positions'].mean():.1f}")
    else:
        print("No position data available")

    # Show sample data
    print("\n=== SAMPLE DATA (Key dates with position changes) ===")
    # Find dates with position changes
    position_change_dates = me_df[
        (me_df["Long_Positions"].diff() != 0)
        | (me_df["Short_Positions"].diff() != 0)
        | (me_df["ME_Ratio"].diff().abs() > 1)
    ].head(10)

    print(
        position_change_dates[
            [
                "Date",
                "ME_Ratio",
                "Long_Positions",
                "Short_Positions",
                "Portfolio_Equity",
                "Total_Position_Value",
            ]
        ].to_string(index=False)
    )

    # Save summary for quick access
    try:
        summary = {
            "average_me_ratio": (
                float(me_df["ME_Ratio"].mean()) if len(me_df) > 0 else 0.0
            ),
            "max_me_ratio": float(me_df["ME_Ratio"].max()) if len(me_df) > 0 else 0.0,
            "min_me_ratio": float(me_df["ME_Ratio"].min()) if len(me_df) > 0 else 0.0,
            "std_me_ratio": float(me_df["ME_Ratio"].std()) if len(me_df) > 0 else 0.0,
            "average_positions": (
                float(me_df["Total_Positions"].mean()) if len(me_df) > 0 else 0.0
            ),
            "calculation_date": datetime.now().isoformat(),
            "total_days": len(me_df),
            "total_trades": len(trades_df),
            "initial_value": initial_value,
            "final_equity": (
                float(me_df["Portfolio_Equity"].iloc[-1])
                if len(me_df) > 0
                else initial_value
            ),
            "total_profit": (
                float(me_df["Cumulative_Profit"].iloc[-1]) if len(me_df) > 0 else 0.0
            ),
        }

        import json

        with open("me_ratio_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved summary to me_ratio_summary.json")
    except Exception as e:
        print(f"Warning: Could not save summary file: {e}")

    # Create a chart
    if len(me_df) > 0:
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

            # M/E Ratio over time
            ax1.plot(
                me_df["Date"],
                me_df["ME_Ratio"],
                label="M/E Ratio",
                color="blue",
                linewidth=2,
            )
            ax1.axhline(
                y=me_df["ME_Ratio"].mean(),
                color="red",
                linestyle="--",
                label=f'Average ({me_df["ME_Ratio"].mean():.1f}%)',
            )
            ax1.set_ylabel("M/E Ratio (%)")
            ax1.set_title("Historical Margin-to-Equity Ratio")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Position values
            ax2.plot(
                me_df["Date"], me_df["Long_Value"], label="Long Value", color="green"
            )
            ax2.plot(
                me_df["Date"], me_df["Short_Value"], label="Short Value", color="red"
            )
            ax2.plot(
                me_df["Date"],
                me_df["Total_Position_Value"],
                label="Total Position Value",
                color="black",
                linestyle="--",
            )
            ax2.set_ylabel("Position Value ($)")
            ax2.set_title("Position Values Over Time")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Portfolio equity
            ax3.plot(
                me_df["Date"],
                me_df["Portfolio_Equity"],
                label="Portfolio Equity",
                color="purple",
            )
            ax3.axhline(
                y=initial_value,
                color="gray",
                linestyle=":",
                label=f"Initial (${initial_value:,})",
            )
            ax3.set_xlabel("Date")
            ax3.set_ylabel("Portfolio Equity ($)")
            ax3.set_title("Portfolio Equity Over Time")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("me_ratio_history.png", dpi=150, bbox_inches="tight")
            print("\nSaved chart to me_ratio_history.png")
            plt.close()

        except ImportError:
            print("\nMatplotlib not available - skipping chart generation")
        except Exception as e:
            print(f"\nWarning: Could not create chart: {e}")
    else:
        print("\nNo data available for chart creation")

    return me_df


if __name__ == "__main__":
    result = calculate_daily_me_ratios()
    if result is not None:
        print("\n M/E ratio calculation completed successfully!")
    else:
        print("\n✗ M/E ratio calculation failed. Check error messages above.")
        print("\n M/E ratio calculation failed. Check error messages above.")


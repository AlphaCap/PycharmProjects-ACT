"""
Sector Parameter Manager
Manages sector-specific optimized parameters for the nGS strategy.
Integrates with existing sector mapping system.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

# Import your existing sector functions
try:
    from data_manager import get_all_sectors, get_symbol_sector

    SECTOR_INTEGRATION = True
except ImportError:
    print("Warning: data_manager not found. Using fallback sector mapping.")
    SECTOR_INTEGRATION = False


class SectorParameterManager:
    """
    Manages sector-specific parameters for strategy optimization.

    Features:
    - Store optimized parameters by sector
    - Retrieve parameters for any symbol based on its sector
    - Version control for parameter updates
    - Fallback to default parameters
    """

    def __init__(self, config_dir: str = "optimization_framework/config"):
        self.config_dir = config_dir
        self.parameters_file = os.path.join(config_dir, "sector_parameters.json")
        self.history_file = os.path.join(config_dir, "parameter_history.json")

        # Ensure config directory exists
        os.makedirs(config_dir, exist_ok=True)

        # Sector ETF mappings for optimization
        self.sector_etfs = {
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

        # Default nGS parameters (from your existing strategy)
        self.default_parameters = {
            "Length": 25,
            "NumDevs": 2,
            "MinPrice": 10,
            "MaxPrice": 500,
            "AfStep": 0.05,
            "AfLimit": 0.21,
            "PositionSize": 5000,
            "me_target_min": 50.0,
            "me_target_max": 80.0,
            "min_positions_for_scaling_up": 5,
            "profit_target_pct": 1.05,  # 5% gap out
            "stop_loss_pct": 0.90,  # 10% stop loss
        }

        # Load existing parameters
        self.sector_parameters = self._load_parameters()

    def _load_parameters(self) -> Dict[str, Dict]:
        """Load sector parameters from JSON file"""
        if os.path.exists(self.parameters_file):
            try:
                with open(self.parameters_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading parameters: {e}")
                return {}
        return {}

    def _save_parameters(self):
        """Save sector parameters to JSON file"""
        try:
            with open(self.parameters_file, "w") as f:
                json.dump(self.sector_parameters, f, indent=2)
        except Exception as e:
            print(f"Error saving parameters: {e}")

    def _save_to_history(
        self,
        sector: str,
        old_params: Dict,
        new_params: Dict,
        optimization_results: Dict = None,
    ):
        """Save parameter changes to history for tracking"""
        try:
            history = []
            if os.path.exists(self.history_file):
                with open(self.history_file, "r") as f:
                    history = json.load(f)

            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "sector": sector,
                "old_parameters": old_params,
                "new_parameters": new_params,
                "optimization_results": optimization_results or {},
            }

            history.append(history_entry)

            # Keep only last 100 entries
            history = history[-100:]

            with open(self.history_file, "w") as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            print(f"Error saving to history: {e}")

    def update_sector_parameters(
        self, sector: str, new_parameters: Dict, optimization_results: Dict = None
    ):
        """
        Update parameters for a specific sector

        Args:
            sector: Sector name (e.g., 'Technology')
            new_parameters: Dict of parameter updates
            optimization_results: Optional results from optimization run
        """
        old_params = self.sector_parameters.get(sector, {}).copy()

        if sector not in self.sector_parameters:
            self.sector_parameters[sector] = self.default_parameters.copy()

        # Update with new parameters
        self.sector_parameters[sector].update(new_parameters)

        # Add metadata
        self.sector_parameters[sector]["last_updated"] = datetime.now().isoformat()
        self.sector_parameters[sector][
            "optimization_source"
        ] = f"{self.sector_etfs.get(sector, 'Unknown')} ETF"

        # Save to file and history
        self._save_parameters()
        self._save_to_history(sector, old_params, new_parameters, optimization_results)

        print(f" Updated {sector} parameters:")
        for key, value in new_parameters.items():
            print(f"   {key}: {value}")

    def get_parameters_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Get optimized parameters for a specific symbol based on its sector

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Dict of parameters to use for this symbol
        """
        try:
            # Get sector for symbol using your existing function
            if SECTOR_INTEGRATION:
                sector = get_symbol_sector(symbol)
            else:
                # Fallback sector mapping (simplified)
                sector = self._fallback_sector_mapping(symbol)

            # Return sector-specific parameters if available
            if sector in self.sector_parameters:
                params = self.sector_parameters[sector].copy()
                params["assigned_sector"] = sector
                params["optimization_status"] = "optimized"
                return params
            else:
                # Return default parameters with metadata
                params = self.default_parameters.copy()
                params["assigned_sector"] = sector
                params["optimization_status"] = "default"
                return params

        except Exception as e:
            print(f"Error getting parameters for {symbol}: {e}")
            # Return safe defaults
            params = self.default_parameters.copy()
            params["assigned_sector"] = "Unknown"
            params["optimization_status"] = "error"
            return params

    def _fallback_sector_mapping(self, symbol: str) -> str:
        """Simplified sector mapping if data_manager not available"""
        tech_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
        finance_symbols = ["JPM", "BAC", "WFC", "C", "GS", "MS"]

        if symbol in tech_symbols:
            return "Technology"
        elif symbol in finance_symbols:
            return "Financials"
        else:
            return "Unknown"

    def get_sector_etf(self, sector: str) -> Optional[str]:
        """Get the ETF symbol for a sector"""
        return self.sector_etfs.get(sector)

    def get_all_optimized_sectors(self) -> list:
        """Get list of sectors with optimized parameters"""
        return list(self.sector_parameters.keys())

    def get_optimization_summary(self) -> pd.DataFrame:
        """Get summary of all sector optimizations"""
        summary_data = []

        for sector, params in self.sector_parameters.items():
            summary_data.append(
                {
                    "Sector": sector,
                    "ETF": self.sector_etfs.get(sector, "N/A"),
                    "Last Updated": params.get("last_updated", "Never"),
                    "Position Size": params.get("PositionSize", "Default"),
                    "M/E Min": params.get("me_target_min", "Default"),
                    "M/E Max": params.get("me_target_max", "Default"),
                    "BB Length": params.get("Length", "Default"),
                    "Status": "Optimized",
                }
            )

        # Add unoptimized sectors
        for sector, etf in self.sector_etfs.items():
            if sector not in self.sector_parameters:
                summary_data.append(
                    {
                        "Sector": sector,
                        "ETF": etf,
                        "Last Updated": "Never",
                        "Position Size": "Default",
                        "M/E Min": "Default",
                        "M/E Max": "Default",
                        "BB Length": "Default",
                        "Status": "Not Optimized",
                    }
                )

        return pd.DataFrame(summary_data)

    def reset_sector_parameters(self, sector: str):
        """Reset a sector to default parameters"""
        if sector in self.sector_parameters:
            old_params = self.sector_parameters[sector].copy()
            del self.sector_parameters[sector]
            self._save_parameters()
            self._save_to_history(
                sector, old_params, {}, {"action": "reset_to_default"}
            )
            print(f" Reset {sector} to default parameters")
        else:
            print(f"ℹ  {sector} was already using default parameters")


# Example usage and testing
if __name__ == "__main__":
    # Initialize parameter manager
    manager = SectorParameterManager()

    # Test parameter retrieval
    print(" Testing parameter retrieval:")
    aapl_params = manager.get_parameters_for_symbol("AAPL")
    print(f"AAPL parameters: {aapl_params}")

    # Test parameter update
    print("\n Testing parameter update:")
    tech_optimized_params = {
        "PositionSize": 6000,
        "me_target_min": 45.0,
        "me_target_max": 85.0,
        "Length": 20,
    }

    manager.update_sector_parameters(
        "Technology",
        tech_optimized_params,
        {"roi_improvement": 2.3, "test_period": "2024-Q1"},
    )

    # Test updated retrieval
    aapl_params_updated = manager.get_parameters_for_symbol("AAPL")
    print(f"AAPL parameters after update: {aapl_params_updated}")

    # Show optimization summary
    print("\n Optimization summary:")
    summary = manager.get_optimization_summary()
    print(summary.to_string(index=False))

"""
Test suite for backtesting system improvements.
Validates production mode, edge cases, and enhanced logging.
"""

import sys
import os
import pandas as pd
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ngs_integrated_ai_system import NGSAIBacktestingSystem, BacktestResult
from ngs_ai_integration_manager import NGSAIIntegrationManager


class TestProductionModeConfiguration:
    """Test production mode configuration and cost bypassing"""

    def test_production_mode_enabled_by_default(self):
        """Test that production mode is enabled by default"""
        backtester = NGSAIBacktestingSystem()
        assert backtester.production_mode is True
        assert backtester.backtest_config["commission_per_trade"] == 0.0
        assert backtester.backtest_config["slippage_pct"] == 0.0

    def test_production_mode_can_be_disabled(self):
        """Test that production mode can be disabled for non-production use"""
        backtester = NGSAIBacktestingSystem(production_mode=False)
        assert backtester.production_mode is False
        assert backtester.backtest_config["commission_per_trade"] == 1.0
        assert backtester.backtest_config["slippage_pct"] == 0.05

    def test_integration_manager_production_mode(self):
        """Test that integration manager respects production mode"""
        manager = NGSAIIntegrationManager(production_mode=True)
        assert manager.production_mode is True
        assert manager.execution_config["commission_per_trade"] == 0.0
        assert manager.execution_config["slippage_pct"] == 0.0

        manager_non_prod = NGSAIIntegrationManager(production_mode=False)
        assert manager_non_prod.production_mode is False
        assert manager_non_prod.execution_config["commission_per_trade"] == 1.0
        assert manager_non_prod.execution_config["slippage_pct"] == 0.05


class TestTradingCostsBypass:
    """Test that trading costs are properly bypassed in production mode"""

    def test_apply_trading_costs_production_mode(self):
        """Test that _apply_trading_costs bypasses costs in production mode"""
        backtester = NGSAIBacktestingSystem(production_mode=True)
        
        # Sample trades
        trades = [
            {"profit": 100.0, "entry_price": 50.0, "shares": 10},
            {"profit": -50.0, "entry_price": 60.0, "shares": 5},
        ]
        
        adjusted_trades = backtester._apply_trading_costs(trades)
        
        # In production mode, profits should remain unchanged
        assert adjusted_trades[0]["profit"] == 100.0
        assert adjusted_trades[1]["profit"] == -50.0
        
        # Commission and slippage should be 0
        assert adjusted_trades[0]["commission"] == 0.0
        assert adjusted_trades[0]["slippage"] == 0.0
        assert adjusted_trades[1]["commission"] == 0.0
        assert adjusted_trades[1]["slippage"] == 0.0

    def test_apply_trading_costs_non_production_mode(self):
        """Test that _apply_trading_costs applies costs in non-production mode"""
        backtester = NGSAIBacktestingSystem(production_mode=False)
        
        # Sample trades
        trades = [
            {"profit": 100.0, "entry_price": 50.0, "shares": 10},
        ]
        
        adjusted_trades = backtester._apply_trading_costs(trades)
        
        # In non-production mode, costs should be applied
        original_profit = 100.0
        commission = 1.0
        slippage_cost = 50.0 * 0.0005 * 10  # entry_price * slippage_pct/100 * shares
        expected_profit = original_profit - commission - slippage_cost
        
        assert adjusted_trades[0]["profit"] == expected_profit
        assert adjusted_trades[0]["commission"] == commission
        assert adjusted_trades[0]["slippage"] == slippage_cost

    def test_apply_slippage_production_mode(self):
        """Test that slippage is bypassed in production mode for integration manager"""
        manager = NGSAIIntegrationManager(production_mode=True)
        
        price = 100.0
        buy_price = manager._apply_slippage(price, "buy")
        sell_price = manager._apply_slippage(price, "sell")
        
        # In production mode, prices should remain unchanged
        assert buy_price == price
        assert sell_price == price

    def test_apply_slippage_non_production_mode(self):
        """Test that slippage is applied in non-production mode for integration manager"""
        manager = NGSAIIntegrationManager(production_mode=False)
        
        price = 100.0
        slip_pct = 0.05 / 100  # 0.05%
        
        buy_price = manager._apply_slippage(price, "buy")
        sell_price = manager._apply_slippage(price, "sell")
        
        # In non-production mode, slippage should be applied
        assert buy_price == price * (1.0 + slip_pct)
        assert sell_price == price * (1.0 - slip_pct)


class TestEdgeCaseHandling:
    """Test handling of edge cases and invalid inputs"""

    def test_empty_trades_list(self):
        """Test handling of empty trades list"""
        backtester = NGSAIBacktestingSystem()
        
        empty_trades = []
        adjusted_trades = backtester._apply_trading_costs(empty_trades)
        
        assert adjusted_trades == []

    def test_invalid_trade_data(self):
        """Test handling of trades with invalid data"""
        backtester = NGSAIBacktestingSystem(production_mode=False)
        
        # Trades with missing or invalid fields
        invalid_trades = [
            {"profit": "invalid", "entry_price": 50.0, "shares": 10},
            {"profit": 100.0},  # missing entry_price
            {"profit": 100.0, "entry_price": 0, "shares": 10},  # zero entry_price
            {"profit": 100.0, "entry_price": -10, "shares": 10},  # negative entry_price
        ]
        
        # Should not raise an exception, but handle gracefully
        adjusted_trades = backtester._apply_trading_costs(invalid_trades)
        
        assert len(adjusted_trades) == len(invalid_trades)
        # All trades should have commission and slippage fields, even if zero
        for trade in adjusted_trades:
            assert "commission" in trade
            assert "slippage" in trade

    def test_comprehensive_metrics_with_empty_data(self):
        """Test comprehensive metrics calculation with empty or invalid data"""
        backtester = NGSAIBacktestingSystem()
        
        # Mock strategy
        mock_strategy = Mock()
        mock_strategy.strategy_id = "test_strategy"
        mock_strategy.objective_name = "test_objective"
        
        # Test with empty trades
        empty_result = backtester._calculate_comprehensive_metrics(
            mock_strategy, [], pd.Series([1000000])
        )
        
        assert empty_result.total_trades == 0
        assert empty_result.total_return_pct == 0.0
        assert empty_result.win_rate == 0.0

    def test_comprehensive_metrics_with_invalid_equity_curve(self):
        """Test comprehensive metrics with invalid equity curve"""
        backtester = NGSAIBacktestingSystem()
        
        # Mock strategy
        mock_strategy = Mock()
        mock_strategy.strategy_id = "test_strategy"
        mock_strategy.objective_name = "test_objective"
        
        # Mock objective manager to avoid errors
        with patch.object(backtester, 'objective_manager') as mock_obj_mgr:
            mock_objective = Mock()
            mock_objective.calculate_fitness.return_value = 10.0
            mock_obj_mgr.get_objective.return_value = mock_objective
            
            # Test with empty equity curve
            result = backtester._calculate_comprehensive_metrics(
                mock_strategy, 
                [{"profit": 100.0}], 
                pd.Series([])  # Empty series
            )
            
            assert result.volatility_pct == 0.0
            assert result.sharpe_ratio == 0.0

    def test_data_validation_in_backtest_ai_strategy(self):
        """Test data validation in backtest_ai_strategy method"""
        backtester = NGSAIBacktestingSystem()
        
        # Mock strategy
        mock_strategy = Mock()
        mock_strategy.strategy_id = "test_strategy"
        mock_strategy.objective_name = "test_objective"
        
        # Test with empty data
        with pytest.raises(ValueError, match="Data dictionary cannot be empty"):
            backtester.backtest_ai_strategy(mock_strategy, {})
        
        # Test with None strategy
        with pytest.raises(ValueError, match="Strategy cannot be None"):
            backtester.backtest_ai_strategy(None, {"AAPL": pd.DataFrame()})


class TestEnhancedLogging:
    """Test enhanced logging functionality"""

    @patch('ngs_integrated_ai_system.logger')
    def test_initialization_logging(self, mock_logger):
        """Test that initialization logs appropriate messages"""
        backtester = NGSAIBacktestingSystem(production_mode=True)
        
        # Verify production mode logging
        mock_logger.info.assert_any_call("nGS AI Backtesting System initialized")
        mock_logger.info.assert_any_call("Production Mode: True")
        mock_logger.info.assert_any_call("Trading costs (commission/slippage) bypassed for production mode")

    @patch('ngs_integrated_ai_system.logger')
    def test_backtest_strategy_logging(self, mock_logger):
        """Test logging during strategy backtesting"""
        backtester = NGSAIBacktestingSystem()
        
        # Mock strategy and required methods
        mock_strategy = Mock()
        mock_strategy.strategy_id = "test_strategy"
        mock_strategy.objective_name = "test_objective"
        mock_strategy.execute_on_data.return_value = {
            "trades": [{"profit": 100.0}],
            "equity_curve": pd.Series([1000000, 1000100])
        }
        
        # Mock objective manager
        with patch.object(backtester, 'objective_manager') as mock_obj_mgr:
            mock_objective = Mock()
            mock_objective.calculate_fitness.return_value = 10.0
            mock_obj_mgr.get_objective.return_value = mock_objective
            
            # Create sample data
            sample_data = {"AAPL": pd.DataFrame({"Close": [100, 101], "Date": ["2024-01-01", "2024-01-02"]})}
            
            try:
                backtester.backtest_ai_strategy(mock_strategy, sample_data)
            except Exception:
                pass  # We're testing logging, not full execution
            
            # Verify key logging calls
            mock_logger.info.assert_any_call("Starting backtest for AI strategy: test_objective")
            mock_logger.debug.assert_any_call("Executing strategy on historical data")


class TestBackwardCompatibility:
    """Test that changes maintain backward compatibility"""

    def test_default_production_mode_compatibility(self):
        """Test that default behavior uses production mode"""
        # Default initialization should use production mode
        backtester = NGSAIBacktestingSystem()
        assert backtester.production_mode is True

    def test_existing_api_compatibility(self):
        """Test that existing API methods still work"""
        backtester = NGSAIBacktestingSystem()
        
        # Check that all expected methods exist
        assert hasattr(backtester, 'backtest_ai_strategy')
        assert hasattr(backtester, 'backtest_original_ngs')
        assert hasattr(backtester, '_apply_trading_costs')
        assert hasattr(backtester, '_calculate_comprehensive_metrics')

    def test_backtest_config_structure(self):
        """Test that backtest config maintains expected structure"""
        backtester = NGSAIBacktestingSystem()
        
        required_keys = [
            "commission_per_trade", "slippage_pct", "min_trade_spacing",
            "max_concurrent_positions", "benchmark_symbol", "risk_free_rate",
            "lookback_periods", "production_mode"
        ]
        
        for key in required_keys:
            assert key in backtester.backtest_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
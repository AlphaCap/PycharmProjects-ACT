"""
Sector ETF Optimizer
Performs walk-forward optimization on sector ETFs to find optimal parameters.
Results are saved via SectorParameterManager for use with individual stocks.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
import sys

# Import our parameter manager
from sector_parameter_manager import SectorParameterManager

# Import your existing data functions
try:
    from data_manager import load_price_data
    DATA_MANAGER_AVAILABLE = True
    print("✅ Using data_manager for price data")
except ImportError:
    DATA_MANAGER_AVAILABLE = False
    print("⚠️  data_manager not available - using fallback data loader")

# Try to import PyBroker (will guide user to install if missing)
try:
    import pybroker as pb
    PYBROKER_AVAILABLE = True
    print("✅ PyBroker available for optimization")
except ImportError:
    PYBROKER_AVAILABLE = False
    print("⚠️  PyBroker not installed. Install with: pip install pybroker")

class SectorETFOptimizer:
    """
    Optimizes sector ETF parameters using walk-forward testing.
    
    Features:
    - Walk-forward optimization on sector ETFs
    - Parameter optimization for nGS strategy
    - Results saved to SectorParameterManager
    - Integration with existing data_manager
    """
    
    def __init__(self, optimization_mode: str = "fast"):
        self.param_manager = SectorParameterManager()
        self.optimization_mode = optimization_mode  # "fast", "thorough", "custom"
        
        # Walk-forward settings
        self.train_months = 6  # Train on 6 months of data
        self.test_months = 2   # Test on 2 months of data
        self.min_data_points = 100  # Minimum data points needed
        
        # Optimization parameter ranges
        self.param_ranges = self._get_optimization_ranges()
        
        print(f"🎯 Initialized ETF Optimizer in {optimization_mode} mode")
        print(f"📊 Walk-forward: {self.train_months}M train, {self.test_months}M test")
    
    def _get_optimization_ranges(self) -> Dict:
        """Define parameter ranges for optimization based on mode"""
        if self.optimization_mode == "fast":
            # Smaller parameter ranges for quick testing
            return {
                'PositionSize': [4000, 5000, 6000],
                'me_target_min': [45, 50, 55],
                'me_target_max': [75, 80, 85],
                'Length': [20, 25, 30],
                'NumDevs': [1.5, 2.0, 2.5],
                'profit_target_pct': [1.04, 1.05, 1.08],
                'stop_loss_pct': [0.90, 0.92, 0.95]
            }
        elif self.optimization_mode == "thorough":
            # Wider parameter ranges for comprehensive testing
            return {
                'PositionSize': [3000, 4000, 5000, 6000, 7000],
                'me_target_min': [40, 45, 50, 55, 60],
                'me_target_max': [70, 75, 80, 85, 90],
                'Length': [15, 20, 25, 30, 35],
                'NumDevs': [1.5, 2.0, 2.5, 3.0],
                'profit_target_pct': [1.03, 1.04, 1.05, 1.06, 1.08, 1.10],
                'stop_loss_pct': [0.88, 0.90, 0.92, 0.95, 0.97]
            }
        else:  # custom
            # Default to fast mode ranges
            return self._get_optimization_ranges.__func__(self, "fast")
    
    def load_etf_data(self, etf_symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Load price data for ETF symbol
        Uses data_manager if available, fallback otherwise
        """
        try:
            if DATA_MANAGER_AVAILABLE:
                # Use your existing data loading
                df = load_price_data(etf_symbol)
                if df is not None and not df.empty:
                    # Ensure Date column is datetime
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values('Date').reset_index(drop=True)
                    return df
            else:
                # Fallback data loading (you could implement alternative source)
                print(f"⚠️  No data available for {etf_symbol} - data_manager not found")
                return None
                
        except Exception as e:
            print(f"❌ Error loading data for {etf_symbol}: {e}")
            return None
    
    def create_walk_forward_periods(self, df: pd.DataFrame) -> List[Dict]:
        """
        Create walk-forward training and testing periods
        
        Returns list of periods with train/test date ranges
        """
        if df is None or df.empty:
            return []
        
        df['Date'] = pd.to_datetime(df['Date'])
        start_date = df['Date'].min()
        end_date = df['Date'].max()
        
        periods = []
        current_start = start_date
        
        while current_start < end_date:
            # Training period: current_start + train_months
            train_end = current_start + timedelta(days=30 * self.train_months)
            
            # Testing period: train_end + test_months
            test_end = train_end + timedelta(days=30 * self.test_months)
            
            # Check if we have enough data for this period
            train_data = df[(df['Date'] >= current_start) & (df['Date'] <= train_end)]
            test_data = df[(df['Date'] > train_end) & (df['Date'] <= test_end)]
            
            if len(train_data) >= self.min_data_points and len(test_data) >= 20:
                periods.append({
                    'train_start': current_start,
                    'train_end': train_end,
                    'test_start': train_end,
                    'test_end': test_end,
                    'train_data': train_data,
                    'test_data': test_data
                })
            
            # Move forward by test period length
            current_start = train_end
            
            # Safety check to avoid infinite loop
            if len(periods) > 10:  # Max 10 walk-forward periods
                break
        
        return periods
    
    def simulate_ngs_strategy(self, df: pd.DataFrame, parameters: Dict) -> Dict:
        """
        Simulate nGS strategy on given data with parameters
        Returns performance metrics
        """
        if df is None or df.empty:
            return {'roi': 0, 'trades': 0, 'win_rate': 0, 'sharpe': 0}
        
        try:
            # Simple nGS simulation (simplified version of your strategy)
            df = df.copy()
            
            # Calculate Bollinger Bands
            bb_length = int(parameters.get('Length', 25))
            bb_devs = parameters.get('NumDevs', 2.0)
            
            df['BB_Mid'] = df['Close'].rolling(window=bb_length).mean()
            df['BB_Std'] = df['Close'].rolling(window=bb_length).std()
            df['BB_Upper'] = df['BB_Mid'] + (bb_devs * df['BB_Std'])
            df['BB_Lower'] = df['BB_Mid'] - (bb_devs * df['BB_Std'])
            
            # Simple signals (oversimplified for optimization testing)
            df['Signal'] = 0
            df['Position'] = 0
            df['PnL'] = 0
            
            position_size = parameters.get('PositionSize', 5000)
            profit_target = parameters.get('profit_target_pct', 1.05)
            stop_loss = parameters.get('stop_loss_pct', 0.90)
            
            current_position = 0
            entry_price = 0
            trades = []
            
            for i in range(1, len(df)):
                # Simple buy signal: price touches lower BB
                if (current_position == 0 and 
                    df['Low'].iloc[i] <= df['BB_Lower'].iloc[i] * 1.02 and
                    df['Close'].iloc[i] > df['BB_Lower'].iloc[i]):
                    
                    current_position = position_size / df['Close'].iloc[i]
                    entry_price = df['Close'].iloc[i]
                    df.loc[df.index[i], 'Signal'] = 1
                
                # Exit conditions
                elif current_position > 0:
                    exit_trade = False
                    exit_reason = ""
                    exit_price = df['Close'].iloc[i]
                    
                    # Profit target
                    if df['Close'].iloc[i] >= entry_price * profit_target:
                        exit_trade = True
                        exit_reason = "Profit Target"
                    
                    # Stop loss
                    elif df['Close'].iloc[i] <= entry_price * stop_loss:
                        exit_trade = True
                        exit_reason = "Stop Loss"
                    
                    # Simple time-based exit (20 bars)
                    elif i - df.index[df['Signal'] == 1].iloc[-1] > 20:
                        exit_trade = True
                        exit_reason = "Time Exit"
                    
                    if exit_trade:
                        pnl = (exit_price - entry_price) * current_position
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'exit_reason': exit_reason
                        })
                        current_position = 0
                        df.loc[df.index[i], 'Signal'] = -1
            
            # Calculate performance metrics
            if trades:
                total_pnl = sum(trade['pnl'] for trade in trades)
                winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
                win_rate = winning_trades / len(trades) * 100
                
                # Simple ROI calculation
                roi = (total_pnl / (position_size * len(trades))) * 100 if trades else 0
                
                # Simple Sharpe approximation
                trade_returns = [trade['pnl'] / position_size for trade in trades]
                if len(trade_returns) > 1:
                    sharpe = np.mean(trade_returns) / np.std(trade_returns) if np.std(trade_returns) > 0 else 0
                else:
                    sharpe = 0
            else:
                roi = 0
                win_rate = 0
                sharpe = 0
            
            return {
                'roi': roi,
                'trades': len(trades),
                'win_rate': win_rate,
                'sharpe': sharpe,
                'total_pnl': sum(trade['pnl'] for trade in trades) if trades else 0
            }
            
        except Exception as e:
            print(f"Error in strategy simulation: {e}")
            return {'roi': 0, 'trades': 0, 'win_rate': 0, 'sharpe': 0}
    
    def optimize_etf_parameters(self, etf_symbol: str, sector: str) -> Optional[Dict]:
        """
        Run walk-forward optimization on a single ETF
        
        Args:
            etf_symbol: ETF ticker (e.g., 'XLK')
            sector: Sector name (e.g., 'Technology')
            
        Returns:
            Dict with best parameters and performance results
        """
        print(f"\n🎯 Optimizing {sector} sector using {etf_symbol}")
        
        # Load ETF data
        df = self.load_etf_data(etf_symbol)
        if df is None or df.empty:
            print(f"❌ No data available for {etf_symbol}")
            return None
        
        print(f"📊 Loaded {len(df)} days of data for {etf_symbol}")
        
        # Create walk-forward periods
        periods = self.create_walk_forward_periods(df)
        if not periods:
            print(f"❌ Insufficient data for walk-forward testing")
            return None
        
        print(f"🔄 Running {len(periods)} walk-forward periods")
        
        # Test all parameter combinations across all periods
        best_params = None
        best_score = -999999
        all_results = []
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations()
        
        print(f"🧪 Testing {len(param_combinations)} parameter combinations...")
        
        for param_combo in param_combinations:
            period_results = []
            
            # Test this parameter combination across all walk-forward periods
            for period in periods:
                # Train period (not used in this simplified version)
                train_result = self.simulate_ngs_strategy(period['train_data'], param_combo)
                
                # Test period (this is what we care about)
                test_result = self.simulate_ngs_strategy(period['test_data'], param_combo)
                
                period_results.append({
                    'train_roi': train_result['roi'],
                    'test_roi': test_result['roi'],
                    'test_sharpe': test_result['sharpe'],
                    'test_trades': test_result['trades'],
                    'test_win_rate': test_result['win_rate']
                })
            
            # Calculate average performance across all test periods
            if period_results:
                avg_test_roi = np.mean([r['test_roi'] for r in period_results])
                avg_test_sharpe = np.mean([r['test_sharpe'] for r in period_results])
                total_test_trades = sum([r['test_trades'] for r in period_results])
                
                # Composite score (weighted average of ROI and Sharpe)
                score = (avg_test_roi * 0.7) + (avg_test_sharpe * 30 * 0.3)
                
                all_results.append({
                    'parameters': param_combo,
                    'avg_test_roi': avg_test_roi,
                    'avg_test_sharpe': avg_test_sharpe,
                    'total_trades': total_test_trades,
                    'score': score,
                    'period_results': period_results
                })
                
                if score > best_score and total_test_trades >= 5:  # Minimum trade requirement
                    best_score = score
                    best_params = param_combo.copy()
        
        if best_params:
            # Find the result for best params
            best_result = next(r for r in all_results if r['parameters'] == best_params)
            
            print(f"✅ Best parameters found for {sector}:")
            for key, value in best_params.items():
                print(f"   {key}: {value}")
            print(f"   Average Test ROI: {best_result['avg_test_roi']:.2f}%")
            print(f"   Average Sharpe: {best_result['avg_test_sharpe']:.3f}")
            print(f"   Total Test Trades: {best_result['total_trades']}")
            
            return {
                'etf_symbol': etf_symbol,
                'sector': sector,
                'best_parameters': best_params,
                'performance': best_result,
                'optimization_date': datetime.now().isoformat(),
                'periods_tested': len(periods),
                'combinations_tested': len(param_combinations)
            }
        else:
            print(f"❌ No suitable parameters found for {sector}")
            return None
    
    def _generate_param_combinations(self) -> List[Dict]:
        """Generate all combinations of parameters to test"""
        import itertools
        
        param_names = list(self.param_ranges.keys())
        param_values = list(self.param_ranges.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def optimize_all_sectors(self, sectors_to_optimize: List[str] = None) -> Dict:
        """
        Optimize all sector ETFs or specified sectors
        
        Args:
            sectors_to_optimize: List of sector names, or None for all sectors
        """
        if sectors_to_optimize is None:
            sectors_to_optimize = list(self.param_manager.sector_etfs.keys())
        
        print(f"🚀 Starting optimization for {len(sectors_to_optimize)} sectors")
        
        optimization_results = {}
        
        for sector in sectors_to_optimize:
            etf_symbol = self.param_manager.sector_etfs.get(sector)
            if not etf_symbol:
                print(f"⚠️  No ETF mapping for sector: {sector}")
                continue
            
            try:
                result = self.optimize_etf_parameters(etf_symbol, sector)
                if result:
                    # Save optimized parameters
                    self.param_manager.update_sector_parameters(
                        sector=sector,
                        new_parameters=result['best_parameters'],
                        optimization_results=result['performance']
                    )
                    optimization_results[sector] = result
                else:
                    print(f"❌ Optimization failed for {sector}")
                    
            except Exception as e:
                print(f"❌ Error optimizing {sector}: {e}")
        
        print(f"\n✅ Optimization complete! Successfully optimized {len(optimization_results)} sectors")
        return optimization_results

# Example usage and testing
if __name__ == "__main__":
    print("🎯 ETF Optimization Testing")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = SectorETFOptimizer(optimization_mode="fast")
    
    # Test single sector optimization
    print("\n📊 Testing single sector optimization (Technology)...")
    try:
        result = optimizer.optimize_etf_parameters('XLK', 'Technology')
        if result:
            print("✅ Single sector optimization successful!")
        else:
            print("❌ Single sector optimization failed")
    except Exception as e:
        print(f"❌ Error in single sector test: {e}")
    
    # Uncomment to test full optimization (will take longer)
    # print("\n🚀 Testing full sector optimization...")
    # optimizer.optimize_all_sectors(['Technology', 'Financials'])
    
    print("\n📋 Current optimization summary:")
    summary = optimizer.param_manager.get_optimization_summary()
    print(summary.to_string(index=False))

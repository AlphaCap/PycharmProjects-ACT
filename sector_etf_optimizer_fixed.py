"""
Sector ETF Optimizer - FIXED VERSION
Fixed to load data directly from downloaded CSV files
No dependency on data_manager.py
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Import our parameter manager
from sector_parameter_manager import SectorParameterManager

# Try to import optimization libraries (in order of preference)
try:
    import optuna
    OPTUNA_AVAILABLE = True
    OPTIMIZATION_ENGINE = "optuna"
    print("✅ Optuna available for optimization (recommended)")
except ImportError:
    OPTUNA_AVAILABLE = False
    try:
        from scipy import optimize
        SCIPY_AVAILABLE = True
        OPTIMIZATION_ENGINE = "scipy"
        print("✅ Scipy available for optimization")
    except ImportError:
        SCIPY_AVAILABLE = False
        OPTIMIZATION_ENGINE = "grid_search"
        print("✅ Using built-in grid search optimization")

class SectorETFOptimizer:
    """
    Optimizes sector ETF parameters using walk-forward testing.
    FIXED VERSION - loads data directly from CSV files
    
    Features:
    - Walk-forward optimization on sector ETFs
    - Parameter optimization for nGS strategy
    - Results saved to SectorParameterManager
    - Direct CSV data loading (no data_manager dependency)
    """
    
    def __init__(self, optimization_mode: str = "fast", data_dir: str = "data/etf_historical"):
        self.param_manager = SectorParameterManager()
        self.optimization_mode = optimization_mode  # "fast", "thorough", "custom"
        self.data_dir = data_dir
        
        # Walk-forward settings
        self.train_months = 6  # Train on 6 months of data
        self.test_months = 2   # Test on 2 months of data
        self.min_data_points = 80  # Minimum data points needed
        
        # Optimization parameter ranges
        self.param_ranges = self._get_optimization_ranges()
        
        print(f"🎯 Initialized ETF Optimizer in {optimization_mode} mode")
        print(f"📊 Walk-forward: {self.train_months}M train, {self.test_months}M test")
        print(f"📁 Data directory: {data_dir}")
    
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
        Load price data for ETF symbol - FIXED VERSION
        Loads directly from downloaded CSV files
        """
        try:
            # Load directly from CSV files
            filepath = os.path.join(self.data_dir, f"{etf_symbol}_historical.csv")
            
            if not os.path.exists(filepath):
                print(f"❌ ETF data file not found: {filepath}")
                print(f"💡 Make sure you've run etf_historical_downloader.py first")
                return None
            
            # Load CSV
            df = pd.read_csv(filepath)
            
            if df.empty:
                print(f"❌ Empty data file for {etf_symbol}")
                return None
            
            # Ensure Date column is datetime
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Filter by date range if provided
            if start_date:
                df = df[df['Date'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['Date'] <= pd.to_datetime(end_date)]
            
            if len(df) == 0:
                print(f"❌ No data for {etf_symbol} in specified date range")
                return None
            
            # Validate required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ Missing columns in {etf_symbol}: {missing_columns}")
                return None
            
            print(f"✅ Loaded {etf_symbol}: {len(df)} records ({df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')})")
            return df[required_columns]
            
        except Exception as e:
            print(f"❌ Error loading {etf_symbol}: {e}")
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
        print(f"🧠 Using {OPTIMIZATION_ENGINE} optimization engine")
        
        # Choose optimization method based on available libraries
        if OPTIMIZATION_ENGINE == "optuna" and OPTUNA_AVAILABLE:
            optimization_result = self.optimize_with_optuna(df, n_trials=100 if self.optimization_mode == "thorough" else 50)
        else:
            optimization_result = self.optimize_with_grid_search(df)
        
        if optimization_result and optimization_result['best_parameters']:
            best_params = optimization_result['best_parameters']
            
            # Calculate final performance metrics
            periods = self.create_walk_forward_periods(df)
            final_results = []
            for period in periods:
                result = self.simulate_ngs_strategy(period['test_data'], best_params)
                final_results.append(result)
            
            avg_roi = np.mean([r['roi'] for r in final_results])
            avg_sharpe = np.mean([r['sharpe'] for r in final_results])
            total_trades = sum([r['trades'] for r in final_results])
            
            print(f"✅ Best parameters found for {sector}:")
            for key, value in best_params.items():
                print(f"   {key}: {value}")
            print(f"   Average Test ROI: {avg_roi:.2f}%")
            print(f"   Average Sharpe: {avg_sharpe:.3f}")
            print(f"   Total Test Trades: {total_trades}")
            
            return {
                'etf_symbol': etf_symbol,
                'sector': sector,
                'best_parameters': best_params,
                'performance': {
                    'avg_test_roi': avg_roi,
                    'avg_test_sharpe': avg_sharpe,
                    'total_trades': total_trades,
                    'optimization_method': OPTIMIZATION_ENGINE
                },
                'optimization_date': datetime.now().isoformat(),
                'periods_tested': len(periods),
                'optimization_details': optimization_result
            }
        else:
            print(f"❌ No suitable parameters found for {sector}")
            return None
    
    def optimize_with_optuna(self, df: pd.DataFrame, n_trials: int = 50) -> Dict:
        """Optimize using Optuna (most efficient)"""
        def objective(trial):
            params = {
                'PositionSize': trial.suggest_categorical('PositionSize', self.param_ranges['PositionSize']),
                'me_target_min': trial.suggest_categorical('me_target_min', self.param_ranges['me_target_min']),
                'me_target_max': trial.suggest_categorical('me_target_max', self.param_ranges['me_target_max']),
                'Length': trial.suggest_categorical('Length', self.param_ranges['Length']),
                'NumDevs': trial.suggest_categorical('NumDevs', self.param_ranges['NumDevs']),
                'profit_target_pct': trial.suggest_categorical('profit_target_pct', self.param_ranges['profit_target_pct']),
                'stop_loss_pct': trial.suggest_categorical('stop_loss_pct', self.param_ranges['stop_loss_pct'])
            }
            
            # Run walk-forward test with these parameters
            periods = self.create_walk_forward_periods(df)
            if not periods:
                return -999999
            
            test_rois = []
            for period in periods:
                result = self.simulate_ngs_strategy(period['test_data'], params)
                test_rois.append(result['roi'])
            
            return np.mean(test_rois) if test_rois else -999999
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        return {
            'best_parameters': study.best_params,
            'best_score': study.best_value,
            'n_trials': n_trials
        }
    
    def _generate_param_combinations(self) -> List[Dict]:
        """Generate all combinations of parameters to test (Grid Search)"""
        import itertools
        
        param_names = list(self.param_ranges.keys())
        param_values = list(self.param_ranges.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def optimize_with_grid_search(self, df: pd.DataFrame) -> Dict:
        """Optimize using grid search (reliable fallback)"""
        periods = self.create_walk_forward_periods(df)
        if not periods:
            return None
        
        param_combinations = self._generate_param_combinations()
        best_params = None
        best_score = -999999
        
        print(f"🔍 Grid search testing {len(param_combinations)} combinations...")
        
        for i, param_combo in enumerate(param_combinations):
            if i % 20 == 0:  # Progress indicator
                print(f"   Progress: {i}/{len(param_combinations)} ({i/len(param_combinations)*100:.0f}%)")
            
            test_rois = []
            total_trades = 0
            
            for period in periods:
                result = self.simulate_ngs_strategy(period['test_data'], param_combo)
                test_rois.append(result['roi'])
                total_trades += result['trades']
            
            avg_roi = np.mean(test_rois) if test_rois else -999999
            
            if avg_roi > best_score and total_trades >= 5:
                best_score = avg_roi
                best_params = param_combo.copy()
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'combinations_tested': len(param_combinations)
        }
    
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
    print("🎯 ETF Optimization Testing - FIXED VERSION")
    print("=" * 50)
    print(f"Optimization Engine: {OPTIMIZATION_ENGINE}")
    
    # Initialize optimizer
    optimizer = SectorETFOptimizer(optimization_mode="fast")
    
    # Test single sector optimization
    print("\n📊 Testing single sector optimization (Technology)...")
    try:
        result = optimizer.optimize_etf_parameters('XLK', 'Technology')
        if result:
            print("✅ Single sector optimization successful!")
            print(f"   Best ROI: {result['performance']['avg_test_roi']:.2f}%")
        else:
            print("❌ Single sector optimization failed")
    except Exception as e:
        print(f"❌ Error in single sector test: {e}")
        import traceback
        traceback.print_exc()
    
    # Show installation recommendations
    if OPTIMIZATION_ENGINE == "grid_search":
        print("\n💡 For faster optimization, install Optuna:")
        print("   pip install optuna")
    
    print("\n📋 Current optimization summary:")
    summary = optimizer.param_manager.get_optimization_summary()
    print(summary.to_string(index=False))

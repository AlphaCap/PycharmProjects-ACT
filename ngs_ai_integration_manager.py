import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
import json
import os
<<<<<<< HEAD
from shared_utils import load_polygon_data
=======
from data_utils import load_polygon_data
>>>>>>> c108ef4 (Bypass pre-commit for now)
from comprehensive_indicator_library import ComprehensiveIndicatorLibrary
from performance_objectives import ObjectiveManager
from strategy_generator_ai import ObjectiveAwareStrategyGenerator, TradingStrategy
from data_manager import save_trades, save_positions, load_price_data
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from ngs_integrated_ai_system import NGSIndicatorLibrary, NGSAwareStrategyGenerator

logger = logging.getLogger(__name__)


class NGSAIIntegrationManager:
    def __init__(self, account_size: float = 1000000, data_dir: str = 'data'):
        self.account_size = account_size
        self.data_dir = data_dir
<<<<<<< HEAD
        
        from nGS_Revised_Strategy import NGSStrategy
        self.original_ngs = NGSStrategy(account_size=account_size, data_dir=data_dir)
        
=======

        from nGS_Revised_Strategy import NGSStrategy
        self.original_ngs = NGSStrategy(account_size=account_size, data_dir=data_dir)

>>>>>>> c108ef4 (Bypass pre-commit for now)
        # Initialize AI components
        self.ngs_indicator_lib = NGSIndicatorLibrary()
        self.objective_manager = ObjectiveManager()
        # Debugging statements
        print("ObjectiveManager initialized:", isinstance(self.objective_manager, ObjectiveManager))
        print("Available methods in ObjectiveManager:", dir(self.objective_manager))
        print("Primary objective (test):", self.objective_manager.get_primary_objective())
        self.ai_generator = NGSAwareStrategyGenerator(self.ngs_indicator_lib, self.objective_manager)
        logger.info("Initialized NGSIndicatorLibrary and NGSAwareStrategyGenerator from ngs_ai_components")
<<<<<<< HEAD
        
=======

>>>>>>> c108ef4 (Bypass pre-commit for now)
        self.active_strategies = {}
        self.strategy_performance = {}
        self.operating_mode = 'ai_only'
        self.integration_config = {
            'ai_allocation_pct': 100.0,
            'max_ai_strategies': 3,
            'rebalance_frequency': 'weekly',
            'performance_tracking': True,
            'risk_sync': False,
        }
        self.results_dir = os.path.join(self.data_dir, "integration_results")
        os.makedirs(self.results_dir, exist_ok=True)
<<<<<<< HEAD
        
        print("🎯 nGS AI Integration Manager initialized")
        print(f"   AI Generator:        Ready with YOUR parameters")
        print(f"   Operating Mode:      {self.operating_mode}")
        print(f"   Integration Config:  {self.integration_config['ai_allocation_pct']:.0f}% AI")
    
=======

        print("  nGS AI Integration Manager initialized")
        print(f"   AI Generator:        Ready with YOUR parameters")
        print(f"   Operating Mode:      {self.operating_mode}")
        print(f"   Integration Config:  {self.integration_config['ai_allocation_pct']:.0f}% AI")

>>>>>>> c108ef4 (Bypass pre-commit for now)
    def set_operating_mode(self, mode: str) -> None:
        """Set operating mode for the integration manager."""
        valid_modes = ['ai_only', 'comparison']
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}")
        self.operating_mode = mode
        logger.info(f"Operating mode set to: {mode}")
        print(f"🔄 Operating mode updated to: {mode}")
<<<<<<< HEAD
    
=======

>>>>>>> c108ef4 (Bypass pre-commit for now)
    def create_ai_strategy_set(self, data: Dict[str, pd.DataFrame], objective: str) -> Dict[str, TradingStrategy]:
        """Create a set of AI-generated strategies for given data and objective."""
        strategies = {}
        for symbol, df in data.items():
            try:
                strategy = self.ai_generator.generate_strategy_for_objective(objective)
                strategy.data = df
                strategies[symbol] = strategy
                logger.debug(f"Generated AI strategy for {symbol}")
            except Exception as e:
                logger.error(f"Failed to generate strategy for {symbol}: {str(e)}")
        return strategies
<<<<<<< HEAD
    
=======

>>>>>>> c108ef4 (Bypass pre-commit for now)
    def _run_original_strategy(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Run the original nGS strategy for comparison."""
        results = {}
        for symbol, df in data.items():
            try:
                self.original_ngs.process_symbol(df, symbol)
                trades = load_trades(symbol, self.data_dir)
                positions = load_positions(symbol, self.data_dir)
                results[symbol] = {
                    'trades': trades,
                    'positions': positions,
                    'performance': self.original_ngs.calculate_performance(symbol)
                }
                logger.debug(f"Processed original nGS strategy for {symbol}")
            except Exception as e:
                logger.error(f"Error processing original strategy for {symbol}: {str(e)}")
        return results
<<<<<<< HEAD
    
   # Line ~95: Add debugging inside `_run_ai_strategies_only`
def _run_ai_strategies_only(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """Run AI-generated strategies exclusively."""
    results = {}
    
=======

def _run_ai_strategies_only(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """Run AI-generated strategies exclusively."""
    results = {}

>>>>>>> c108ef4 (Bypass pre-commit for now)
    # Debugging before objective retrieval
    print("Debug: Checking ObjectiveManager instance...")
    print("ObjectiveManager type:", type(self.objective_manager))
    print("Available methods:", dir(self.objective_manager))
<<<<<<< HEAD
    
    # Objective retrieval
    objective = self.objective_manager.get_primary_objective()  # This is where the error occurs
    print("Primary objective retrieved:", objective)
        self.active_strategies = self.create_ai_strategy_set(data, objective)
        
        for symbol, strategy in self.active_strategies.items():
            try:
                trades = []
                positions = []
                for i in range(len(strategy.data)):
                    signal = strategy.generate_signal(strategy.data.iloc[i])
                    if signal == 1:
                        trades.append({
                            'symbol': symbol,
                            'timestamp': strategy.data.index[i],
                            'price': strategy.data['Close'].iloc[i],
                            'type': 'buy',
                            'shares': int(self.account_size / strategy.data['Close'].iloc[i] / len(data))
                        })
                    elif signal == -1:
                        trades.append({
                            'symbol': symbol,
                            'timestamp': strategy.data.index[i],
                            'price': strategy.data['Close'].iloc[i],
                            'type': 'sell',
                            'shares': int(self.account_size / strategy.data['Close'].iloc[i] / len(data))
                        })
                
                save_trades(symbol, trades, self.data_dir)
                positions = self._calculate_positions(trades, strategy.data)
                save_positions(symbol, positions, self.data_dir)
                
                results[symbol] = {
                    'trades': trades,
                    'positions': positions,
                    'performance': self._calculate_performance(trades, strategy.data)
                }
                logger.debug(f"Processed AI strategy for {symbol}")
            except Exception as e:
                logger.error(f"Error processing AI strategy for {symbol}: {str(e)}")
        
        return results
    
    def _calculate_positions(self, trades: List[Dict], data: pd.DataFrame) -> List[Dict]:
        """Calculate positions from trades."""
        positions = []
        current_position = 0
        entry_price = 0
        entry_time = None
        
        for trade in trades:
            if trade['type'] == 'buy':
                if current_position == 0:
                    current_position = trade['shares']
                    entry_price = trade['price']
                    entry_time = trade['timestamp']
            elif trade['type'] == 'sell' and current_position > 0:
                positions.append({
                    'symbol': trade['symbol'],
                    'entry_time': entry_time,
                    'exit_time': trade['timestamp'],
                    'entry_price': entry_price,
                    'exit_price': trade['price'],
                    'shares': current_position
                })
                current_position = 0
                entry_price = 0
                entry_time = None
        
        return positions
    
    def _calculate_performance(self, trades: List[Dict], data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for a set of trades."""
        returns = []
        for trade in trades:
            if trade['type'] == 'sell':
                entry_price = next((t['price'] for t in trades[::-1] if t['type'] == 'buy' and t['timestamp'] < trade['timestamp']), None)
                if entry_price:
                    returns.append((trade['price'] - entry_price) / entry_price)
        
        if not returns:
            return {'sharpe': 0.0, 'total_return': 0.0}
        
        returns = np.array(returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 1.0
        sharpe = mean_return / std_return * np.sqrt(252) if std_return != 0 else 0.0
        total_return = np.prod(1 + returns) - 1
        
        return {
            'sharpe': sharpe,
            'total_return': total_return
        }
    
    def run_integrated_strategy(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Run integrated AI and nGS strategy based on operating mode."""
        logger.info(f"Running integrated strategy in {self.operating_mode} mode")
        
        if self.operating_mode == 'ai_only':
            results = self._run_ai_strategies_only(data)
        else:  # comparison mode
            ai_results = self._run_ai_strategies_only(data)
            original_results = self._run_original_strategy(data)
            results = {
                'ai': ai_results,
                'original': original_results,
                'comparison': self._compare_strategies(ai_results, original_results)
            }
        
        self.strategy_performance.update(results)
        return results
    
    def _compare_strategies(self, ai_results: Dict, original_results: Dict) -> Dict[str, Any]:
        """Compare AI and original strategy performance."""
        comparison = {}
        for symbol in ai_results.keys():
            try:
                ai_perf = ai_results[symbol]['performance']
                orig_perf = original_results.get(symbol, {}).get('performance', {'sharpe': 0.0, 'total_return': 0.0})
                
                comparison[symbol] = {
                    'ai_sharpe': ai_perf['sharpe'],
                    'original_sharpe': orig_perf['sharpe'],
                    'ai_total_return': ai_perf['total_return'],
                    'original_total_return': orig_perf['total_return'],
                    'sharpe_diff': ai_perf['sharpe'] - orig_perf['sharpe'],
                    'return_diff': ai_perf['total_return'] - orig_perf['total_return']
                }
            except Exception as e:
                logger.error(f"Error comparing strategies for {symbol}: {str(e)}")
                comparison[symbol] = {}
        
        return comparison
    
    def rebalance_portfolio(self, data: Dict[str, pd.DataFrame]) -> None:
        """Rebalance portfolio based on AI strategy performance."""
        if self.integration_config['rebalance_frequency'] == 'weekly':
            current_time = datetime.now()
            if current_time.weekday() != 0:  # Not Monday
                logger.debug("Skipping rebalance: not a Monday")
                return
        
        performance = self.strategy_performance
        allocations = {}
        total_weight = 0
        
        for symbol, result in performance.items():
            if isinstance(result, dict) and 'performance' in result:
                weight = result['performance']['sharpe'] if result['performance']['sharpe'] > 0 else 0
                allocations[symbol] = weight
                total_weight += weight
        
        if total_weight == 0:
            logger.warning("No positive Sharpe ratios found for rebalancing")
            return
        
        for symbol in allocations:
            allocations[symbol] = (allocations[symbol] / total_weight) * self.integration_config['ai_allocation_pct']
            logger.debug(f"Rebalanced allocation for {symbol}: {allocations[symbol]:.2f}%")
        
        # Update active strategies based on new allocations
        objective = self.objective_manager.get_primary_objective()
        self.active_strategies = self.create_ai_strategy_set(data, objective)
        logger.info("Portfolio rebalanced based on performance")
    
    def save_integration_session(self, results: Dict[str, Any], filename: str = "integration_results.json") -> None:
        """Save integration results to a file."""
        try:
            output_path = os.path.join(self.results_dir, filename)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Saved integration results to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save integration results: {str(e)}")
    
    def plot_performance_comparison(self, results: Dict[str, Any]) -> None:
        """Plot performance comparison between AI and original strategies."""
        if 'comparison' not in results:
            logger.warning("No comparison data available for plotting")
            return
        
        plt.figure(figsize=(12, 6))
        symbols = list(results['comparison'].keys())
        ai_sharpes = [results['comparison'][s]['ai_sharpe'] for s in symbols]
        orig_sharpes = [results['comparison'][s]['original_sharpe'] for s in symbols]
        
        x = np.arange(len(symbols))
        width = 0.35
        
        plt.bar(x - width/2, ai_sharpes, width, label='AI Strategy')
        plt.bar(x + width/2, orig_sharpes, width, label='Original nGS')
        
        plt.xlabel('Symbols')
        plt.ylabel('Sharpe Ratio')
        plt.title('AI vs Original nGS Strategy Performance')
        plt.xticks(x, symbols, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(self.results_dir, 'performance_comparison.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved performance comparison plot to {plot_path}")
=======

    # Objective retrieval
    objective = self.objective_manager.get_primary_objective()  # This is where the error occurs
    print("Primary objective retrieved:", objective)
    self.active_strategies = self.create_ai_strategy_set(data, objective)

    for symbol, strategy in self.active_strategies.items():
        try:
            trades = []
            positions = []
            for i in range(len(strategy.data)):
                signal = strategy.generate_signal(strategy.data.iloc[i])
                if signal == 1:
                    trades.append({
                        'symbol': symbol,
                        'timestamp': strategy.data.index[i],
                        'price': strategy.data['Close'].iloc[i],
                        'type': 'buy',
                        'shares': int(self.account_size / strategy.data['Close'].iloc[i] / len(data))
                    })
                elif signal == -1:
                    trades.append({
                        'symbol': symbol,
                        'timestamp': strategy.data.index[i],
                        'price': strategy.data['Close'].iloc[i],
                        'type': 'sell',
                        'shares': int(self.account_size / strategy.data['Close'].iloc[i] / len(data))
                    })

            save_trades(symbol, trades, self.data_dir)
            positions = self._calculate_positions(trades, strategy.data)
            save_positions(symbol, positions, self.data_dir)

            results[symbol] = {
                'trades': trades,
                'positions': positions,
                'performance': self._calculate_performance(trades, strategy.data)
            }
            logger.debug(f"Processed AI strategy for {symbol}")
        except Exception as e:
            logger.error(f"Error processing AI strategy for {symbol}: {str(e)}")

    return results


def _calculate_positions(self, trades: List[Dict], data: pd.DataFrame) -> List[Dict]:
    """Calculate positions from trades."""
    positions = []
    current_position = 0
    entry_price = 0
    entry_time = None

    for trade in trades:
        if trade['type'] == 'buy':
            if current_position == 0:
                current_position = trade['shares']
                entry_price = trade['price']
                entry_time = trade['timestamp']
        elif trade['type'] == 'sell' and current_position > 0:
            positions.append({
                'symbol': trade['symbol'],
                'entry_time': entry_time,
                'exit_time': trade['timestamp'],
                'entry_price': entry_price,
                'exit_price': trade['price'],
                'shares': current_position
            })
            current_position = 0
            entry_price = 0
            entry_time = None

    return positions


def _calculate_performance(self, trades: List[Dict], data: pd.DataFrame) -> Dict[str, float]:
    """Calculate performance metrics for a set of trades."""
    returns = []
    for trade in trades:
        if trade['type'] == 'sell':
            entry_price = next(
                (t['price'] for t in trades[::-1] if t['type'] == 'buy' and t['timestamp'] < trade['timestamp']), None)
            if entry_price:
                returns.append((trade['price'] - entry_price) / entry_price)

    if not returns:
        return {'sharpe': 0.0, 'total_return': 0.0}

    returns = np.array(returns)
    mean_return = np.mean(returns)
    std_return = np.std(returns) if len(returns) > 1 else 1.0
    sharpe = mean_return / std_return * np.sqrt(252) if std_return != 0 else 0.0
    total_return = np.prod(1 + returns) - 1

    return {
        'sharpe': sharpe,
        'total_return': total_return
    }


def run_integrated_strategy(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """Run integrated AI and nGS strategy based on operating mode."""
    logger.info(f"Running integrated strategy in {self.operating_mode} mode")

    if self.operating_mode == 'ai_only':
        results = self._run_ai_strategies_only(data)
    else:  # comparison mode
        ai_results = self._run_ai_strategies_only(data)
        original_results = self._run_original_strategy(data)
        results = {
            'ai': ai_results,
            'original': original_results,
            'comparison': self._compare_strategies(ai_results, original_results)
        }

    self.strategy_performance.update(results)
    return results


def _compare_strategies(self, ai_results: Dict, original_results: Dict) -> Dict[str, Any]:
    """Compare AI and original strategy performance."""
    comparison = {}
    for symbol in ai_results.keys():
        try:
            ai_perf = ai_results[symbol]['performance']
            orig_perf = original_results.get(symbol, {}).get('performance', {'sharpe': 0.0, 'total_return': 0.0})

            comparison[symbol] = {
                'ai_sharpe': ai_perf['sharpe'],
                'original_sharpe': orig_perf['sharpe'],
                'ai_total_return': ai_perf['total_return'],
                'original_total_return': orig_perf['total_return'],
                'sharpe_diff': ai_perf['sharpe'] - orig_perf['sharpe'],
                'return_diff': ai_perf['total_return'] - orig_perf['total_return']
            }
        except Exception as e:
            logger.error(f"Error comparing strategies for {symbol}: {str(e)}")
            comparison[symbol] = {}

    return comparison


def rebalance_portfolio(self, data: Dict[str, pd.DataFrame]) -> None:
    """Rebalance portfolio based on AI strategy performance."""
    if self.integration_config['rebalance_frequency'] == 'weekly':
        current_time = datetime.now()
        if current_time.weekday() != 0:  # Not Monday
            logger.debug("Skipping rebalance: not a Monday")
            return

    performance = self.strategy_performance
    allocations = {}
    total_weight = 0

    for symbol, result in performance.items():
        if isinstance(result, dict) and 'performance' in result:
            weight = result['performance']['sharpe'] if result['performance']['sharpe'] > 0 else 0
            allocations[symbol] = weight
            total_weight += weight

    if total_weight == 0:
        logger.warning("No positive Sharpe ratios found for rebalancing")
        return

    for symbol in allocations:
        allocations[symbol] = (allocations[symbol] / total_weight) * self.integration_config['ai_allocation_pct']
        logger.debug(f"Rebalanced allocation for {symbol}: {allocations[symbol]:.2f}%")

    # Update active strategies based on new allocations
    objective = self.objective_manager.get_primary_objective()
    self.active_strategies = self.create_ai_strategy_set(data, objective)
    logger.info("Portfolio rebalanced based on performance")


def save_integration_session(self, results: Dict[str, Any], filename: str = "integration_results.json") -> None:
    """Save integration results to a file."""
    try:
        output_path = os.path.join(self.results_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved integration results to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save integration results: {str(e)}")


def plot_performance_comparison(self, results: Dict[str, Any]) -> None:
    """Plot performance comparison between AI and original strategies."""
    if 'comparison' not in results:
        logger.warning("No comparison data available for plotting")
        return

    plt.figure(figsize=(12, 6))
    symbols = list(results['comparison'].keys())
    ai_sharpes = [results['comparison'][s]['ai_sharpe'] for s in symbols]
    orig_sharpes = [results['comparison'][s]['original_sharpe'] for s in symbols]

    x = np.arange(len(symbols))
    width = 0.35

    plt.bar(x - width / 2, ai_sharpes, width, label='AI Strategy')
    plt.bar(x + width / 2, orig_sharpes, width, label='Original nGS')

    plt.xlabel('Symbols')
    plt.ylabel('Sharpe Ratio')
    plt.title('AI vs Original nGS Strategy Performance')
    plt.xticks(x, symbols, rotation=45)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(self.results_dir, 'performance_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved performance comparison plot to {plot_path}")

>>>>>>> c108ef4 (Bypass pre-commit for now)

def run_ngs_automated_reporting(comparison=None):
    import pandas as pd
    import os

    HISTORICAL_DATA_PATH = "signal_analysis.json"
    DATA_FORMAT = "json"

    def load_data(path, data_format):
        if data_format == "json":
            df = pd.read_json(path)
            if "symbol" in df.columns:
                data_dict = {sym: df[df["symbol"] == sym].copy() for sym in df["symbol"].unique()}
                return data_dict
            return {"default": df}
        elif data_format == "csv":
            df = pd.read_csv(path)
            if "symbol" in df.columns:
                data_dict = {sym: df[df["symbol"] == sym].copy() for sym in df["symbol"].unique()}
                return data_dict
            return {"default": df}
        else:
            raise ValueError("Unsupported data format")

    print("🚀 nGS Trading Strategy with AI SELECTION ENABLED")
    print("=" * 70)

    # Load historical data
    data = load_data(HISTORICAL_DATA_PATH, DATA_FORMAT)

    # Initialize integration manager
    manager = NGSAIIntegrationManager(account_size=1_000_000, data_dir="data")

    # Run AI integration manager
    results = manager.run_integrated_strategy(data)

    # Save results
    manager.save_integration_session(results, filename="latest_results.json")
<<<<<<< HEAD
    print("\n✅ AI integration complete. Results saved for dashboard.")

if __name__ == "__main__":
    run_ngs_automated_reporting()
=======
    print("\n AI integration complete. Results saved for dashboard.")


if __name__ == "__main__":
    run_ngs_automated_reporting()
>>>>>>> c108ef4 (Bypass pre-commit for now)

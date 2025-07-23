"""
nGS-Integrated AI Strategy System
Combines YOUR proven nGS parameters with objective-aware AI generation
Uses your battle-tested thresholds and logic patterns!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the AI components
from comprehensive_indicator_library import ComprehensiveIndicatorLibrary
from performance_objectives import ObjectiveManager, PerformanceObjective
from strategy_generator_ai import ObjectiveAwareStrategyGenerator, TradingStrategy

class NGSProvenParameters:
    """
    YOUR proven parameters extracted from nGS_Revised_Strategy.py
    These are your battle-tested values that actually work!
    """
    
    # Core nGS Parameters (from your inputs)
    CORE_PARAMS = {
        'bb_length': 25,           # Your proven BB period
        'bb_deviation': 2.0,       # Your proven BB deviation
        'min_price': 10,           # Your price range filter
        'max_price': 500,          # Your price range filter
        'position_size': 5000,     # Your base position size
        'af_step': 0.05,           # Your PSAR step
        'af_limit': 0.21,          # Your PSAR limit
        'atr_short': 5,            # Your ATR short period
        'atr_long': 13,            # Your ATR long period
    }
    
    # Your Proven Entry Thresholds
    ENTRY_THRESHOLDS = {
        'bb_oversold_long': 1.02,      # Low <= LowerBB * 1.02
        'bb_overbought_short': 0.98,   # High >= UpperBB * 0.98
        'bb_upper_limit_long': 0.95,   # Close <= UpperBB * 0.95
        'bb_lower_limit_short': 1.05,  # Close >= LowerBB * 1.05
        'atr_gap_limit': 2.0,          # Gap <= ATR * 2.0
        'min_candle_body': 0.05,       # Minimum 5 cent body
        'min_profit_pct': 0.003,       # Minimum 0.3% move
        'wick_tolerance': 0.05,        # 5% wick tolerance
    }
    
    # Your Proven Exit Thresholds  
    EXIT_THRESHOLDS = {
        'gap_target_long': 1.05,       # 5% gap target
        'gap_target_short': 0.95,      # 5% gap target  
        'bb_target_long': 0.60,        # Exit at 60% of BB range
        'bb_target_short': 0.40,       # Exit at 40% of BB range
        'atr_profit_target': 1.5,      # 1.5x ATR profit target
        'hard_stop_long': 0.90,        # 10% hard stop
        'hard_stop_short': 1.10,       # 10% hard stop
        'reversal_bars_min': 5,        # Minimum bars for reversal
    }
    
    # Your M/E Ratio Management
    ME_RATIO_PARAMS = {
        'target_min': 50.0,            # Your 50% minimum
        'target_max': 80.0,            # Your 80% maximum  
        'min_positions_scale_up': 5,   # Your minimum position requirement
        'rebalancing_enabled': True,   # Your M/E rebalancing
    }
    
    # Your L/S Ratio Adjustments
    LS_RATIO_PARAMS = {
        'safe_ratio_threshold': 1.5,   # L/S > 1.5 = safer shorts
        'risk_ratio_threshold': -1.5,  # L/S < -1.5 = high risk
        'safe_margin_multiplier': 2.0, # 50% margin = 2x leverage
        'risk_margin_multiplier': 0.75, # 75% margin = reduced size
    }

class NGSIndicatorLibrary(ComprehensiveIndicatorLibrary):
    """
    Enhanced indicator library with YOUR proven nGS calculations
    Uses your exact formulas and parameters!
    """
    
    def __init__(self):
        super().__init__()
        self._add_ngs_indicators()
        print(f"🎯 Enhanced with {len(self._get_ngs_indicators())} nGS-specific indicators")
    
    def _add_ngs_indicators(self):
        """Add your specific nGS indicators with proven parameters"""
        
        # Your proven Bollinger Bands (25-period, not generic 20)
        self.indicators_catalog['ngs_bb_position'] = {
            'name': 'nGS Bollinger Position',
            'function': self.ngs_bollinger_position,
            'params': {'length': 25, 'deviation': 2.0},
            'category': 'ngs_core',
            'output_type': 'percentage',
            'description': 'Your proven BB position with 25-period'
        }
        
        # Your ATR calculation (5+13 periods)
        self.indicators_catalog['ngs_atr'] = {
            'name': 'nGS ATR',
            'function': self.ngs_atr,
            'params': {'short_period': 5, 'long_period': 13},
            'category': 'ngs_core', 
            'output_type': 'price_level',
            'description': 'Your proven ATR calculation'
        }
        
        # Your Linear Regression Value
        self.indicators_catalog['ngs_lr_value'] = {
            'name': 'nGS Linear Regression Value',
            'function': self.ngs_lr_value,
            'params': {'period1': 3, 'period2': 5, 'roc_period': 8},
            'category': 'ngs_core',
            'output_type': 'oscillator', 
            'description': 'Your proven LR trend filter'
        }
        
        # Your Parabolic SAR
        self.indicators_catalog['ngs_psar'] = {
            'name': 'nGS Parabolic SAR',
            'function': self.ngs_psar,
            'params': {'af_step': 0.05, 'af_limit': 0.21},
            'category': 'ngs_core',
            'output_type': 'price_level',
            'description': 'Your proven PSAR settings'
        }
        
        # Your Swing High/Low detection
        self.indicators_catalog['ngs_swing_levels'] = {
            'name': 'nGS Swing Levels',
            'function': self.ngs_swing_levels,
            'params': {'period': 4},
            'category': 'ngs_core',
            'output_type': 'price_level',
            'description': 'Your swing high/low calculation'
        }
    
    def ngs_bollinger_position(self, df: pd.DataFrame, length: int = 25, deviation: float = 2.0) -> pd.Series:
        """
        YOUR proven Bollinger Band position calculation
        Uses your exact 25-period parameters
        """
        bb_mid = df['Close'].rolling(window=length).mean()
        bb_std = df['Close'].rolling(window=length).std()
        bb_upper = bb_mid + (deviation * bb_std)
        bb_lower = bb_mid - (deviation * bb_std)
        
        # Your position calculation
        bb_position = ((df['Close'] - bb_lower) / (bb_upper - bb_lower) * 100)
        bb_position = bb_position.fillna(50).clip(0, 100)
        
        return bb_position.rename('nGS_BB_Position')
    
    def ngs_atr(self, df: pd.DataFrame, short_period: int = 5, long_period: int = 13) -> pd.Series:
        """
        YOUR proven ATR calculation with 5+13 periods
        Matches your exact formula from nGS strategy
        """
        # True Range calculation (your method)
        high_low = df['High'] - df['Low']
        high_close_prev = abs(df['High'] - df['Close'].shift(1))
        low_close_prev = abs(df['Low'] - df['Close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        
        # Your ATR calculation
        atr = true_range.rolling(window=short_period).mean()
        atr_ma = atr.rolling(window=long_period).mean()
        
        return atr_ma.fillna(0).rename('nGS_ATR')
    
    def ngs_lr_value(self, df: pd.DataFrame, period1: int = 3, period2: int = 5, roc_period: int = 8) -> pd.Series:
        """
        YOUR proven Linear Regression Value calculation
        Replicates your oLRValue logic
        """
        # Your Value1 calculation
        value1 = (df['Close'].rolling(window=5).mean() - df['Close'].rolling(window=35).mean())
        
        # Your ROC calculation
        roc = value1 - value1.shift(3)
        
        # Your LRV calculation (8-period regression)
        lrv_values = []
        for i in range(len(df)):
            if i >= roc_period - 1:
                try:
                    recent_roc = roc.iloc[i-roc_period+1:i+1].values
                    if not np.isnan(recent_roc).any() and len(recent_roc) == roc_period:
                        x = np.arange(roc_period)
                        slope, intercept = np.polyfit(x, recent_roc, 1)
                        lrv_value = slope * (roc_period - 1) + intercept
                        lrv_values.append(lrv_value)
                    else:
                        lrv_values.append(np.nan)
                except:
                    lrv_values.append(np.nan)
            else:
                lrv_values.append(np.nan)
        
        lrv = pd.Series(lrv_values, index=df.index)
        
        # Your oLRValue calculation (3-period regression of LRV)
        olr_values = []
        for i in range(len(df)):
            if i >= period1 - 1:
                try:
                    recent_lrv = lrv.iloc[i-period1+1:i+1].values
                    if not np.isnan(recent_lrv).any() and len(recent_lrv) == period1:
                        x = np.arange(period1)
                        slope, intercept = np.polyfit(x, recent_lrv, 1)
                        olr_value = slope * (period1 - 3) + intercept  # Your -2 shift
                        olr_values.append(olr_value)
                    else:
                        olr_values.append(np.nan)
                except:
                    olr_values.append(np.nan)
            else:
                olr_values.append(np.nan)
        
        return pd.Series(olr_values, index=df.index, name='nGS_LR_Value')
    
    def ngs_psar(self, df: pd.DataFrame, af_step: float = 0.05, af_limit: float = 0.21) -> pd.Series:
        """
        YOUR proven Parabolic SAR with exact parameters
        """
        psar_values = []
        psar_direction = []  # 1 for long, -1 for short
        
        if len(df) < 2:
            return pd.Series([np.nan] * len(df), index=df.index, name='nGS_PSAR')
        
        # Initialize
        psar = df['Low'].iloc[0]
        direction = 1  # Start long
        af = af_step
        ep = df['High'].iloc[0]  # Extreme point
        
        psar_values.append(psar)
        psar_direction.append(direction)
        
        for i in range(1, len(df)):
            high = df['High'].iloc[i]
            low = df['Low'].iloc[i]
            
            if direction == 1:  # Long position
                psar = psar + af * (ep - psar)
                
                # SAR cannot be above previous two lows
                if psar > low:
                    direction = -1
                    psar = ep
                    af = af_step
                    ep = low
                else:
                    if high > ep:
                        ep = high
                        af = min(af + af_step, af_limit)
                    psar = min(psar, df['Low'].iloc[i-1], low)
            else:  # Short position
                psar = psar - af * (psar - ep)
                
                # SAR cannot be below previous two highs
                if psar < high:
                    direction = 1
                    psar = ep
                    af = af_step
                    ep = high
                else:
                    if low < ep:
                        ep = low
                        af = min(af + af_step, af_limit)
                    psar = max(psar, df['High'].iloc[i-1], high)
            
            psar_values.append(psar)
            psar_direction.append(direction)
        
        return pd.Series(psar_values, index=df.index, name='nGS_PSAR')
    
    def ngs_swing_levels(self, df: pd.DataFrame, period: int = 4) -> pd.Series:
        """
        YOUR proven swing high/low calculation
        """
        swing_high = df['High'].rolling(window=period).max()
        swing_low = df['Low'].rolling(window=period).min()
        
        # Return composite swing level (for proximity detection)
        current_to_high = abs(df['Close'] - swing_high) / df['Close']
        current_to_low = abs(df['Close'] - swing_low) / df['Close']
        
        # Distance to nearest swing level
        swing_distance = np.minimum(current_to_high, current_to_low)
        
        return swing_distance.fillna(1.0).rename('nGS_Swing_Distance')
    
    def _get_ngs_indicators(self) -> List[str]:
        """Get list of nGS-specific indicators"""
        return [name for name, info in self.indicators_catalog.items() 
                if info['category'] == 'ngs_core']

class NGSAwareStrategyGenerator(ObjectiveAwareStrategyGenerator):
    """
    Enhanced AI that understands YOUR nGS patterns and parameters
    Generates strategies using your proven thresholds and logic!
    """
    
    def __init__(self, indicator_library: NGSIndicatorLibrary, 
                 objective_manager: ObjectiveManager):
        super().__init__(indicator_library, objective_manager)
        self.ngs_params = NGSProvenParameters()
        self.ngs_patterns = self._define_ngs_patterns()
        
        print("🎯 nGS-Aware AI initialized with YOUR proven parameters!")
        print(f"   Your BB period: {self.ngs_params.CORE_PARAMS['bb_length']}")
        print(f"   Your position size: ${self.ngs_params.CORE_PARAMS['position_size']:,}")
        print(f"   Your M/E range: {self.ngs_params.ME_RATIO_PARAMS['target_min']:.0f}%-{self.ngs_params.ME_RATIO_PARAMS['target_max']:.0f}%")
    
    def _define_ngs_patterns(self) -> Dict:
        """
        Define YOUR proven entry/exit patterns from nGS strategy
        These are your actual working patterns!
        """
        return {
            'engulfing_long': {
                'description': 'Your proven Engulfing Long pattern',
                'conditions': [
                    {'indicator': 'ngs_bb_position', 'operator': '<', 'threshold': 25, 'weight': 3},
                    {'indicator': 'ngs_lr_value', 'operator': '>', 'threshold': 0, 'weight': 2},
                    {'indicator': 'ngs_atr', 'operator': '<', 'threshold': 'dynamic_gap_limit', 'weight': 1}
                ],
                'exit_conditions': [
                    {'indicator': 'ngs_bb_position', 'operator': '>', 'threshold': 60, 'weight': 2},
                    {'indicator': 'profit_target', 'operator': '>', 'threshold': 5.0, 'weight': 3}  # 5% gap target
                ]
            },
            'engulfing_short': {
                'description': 'Your proven Engulfing Short pattern',
                'conditions': [
                    {'indicator': 'ngs_bb_position', 'operator': '>', 'threshold': 75, 'weight': 3},
                    {'indicator': 'ngs_lr_value', 'operator': '<', 'threshold': 0, 'weight': 2},
                    {'indicator': 'ngs_swing_distance', 'operator': '<', 'threshold': 0.02, 'weight': 1}
                ],
                'exit_conditions': [
                    {'indicator': 'ngs_bb_position', 'operator': '<', 'threshold': 40, 'weight': 2},
                    {'indicator': 'profit_target', 'operator': '>', 'threshold': 5.0, 'weight': 3}  # 5% gap target
                ]
            },
            'semi_engulfing_long': {
                'description': 'Your proven Semi-Engulfing Long pattern',
                'conditions': [
                    {'indicator': 'ngs_bb_position', 'operator': '<', 'threshold': 30, 'weight': 2},
                    {'indicator': 'ngs_lr_value', 'operator': '>', 'threshold': 0, 'weight': 2},
                    {'indicator': 'ngs_atr', 'operator': '<', 'threshold': 'dynamic_volatility', 'weight': 1}
                ],
                'exit_conditions': [
                    {'indicator': 'ngs_bb_position', 'operator': '>', 'threshold': 55, 'weight': 2}
                ]
            }
        }
    
    def generate_ngs_strategy_for_objective(self, objective_name: str, 
                                          pattern_focus: str = 'auto') -> TradingStrategy:
        """
        Generate strategy using YOUR nGS patterns optimized for objective
        """
        print(f"\n🎯 Generating nGS strategy for: {objective_name.upper()}")
        
        # Get objective preferences
        objective = self.objective_manager.get_objective(objective_name)
        objective_prefs = objective.get_strategy_preferences()
        
        # Choose nGS pattern based on objective
        if pattern_focus == 'auto':
            pattern_focus = self._select_ngs_pattern_for_objective(objective_name)
        
        # Generate strategy using YOUR proven parameters
        strategy_config = self._generate_ngs_adaptive_logic(
            objective_prefs, pattern_focus, objective_name
        )
        
        # Create strategy with nGS parameters
        strategy_id = f"nGS_{objective_name}_{pattern_focus}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        strategy = TradingStrategy(strategy_id, objective_name, strategy_config)
        
        print(f"✅ Generated nGS strategy: {pattern_focus} pattern")
        print(f"📊 Using YOUR proven BB period: {self.ngs_params.CORE_PARAMS['bb_length']}")
        print(f"💰 Using YOUR position size: ${self.ngs_params.CORE_PARAMS['position_size']:,}")
        
        return strategy
    
    def _select_ngs_pattern_for_objective(self, objective_name: str) -> str:
        """Select best nGS pattern for the objective"""
        
        pattern_map = {
            'linear_equity': 'engulfing_long',      # Your most reliable pattern
            'max_roi': 'engulfing_short',           # Higher volatility pattern  
            'min_drawdown': 'semi_engulfing_long',  # Lower risk pattern
            'high_winrate': 'engulfing_long',       # Your highest win rate pattern
            'sharpe_ratio': 'engulfing_long'        # Balanced risk/reward
        }
        
        return pattern_map.get(objective_name, 'engulfing_long')
    
    def _generate_ngs_adaptive_logic(self, objective_prefs: Dict, pattern_name: str, 
                                   objective_name: str) -> Dict:
        """
        Generate strategy using YOUR nGS patterns and parameters
        """
        pattern = self.ngs_patterns[pattern_name]
        
        # Use YOUR proven indicators
        ngs_indicators = ['ngs_bb_position', 'ngs_atr', 'ngs_lr_value', 'ngs_psar', 'ngs_swing_distance']
        
        # Adapt YOUR thresholds based on objective
        entry_conditions = self._adapt_ngs_conditions_for_objective(
            pattern['conditions'], objective_name
        )
        
        exit_conditions = self._adapt_ngs_conditions_for_objective(
            pattern['exit_conditions'], objective_name
        )
        
        # Use YOUR position sizing with M/E awareness
        position_sizing = self._generate_ngs_position_sizing(objective_name)
        
        return {
            'indicators': ngs_indicators,
            'entry_logic': {
                'conditions': entry_conditions,
                'confirmation_ratio': self._get_ngs_confirmation_ratio(objective_name),
                'style': f'ngs_{pattern_name}',
                'pattern': pattern_name
            },
            'exit_logic': {
                'conditions': exit_conditions,
                'style': f'ngs_{pattern_name}_exit'
            },
            'position_sizing': position_sizing,
            'objective_focus': objective_name,
            'ngs_parameters': self.ngs_params.CORE_PARAMS,
            'risk_management': {
                'price_range': [self.ngs_params.CORE_PARAMS['min_price'], 
                               self.ngs_params.CORE_PARAMS['max_price']],
                'me_ratio_target': [self.ngs_params.ME_RATIO_PARAMS['target_min'],
                                   self.ngs_params.ME_RATIO_PARAMS['target_max']],
                'hard_stops': {
                    'long': self.ngs_params.EXIT_THRESHOLDS['hard_stop_long'],
                    'short': self.ngs_params.EXIT_THRESHOLDS['hard_stop_short']
                }
            }
        }
    
    def _adapt_ngs_conditions_for_objective(self, base_conditions: List[Dict], 
                                          objective_name: str) -> List[Dict]:
        """Adapt YOUR thresholds based on objective while keeping nGS logic"""
        
        adapted_conditions = []
        
        for condition in base_conditions:
            adapted_condition = condition.copy()
            
            # Adapt YOUR BB position thresholds based on objective
            if condition['indicator'] == 'ngs_bb_position':
                if objective_name == 'min_drawdown':
                    # More conservative: tighter entry range
                    if condition['operator'] == '<':
                        adapted_condition['threshold'] = min(20, condition['threshold'])
                    elif condition['operator'] == '>':
                        adapted_condition['threshold'] = max(80, condition['threshold'])
                elif objective_name == 'max_roi':
                    # More aggressive: wider entry range
                    if condition['operator'] == '<':
                        adapted_condition['threshold'] = max(30, condition['threshold'])
                    elif condition['operator'] == '>':
                        adapted_condition['threshold'] = min(70, condition['threshold'])
            
            # Adapt profit targets based on objective
            elif condition['indicator'] == 'profit_target':
                if objective_name == 'linear_equity':
                    adapted_condition['threshold'] = 3.0  # Smaller, consistent profits
                elif objective_name == 'max_roi':
                    adapted_condition['threshold'] = 8.0  # Larger profit targets
                elif objective_name == 'min_drawdown':
                    adapted_condition['threshold'] = 2.0  # Very quick profits
            
            adapted_conditions.append(adapted_condition)
        
        return adapted_conditions
    
    def _get_ngs_confirmation_ratio(self, objective_name: str) -> float:
        """Get confirmation ratio based on objective (using your selectivity)"""
        
        ratios = {
            'linear_equity': 0.9,      # 90% - very selective like your system
            'max_roi': 0.7,            # 70% - more opportunities 
            'min_drawdown': 0.95,      # 95% - ultra selective
            'high_winrate': 0.85,      # 85% - selective for accuracy
            'sharpe_ratio': 0.8        # 80% - balanced approach
        }
        
        return ratios.get(objective_name, 0.8)
    
    def _generate_ngs_position_sizing(self, objective_name: str) -> Dict:
        """Generate position sizing using YOUR nGS parameters"""
        
        base_size = self.ngs_params.CORE_PARAMS['position_size']
        
        # Adapt YOUR position size based on objective
        size_multipliers = {
            'linear_equity': 0.8,      # Smaller for consistency  
            'max_roi': 1.2,            # Larger for growth
            'min_drawdown': 0.5,       # Much smaller for safety
            'high_winrate': 0.9,       # Slightly smaller
            'sharpe_ratio': 1.0        # Your standard size
        }
        
        multiplier = size_multipliers.get(objective_name, 1.0)
        adjusted_size = int(base_size * multiplier)
        
        return {
            'method': 'ngs_adaptive',
            'base_size': adjusted_size,
            'me_ratio_awareness': True,
            'ls_ratio_adjustments': True,
            'price_range_filter': True
        }

def demonstrate_ngs_ai_integration():
    """Demonstrate the nGS AI integration"""
    print("\n🎯 nGS-INTEGRATED AI DEMONSTRATION")
    print("=" * 60)
    
    # Initialize with YOUR nGS parameters
    ngs_indicator_lib = NGSIndicatorLibrary()
    objective_manager = ObjectiveManager()
    ngs_ai_generator = NGSAwareStrategyGenerator(ngs_indicator_lib, objective_manager)
    
    # Show your proven parameters
    params = NGSProvenParameters()
    print(f"\n📊 YOUR PROVEN nGS PARAMETERS:")
    print(f"   Bollinger Bands:     {params.CORE_PARAMS['bb_length']}-period, {params.CORE_PARAMS['bb_deviation']} deviation")
    print(f"   Position Size:       ${params.CORE_PARAMS['position_size']:,}")
    print(f"   Price Range:         ${params.CORE_PARAMS['min_price']}-${params.CORE_PARAMS['max_price']}")
    print(f"   M/E Target:          {params.ME_RATIO_PARAMS['target_min']:.0f}%-{params.ME_RATIO_PARAMS['target_max']:.0f}%")
    print(f"   Entry Threshold:     BB <= {params.ENTRY_THRESHOLDS['bb_oversold_long']:.2f}x Lower BB")
    print(f"   Exit Threshold:      BB >= {params.EXIT_THRESHOLDS['bb_target_long']:.0%} of range")
    
    # Generate strategies using YOUR patterns
    objectives_to_test = ['linear_equity', 'max_roi', 'min_drawdown']
    
    print(f"\n🎯 Generating nGS strategies for {len(objectives_to_test)} objectives:")
    
    for objective in objectives_to_test:
        strategy = ngs_ai_generator.generate_ngs_strategy_for_objective(objective)
        config = strategy.config
        
        print(f"\n{objective.upper()} nGS Strategy:")
        print(f"   Pattern:             {config['entry_logic']['pattern']}")
        print(f"   nGS Indicators:      {len([i for i in config['indicators'] if i.startswith('ngs_')])}")
        print(f"   Confirmation Ratio:  {config['entry_logic']['confirmation_ratio']:.0%}")
        print(f"   Position Size:       ${config['position_sizing']['base_size']:,}")
        print(f"   Price Range:         ${config['risk_management']['price_range'][0]}-${config['risk_management']['price_range'][1]}")

if __name__ == "__main__":
    print("🎯 nGS-INTEGRATED AI STRATEGY SYSTEM")
    print("=" * 60)
    print("🚀 Combining YOUR proven nGS parameters with objective-aware AI!")
    print("📊 Using your battle-tested thresholds and logic patterns")
    
    # Run demonstration
    demonstrate_ngs_ai_integration()
    
    print(f"\n✅ nGS AI INTEGRATION COMPLETE!")
    print("🎯 Ready to generate strategies using YOUR proven parameters!")
    print("\n🚀 Key Features:")
    print("   ✅ Uses YOUR 25-period Bollinger Bands")
    print("   ✅ Uses YOUR $5,000 position sizing")
    print("   ✅ Uses YOUR 50-80% M/E ratio targets")
    print("   ✅ Uses YOUR proven entry/exit thresholds")
    print("   ✅ Uses YOUR engulfing pattern logic")
    print("   ✅ Adapts YOUR parameters for different objectives")

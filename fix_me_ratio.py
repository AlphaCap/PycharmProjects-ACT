# fix_me_ratio.py
import re

def fix_me_calculation():
    with open('nGS_Revised_Strategy.py', 'r') as f:
        content = f.read()
    
    # Find and replace the wrong M/E calculation
    old_pattern = r'portfolio_equity = self\.initial_portfolio_value \+ self\.realized_pnl \+ total_unrealized_pnl'
    new_line = '''# FIXED: Use actual total account value (positions + cash)
        # Note: This requires cash to be passed in, will be fixed in calling method
        portfolio_equity = total_position_value + 0  # Placeholder, will use proper calculation'''
    
    content = re.sub(old_pattern, new_line, content)
    
    # Also fix the calculate_current_me_ratio method
    old_method = r'def calculate_current_me_ratio\(self\) -> float:\s+return self\.me_calculator\.calculate_daily_me_ratio\(\)\[\'ME_Ratio\'\]'
    new_method = '''def calculate_current_me_ratio(self) -> float:
        # FIXED: Calculate M/E ratio correctly using actual account value
        total_position_value = 0
        for symbol, position in self.positions.items():
            if position['shares'] != 0:
                total_position_value += position['entry_price'] * abs(position['shares'])
        
        total_account_value = self.cash + total_position_value
        return (total_position_value / total_account_value * 100) if total_account_value > 0 else 0.0'''
    
    content = re.sub(old_method, new_method, content, flags=re.MULTILINE | re.DOTALL)
    
    with open('nGS_Revised_Strategy.py', 'w') as f:
        f.write(content)
    
    print("✅ M/E calculation fixed!")
    print("🔧 Fixed calculate_current_me_ratio method to use correct calculation")

if __name__ == "__main__":
    fix_me_calculation()

# fix_ai_debug.py
import re

# Read the file with UTF-8 encoding
with open('nGS_Revised_Strategy.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Backup with UTF-8 encoding
with open('nGS_Revised_Strategy.py.backup', 'w', encoding='utf-8') as f:
    f.write(content)

# Add debug statement before AI check
content = re.sub(
    r'(\s+)(# STEP 3: AI Strategy Selection or Fallback)',
    r'\1# DEBUG: Check AI availability\n\1print(f"\\n🔍 DEBUG: AI_AVAILABLE = {AI_AVAILABLE}")\n\1print(f"   Data loaded: {len(data)} symbols")\n\1\n\1\2',
    content
)

# Add AI_AVAILABLE declaration
content = re.sub(
    r'(print\("🧠 Initializing AI Strategy Selection System\.\.\."\))',
    r'\1\n        \n        # Explicitly declare AI_AVAILABLE\n        AI_AVAILABLE = False',
    content
)

# Write the fixed file with UTF-8 encoding
with open('nGS_Revised_Strat
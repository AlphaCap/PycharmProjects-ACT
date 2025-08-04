# simple_fix.py
# Read the file
with open('nGS_Revised_Strategy.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Backup
with open('nGS_Revised_Strategy.py.backup', 'w', encoding='utf-8') as f:
    f.writelines(lines)

# Find and insert debug line
for i in range(len(lines)):
    if 'if AI_AVAILABLE:' in lines[i] and 'STEP 3' in lines[i-2]:
        # Insert debug line before this line
        indent = '        '  # 8 spaces
        debug_line = f'{indent}print(f"DEBUG: AI_AVAILABLE = {{AI_AVAILABLE}}")\n'
        lines.insert(i, debug_line)
        break

# Write back
with open('nGS_Revised_Strategy.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Debug line added!")
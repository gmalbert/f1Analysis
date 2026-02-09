"""
Script to fix tab4 predictions indentation.
The code from line 3147 onwards needs 8 more spaces of indentation.
"""

# Read the file
with open('raceAnalysis.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Lines 3147-3550 need 8 MORE spaces (currently at 4, need to be at 12)
start_idx = 3146  # Line 3147 (0-indexed)
end_idx = 3550    # Approximate end of predictions block

modified_count = 0

# Process lines - add 8 spaces while preserving relative indentation
for i in range(start_idx, min(end_idx, len(lines))):
    line = lines[i]
    if line.strip():  # Non-empty line
        lines[i] = "        " + line  # Add 8 spaces
        modified_count += 1

# Write the fixed file
with open('raceAnalysis.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Fixed indentation for {modified_count} lines ({start_idx+1} to {min(end_idx, len(lines))})")

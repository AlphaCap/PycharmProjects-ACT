import json
import os
from data_manager import get_positions  # Use your actual module

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Get position data
positions = get_positions()

# Export to JSON file
with open("data/positions.json", "w") as f:
    json.dump(positions, f, indent=4)

print("Position data exported to data/positions.json")
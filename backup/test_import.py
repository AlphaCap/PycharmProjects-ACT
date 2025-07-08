import os
import sys

print("Current working directory:", os.getcwd())
print("Python module search paths:")
for path in sys.path:
    print("  -", path)

print("\nChecking for polygon_api.py file:")
file_path = os.path.join(os.getcwd(), "polygon_api.py")
if os.path.exists(file_path):
    print(f"  File exists at {file_path}")
    print(f"  File size: {os.path.getsize(file_path)} bytes")
else:
    print(f"  File NOT found at {file_path}")

print("\nListing directory contents:")
for file in os.listdir(os.getcwd()):
    print("  -", file)

print("\nTrying import:")
try:
    import polygon_api
    print("  SUCCESS! Module imported!")
except Exception as e:
    print(f"  FAILED! Error: {e}")
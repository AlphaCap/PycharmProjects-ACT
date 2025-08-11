import requests
from utils.config import POLYGON_API_KEY

print(f"API Key from config: {POLYGON_API_KEY}")

# Test the API key
url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-10?apiKey={POLYGON_API_KEY}"
response = requests.get(url)
print(f"Status code: {response.status_code}")
print(f"Response: {response.text[:200]}")



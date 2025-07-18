import os
from polygon import RESTClient

print("Testing Polygon API response...")

try:
    client = RESTClient(os.getenv('POLYGON_API_KEY'))
    response = client.get_aggs('AAPL', 1, 'day', '2025-06-16', '2025-07-16')
    
    print('Response type:', type(response))
    print('Response content:', response)
    
    if hasattr(response, 'results'):
        print('✅ Has results attribute')
        print('Results:', response.results)
    else:
        print('❌ NO results attribute')
        print('Available attributes:', [attr for attr in dir(response) if not attr.startswith('_')])
        
    # Check if it's a list
    if isinstance(response, list):
        print('Response is a LIST with', len(response), 'items')
        if response:
            print('First item:', response[0])
    
except Exception as e:
    print('Error:', e)

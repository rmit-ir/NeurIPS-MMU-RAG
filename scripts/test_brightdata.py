import os

import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("BRIGHT_DATA_API_KEY")
if not api_key:
    raise RuntimeError("BRIGHT_DATA_API_KEY not found in environment")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

data = {
    "zone": "serp_api_test_1",
    "url": "https://www.google.com/search?q=pizza",
    "format": "raw",
}

response = requests.post(
    "https://api.brightdata.com/request",
    json=data,
    headers=headers,
)

print(f"Status: {response.status_code}")
print(response.text[:2000])

# test_connection.py
import requests
import urllib3

# Disable SSL warnings temporarily for testing
urllib3.disable_warnings()

print("Testing connection to Celestrak...")

# Test 1: Basic connection
try:
    response = requests.get("https://celestrak.org", timeout=10, verify=False)
    print(f"✓ HTTPS connection - Status: {response.status_code}")
except Exception as e:
    print(f"✗ HTTPS connection - Error: {type(e).__name__}: {e}")

# Test 2: Direct IP
try:
    response = requests.get("https://104.168.149.178", timeout=10, verify=False)
    print(f"✓ Direct IP - Status: {response.status_code}")
except Exception as e:
    print(f"✗ Direct IP - Error: {type(e).__name__}: {e}")

# Test 3: TLE endpoint
try:
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle"
    response = requests.get(url, timeout=10, verify=False)
    print(f"✓ TLE endpoint - Status: {response.status_code}")
    print(f"  First 200 chars: {response.text[:200]}")
except Exception as e:
    print(f"✗ TLE endpoint - Error: {type(e).__name__}: {e}")

# Test 4: With headers
try:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get("https://celestrak.org", headers=headers, timeout=10)
    print(f"✓ With headers - Status: {response.status_code}")
except Exception as e:
    print(f"✗ With headers - Error: {type(e).__name__}: {e}")
"""Fetches real TLE data from NORAD/Celestrak."""

import requests
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm

from ..utils.config import TLE_SOURCES, CACHE_DIR, API_CONFIG


class TLEFetcher:
    """Fetches and caches Two-Line Element data from Celestrak."""
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.sources = TLE_SOURCES
        self.session = requests.Session()
        # Add retry strategy
        retry_strategy = requests.packages.urllib3.util.retry.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def _get_cache_path(self, source_name: str) -> Path:
        """Get cache file path for a given source."""
        return self.cache_dir / f"{source_name}_tle_cache.json"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache is still valid."""
        if not cache_path.exists():
            return False
            
        # Check cache age
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age_hours = (datetime.now() - cache_time).total_seconds() / 3600
        
        return age_hours < API_CONFIG["cache_expiry_hours"]
    
    def _parse_float(self, s: str) -> float:
        """Parse a float from TLE format."""
        return float(s.strip() or '0')
    
    def _parse_float_exp(self, s: str) -> float:
        """Parse a float in exponential format from TLE."""
        s = s.strip()
        if not s or s == '00000+0' or s == '00000-0':
            return 0.0
        
        # Handle the special TLE exponential format (e.g., '12345-4' means 0.12345e-4)
        if '-' in s[1:] or '+' in s[1:]:
            # Find the exponent sign
            for i in range(1, len(s)):
                if s[i] in '+-':
                    mantissa = s[:i]
                    exponent = s[i:]
                    # Add decimal point after first digit
                    if len(mantissa) > 1:
                        mantissa = mantissa[0] + '.' + mantissa[1:]
                    return float(mantissa + 'e' + exponent)
        
        # Try regular float conversion
        try:
            return float(s)
        except:
            return 0.0
    
    def _parse_tle(self, tle_text: str) -> List[Dict]:
        """Parse TLE text into structured format."""
        lines = tle_text.strip().split('\n')
        satellites = []
        
        # TLE format: name line, line 1, line 2
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break
                
            try:
                name = lines[i].strip()
                line1 = lines[i + 1].strip()
                line2 = lines[i + 2].strip()
                
                # Extract key information
                if line1.startswith('1') and line2.startswith('2'):
                    satellite = {
                        'name': name,
                        'norad_id': int(line1[2:7]),
                        'classification': line1[7],
                        'launch_year': int(line1[9:11]),
                        'launch_number': int(line1[11:14]),
                        'launch_piece': line1[14:17].strip(),
                        'epoch_year': int(line1[18:20]),
                        'epoch_day': float(line1[20:32]),
                        'mean_motion_derivative': self._parse_float(line1[33:43]),
                        'mean_motion_sec_derivative': self._parse_float_exp(line1[44:52]),
                        'bstar': self._parse_float_exp(line1[53:61]),
                        'element_number': int(line1[64:68]),
                        'inclination': float(line2[8:16]),
                        'raan': float(line2[17:25]),
                        'eccentricity': float('0.' + line2[26:33]),
                        'arg_perigee': float(line2[34:42]),
                        'mean_anomaly': float(line2[43:51]),
                        'mean_motion': float(line2[52:63]),
                        'revolution_number': int(line2[63:68]),
                        'tle_line1': line1,
                        'tle_line2': line2,
                        'fetch_time': datetime.now().isoformat()
                    }
                    satellites.append(satellite)
                    
            except (ValueError, IndexError) as e:
                print(f"Error parsing TLE for {name if 'name' in locals() else 'unknown'}: {e}")
                continue
                
        return satellites
    
    def fetch_tle_data(self, source_name: str, force_refresh: bool = False) -> List[Dict]:
        """Fetch TLE data from a specific source."""
        cache_path = self._get_cache_path(source_name)
        
        # Check cache first
        if not force_refresh and self._is_cache_valid(cache_path):
            print(f"Loading {source_name} from cache...")
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        # Fetch fresh data
        print(f"Fetching fresh TLE data for {source_name}...")
        url = self.sources.get(source_name)
        
        if not url:
            raise ValueError(f"Unknown source: {source_name}")
        
        try:
            response = self.session.get(
                url, 
                timeout=API_CONFIG["timeout_seconds"],
                headers={'User-Agent': 'Orbital-Collision-Predictor/1.0'},
                verify=True
            )
            response.raise_for_status()
            
            # Parse TLE data
            satellites = self._parse_tle(response.text)
            
            # Cache the results
            with open(cache_path, 'w') as f:
                json.dump(satellites, f, indent=2)
                
            print(f"Fetched {len(satellites)} satellites from {source_name}")
            
            # Rate limiting
            time.sleep(60 / API_CONFIG["requests_per_minute"])
            
            return satellites
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {source_name}: {e}")
            
            # Try to return cached data if available
            if cache_path.exists():
                print("Falling back to cached data...")
                with open(cache_path, 'r') as f:
                    return json.load(f)
            else:
                return []
    
    def fetch_all_sources(self, sources: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch TLE data from multiple sources and combine."""
        if sources is None:
            sources = ["active_satellites", "space_stations", "starlink", "debris"]
        
        all_satellites = []
        
        for source in tqdm(sources, desc="Fetching TLE sources"):
            satellites = self.fetch_tle_data(source)
            for sat in satellites:
                sat['source'] = source
            all_satellites.extend(satellites)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_satellites)
        
        # Remove duplicates based on NORAD ID
        df = df.drop_duplicates(subset=['norad_id'], keep='first')
        
        print(f"\nTotal unique satellites: {len(df)}")
        print(f"Sources: {df['source'].value_counts().to_dict()}")
        
        return df
    
    def get_satellite_by_id(self, norad_id: int) -> Optional[Dict]:
        """Get a specific satellite by NORAD ID."""
        # Check all cached sources
        for source_name in self.sources:
            cache_path = self._get_cache_path(source_name)
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    satellites = json.load(f)
                    for sat in satellites:
                        if sat['norad_id'] == norad_id:
                            return sat
        
        return None


def demo_tle_fetcher():
    """Demo function to test TLE fetching."""
    fetcher = TLEFetcher()
    
    # Fetch space stations
    stations = fetcher.fetch_tle_data("space_stations")
    print(f"\nSpace Stations: {len(stations)}")
    for station in stations[:3]:
        print(f"  - {station['name']} (NORAD ID: {station['norad_id']})")
    
    # Fetch all and create DataFrame
    df = fetcher.fetch_all_sources(["space_stations", "starlink"])
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


if __name__ == "__main__":
    demo_tle_fetcher()
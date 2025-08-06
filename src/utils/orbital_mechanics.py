"""""Orbital mechanics calculations using SGP4."""

import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
from sgp4.api import Satrec, WGS72
from sgp4.api import jday
import pandas as pd


class OrbitalMechanics:
    """Handles orbital propagation and calculations."""
    
    def __init__(self):
        self.earth_radius_km = 6371.0
        self.mu = 398600.4418  # Earth's gravitational parameter (km³/s²)
        
    def tle_to_satrec(self, tle_line1: str, tle_line2: str) -> Satrec:
        """Convert TLE lines to Satrec object for propagation."""
        return Satrec.twoline2rv(tle_line1, tle_line2, WGS72)
    
    def propagate_orbit(
        self, 
        satrec: Satrec, 
        start_time: datetime, 
        duration_hours: float, 
        time_step_minutes: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """Propagate orbit from start time for given duration."""
        
        # Generate time points
        num_points = int(duration_hours * 60 / time_step_minutes)
        times = [start_time + timedelta(minutes=i * time_step_minutes) 
                for i in range(num_points)]
        
        # Arrays to store results
        positions = np.zeros((num_points, 3))
        velocities = np.zeros((num_points, 3))
        altitudes = np.zeros(num_points)
        
        for i, t in enumerate(times):
            # Convert to Julian date
            jd, fr = jday(t.year, t.month, t.day, 
                         t.hour, t.minute, t.second + t.microsecond/1e6)
            
            # Propagate
            e, r, v = satrec.sgp4(jd, fr)
            
            if e == 0:  # Success
                positions[i] = r
                velocities[i] = v
                altitudes[i] = np.linalg.norm(r) - self.earth_radius_km
            else:
                # Handle propagation errors
                positions[i] = np.nan
                velocities[i] = np.nan
                altitudes[i] = np.nan
        
        return {
            'times': times,
            'positions': positions,
            'velocities': velocities,
            'altitudes': altitudes
        }
    
    def calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate distance between two position vectors."""
        return np.linalg.norm(pos1 - pos2)
    
    def find_close_approaches(
        self,
        sat1_data: Dict,
        sat2_data: Dict,
        threshold_km: float = 10.0
    ) -> List[Dict]:
        """Find times when two satellites are within threshold distance."""
        
        close_approaches = []
        times = sat1_data['times']
        
        for i in range(len(times)):
            if np.any(np.isnan(sat1_data['positions'][i])) or \
               np.any(np.isnan(sat2_data['positions'][i])):
                continue
                
            distance = self.calculate_distance(
                sat1_data['positions'][i],
                sat2_data['positions'][i]
            )
            
            if distance < threshold_km:
                close_approaches.append({
                    'time': times[i],
                    'distance_km': distance,
                    'sat1_position': sat1_data['positions'][i].tolist(),
                    'sat2_position': sat2_data['positions'][i].tolist(),
                    'sat1_altitude': sat1_data['altitudes'][i],
                    'sat2_altitude': sat2_data['altitudes'][i]
                })
        
        return close_approaches
    
    def orbital_elements_from_state(self, r: np.ndarray, v: np.ndarray) -> Dict:
        """Calculate classical orbital elements from position and velocity."""
        
        # Magnitudes
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        
        # Angular momentum
        h = np.cross(r, v)
        h_mag = np.linalg.norm(h)
        
        # Node vector
        n = np.cross([0, 0, 1], h)
        n_mag = np.linalg.norm(n)
        
        # Eccentricity
        e_vec = ((v_mag**2 - self.mu/r_mag) * r - np.dot(r, v) * v) / self.mu
        e = np.linalg.norm(e_vec)
        
        # Semi-major axis
        energy = v_mag**2 / 2 - self.mu / r_mag
        a = -self.mu / (2 * energy) if abs(energy) > 1e-10 else float('inf')
        
        # Inclination
        i = np.arccos(h[2] / h_mag)
        
        # Right ascension of ascending node
        if n_mag > 0:
            raan = np.arccos(n[0] / n_mag)
            if n[1] < 0:
                raan = 2 * np.pi - raan
        else:
            raan = 0
        
        # Argument of periapsis
        if n_mag > 0 and e > 0:
            w = np.arccos(np.dot(n, e_vec) / (n_mag * e))
            if e_vec[2] < 0:
                w = 2 * np.pi - w
        else:
            w = 0
        
        # True anomaly
        if e > 0:
            nu = np.arccos(np.dot(e_vec, r) / (e * r_mag))
            if np.dot(r, v) < 0:
                nu = 2 * np.pi - nu
        else:
            nu = 0
        
        # Period
        if a > 0:
            period_seconds = 2 * np.pi * np.sqrt(a**3 / self.mu)
        else:
            period_seconds = None
        
        return {
            'semi_major_axis_km': a,
            'eccentricity': e,
            'inclination_deg': np.degrees(i),
            'raan_deg': np.degrees(raan),
            'arg_perigee_deg': np.degrees(w),
            'true_anomaly_deg': np.degrees(nu),
            'period_minutes': period_seconds / 60 if period_seconds else None
        }
    
    def predict_collision_probability(
        self,
        sat1_data: Dict,
        sat2_data: Dict,
        position_uncertainty_km: float = 1.0,
        num_samples: int = 1000
    ) -> Dict:
        """Estimate collision probability using Monte Carlo simulation."""
        
        times = sat1_data['times']
        probabilities = []
        min_distances = []
        
        for i in range(len(times)):
            if np.any(np.isnan(sat1_data['positions'][i])) or \
               np.any(np.isnan(sat2_data['positions'][i])):
                probabilities.append(0)
                min_distances.append(float('inf'))
                continue
            
            # Monte Carlo simulation
            collisions = 0
            distances = []
            
            for _ in range(num_samples):
                # Add uncertainty to positions
                pos1_sample = sat1_data['positions'][i] + \
                             np.random.normal(0, position_uncertainty_km, 3)
                pos2_sample = sat2_data['positions'][i] + \
                             np.random.normal(0, position_uncertainty_km, 3)
                
                distance = self.calculate_distance(pos1_sample, pos2_sample)
                distances.append(distance)
                
                # Consider collision if within combined object radius (simplified)
                if distance < 0.01:  # 10 meters
                    collisions += 1
            
            probability = collisions / num_samples
            probabilities.append(probability)
            min_distances.append(min(distances))
        
        # Find time of closest approach
        min_idx = np.argmin(min_distances)
        
        return {
            'times': times,
            'probabilities': probabilities,
            'min_distances': min_distances,
            'max_probability': max(probabilities),
            'closest_approach': {
                'time': times[min_idx],
                'distance_km': min_distances[min_idx],
                'probability': probabilities[min_idx]
            }
        }


def demo_orbital_mechanics():
    """Demo orbital mechanics calculations."""
    om = OrbitalMechanics()
    
    # Example TLE for ISS
    tle_line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  29565-3 0  9993"
    tle_line2 = "2 25544  51.6416 339.6916 0002719  55.9686  85.6604 15.50381554428603"
    
    # Create satellite record
    satrec = om.tle_to_satrec(tle_line1, tle_line2)
    
    # Propagate for 2 hours
    start_time = datetime.utcnow()
    orbit_data = om.propagate_orbit(satrec, start_time, 2.0)
    
    print(f"Propagated {len(orbit_data['times'])} positions")
    print(f"Starting altitude: {orbit_data['altitudes'][0]:.2f} km")
    print(f"Ending altitude: {orbit_data['altitudes'][-1]:.2f} km")
    
    # Calculate orbital elements
    if not np.any(np.isnan(orbit_data['positions'][0])):
        elements = om.orbital_elements_from_state(
            orbit_data['positions'][0],
            orbit_data['velocities'][0]
        )
        print(f"\nOrbital Elements:")
        for key, value in elements.items():
            if value is not None:
                print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    demo_orbital_mechanics()
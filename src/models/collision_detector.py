"""Collision detection and probability estimation system."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings

from ..utils.config import MODEL_CONFIG
from ..utils.orbital_mechanics import OrbitalMechanics
from .orbital_predictor import OrbitalPredictor


class CollisionDetector:
    """Detects potential collisions between space objects."""
    
    def __init__(self, predictor: Optional[OrbitalPredictor] = None):
        self.config = MODEL_CONFIG["collision_detection"]
        self.om = OrbitalMechanics()
        self.predictor = predictor or OrbitalPredictor()
        
    def screen_satellite_pair(
        self,
        sat1_tle: Tuple[str, str],
        sat2_tle: Tuple[str, str],
        sat1_info: Dict,
        sat2_info: Dict,
        start_time: datetime,
        duration_hours: float = 72
    ) -> Optional[Dict]:
        """Screen a satellite pair for potential collisions."""
        
        try:
            # Create satellite records
            satrec1 = self.om.tle_to_satrec(sat1_tle[0], sat1_tle[1])
            satrec2 = self.om.tle_to_satrec(sat2_tle[0], sat2_tle[1])
            
            # Propagate orbits
            orbit1 = self.om.propagate_orbit(
                satrec1, start_time, duration_hours,
                self.config["time_step_minutes"]
            )
            orbit2 = self.om.propagate_orbit(
                satrec2, start_time, duration_hours,
                self.config["time_step_minutes"]
            )
            
            # Find close approaches
            close_approaches = self.om.find_close_approaches(
                orbit1, orbit2,
                self.config["min_distance_km"]
            )
            
            if not close_approaches:
                return None
            
            # Calculate collision probability
            prob_results = self.om.predict_collision_probability(
                orbit1, orbit2,
                position_uncertainty_km=1.0,
                num_samples=self.config["monte_carlo_samples"]
            )
            
            # Find closest approach
            closest = prob_results['closest_approach']
            
            if closest['probability'] < self.config["probability_threshold"]:
                return None
            
            return {
                'sat1_name': sat1_info['name'],
                'sat1_norad_id': sat1_info['norad_id'],
                'sat2_name': sat2_info['name'],
                'sat2_norad_id': sat2_info['norad_id'],
                'tca_time': closest['time'],  # Time of Closest Approach
                'tca_distance_km': closest['distance_km'],
                'collision_probability': closest['probability'],
                'close_approaches': close_approaches,
                'probability_history': {
                    'times': prob_results['times'],
                    'probabilities': prob_results['probabilities'],
                    'distances': prob_results['min_distances']
                }
            }
            
        except Exception as e:
            warnings.warn(f"Error screening {sat1_info['name']} vs {sat2_info['name']}: {e}")
            return None
    
    def batch_collision_screening(
        self,
        satellites_df: pd.DataFrame,
        start_time: Optional[datetime] = None,
        duration_hours: float = 72,
        max_workers: int = 4,
        priority_satellites: Optional[List[int]] = None
    ) -> List[Dict]:
        """Screen multiple satellites for potential collisions."""
        
        if start_time is None:
            start_time = datetime.utcnow()
        
        # Filter satellites if priority list provided
        if priority_satellites:
            df = satellites_df[satellites_df['norad_id'].isin(priority_satellites)]
        else:
            df = satellites_df
        
        # Generate unique pairs
        satellite_pairs = []
        satellites = df.to_dict('records')
        
        for i in range(len(satellites)):
            for j in range(i + 1, len(satellites)):
                sat1 = satellites[i]
                sat2 = satellites[j]
                
                # Skip if same satellite
                if sat1['norad_id'] == sat2['norad_id']:
                    continue
                
                satellite_pairs.append((sat1, sat2))
        
        print(f"Screening {len(satellite_pairs)} satellite pairs...")
        
        # Parallel processing
        collision_risks = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for sat1, sat2 in satellite_pairs:
                future = executor.submit(
                    self.screen_satellite_pair,
                    (sat1['tle_line1'], sat1['tle_line2']),
                    (sat2['tle_line1'], sat2['tle_line2']),
                    sat1,
                    sat2,
                    start_time,
                    duration_hours
                )
                futures.append(future)
            
            # Process results
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    collision_risks.append(result)
        
        # Sort by collision probability
        collision_risks.sort(key=lambda x: x['collision_probability'], reverse=True)
        
        return collision_risks
    
    def analyze_constellation_risk(
        self,
        satellites_df: pd.DataFrame,
        constellation_name: str = "starlink"
    ) -> Dict:
        """Analyze collision risk for a specific constellation."""
        
        # Filter constellation satellites
        constellation_df = satellites_df[
            satellites_df['source'].str.contains(constellation_name, case=False)
        ]
        
        # Get other satellites
        other_df = satellites_df[
            ~satellites_df['source'].str.contains(constellation_name, case=False)
        ]
        
        print(f"Analyzing {len(constellation_df)} {constellation_name} satellites "
              f"against {len(other_df)} other objects...")
        
        # Screen for collisions
        collision_risks = []
        
        for _, const_sat in constellation_df.iterrows():
            for _, other_sat in other_df.iterrows():
                result = self.screen_satellite_pair(
                    (const_sat['tle_line1'], const_sat['tle_line2']),
                    (other_sat['tle_line1'], other_sat['tle_line2']),
                    const_sat.to_dict(),
                    other_sat.to_dict(),
                    datetime.utcnow(),
                    duration_hours=24  # Shorter for constellation analysis
                )
                
                if result:
                    collision_risks.append(result)
        
        # Analyze results
        risk_summary = {
            'constellation': constellation_name,
            'satellites_analyzed': len(constellation_df),
            'total_risks': len(collision_risks),
            'high_risk_events': sum(1 for r in collision_risks 
                                  if r['collision_probability'] > 0.001),
            'average_min_distance': np.mean([r['tca_distance_km'] 
                                           for r in collision_risks]) if collision_risks else 0,
            'top_risks': collision_risks[:10]  # Top 10 risks
        }
        
        return risk_summary
    
    def predict_future_conjunctions(
        self,
        sat1_tle: Tuple[str, str],
        sat2_tle: Tuple[str, str],
        sat1_info: Dict,
        sat2_info: Dict,
        prediction_days: int = 7
    ) -> Dict:
        """Use ML to predict future conjunctions beyond SGP4 accuracy."""
        
        if self.predictor.model is None:
            raise ValueError("Predictor model not loaded")
        
        # Get initial orbital data
        satrec1 = self.om.tle_to_satrec(sat1_tle[0], sat1_tle[1])
        satrec2 = self.om.tle_to_satrec(sat2_tle[0], sat2_tle[1])
        
        start_time = datetime.utcnow()
        
        # Get recent orbital history (24 hours)
        orbit1_history = self.om.propagate_orbit(satrec1, start_time - timedelta(hours=24), 24)
        orbit2_history = self.om.propagate_orbit(satrec2, start_time - timedelta(hours=24), 24)
        
        # Prepare data for ML prediction
        initial_positions1 = orbit1_history['positions'][-24:]  # Last 24 points
        initial_velocities1 = orbit1_history['velocities'][-24:]
        
        initial_positions2 = orbit2_history['positions'][-24:]
        initial_velocities2 = orbit2_history['velocities'][-24:]
        
        # Predict future trajectories
        prediction_steps = prediction_days * 24 * 60 // int(self.config["time_step_minutes"])
        
        predictions1 = self.predictor.predict_with_uncertainty(
            initial_positions1,
            initial_velocities1,
            prediction_steps,
            num_samples=50
        )
        
        predictions2 = self.predictor.predict_with_uncertainty(
            initial_positions2,
            initial_velocities2,
            prediction_steps,
            num_samples=50
        )
        
        # Analyze predicted conjunctions
        min_distances = []
        collision_probabilities = []
        
        for i in range(len(predictions1['mean'])):
            distance = self.om.calculate_distance(
                predictions1['mean'][i],
                predictions2['mean'][i]
            )
            min_distances.append(distance)
            
            # Estimate collision probability based on uncertainty
            uncertainty = np.sqrt(
                np.sum(predictions1['std'][i]**2) + 
                np.sum(predictions2['std'][i]**2)
            )
            
            # Simplified probability calculation
            if distance < self.config["min_distance_km"]:
                prob = np.exp(-distance**2 / (2 * uncertainty**2))
            else:
                prob = 0
            
            collision_probabilities.append(prob)
        
        # Find closest predicted approach
        min_idx = np.argmin(min_distances)
        
        return {
            'sat1_name': sat1_info['name'],
            'sat2_name': sat2_info['name'],
            'prediction_start': start_time,
            'prediction_end': start_time + timedelta(days=prediction_days),
            'ml_predicted_tca': start_time + timedelta(
                minutes=min_idx * self.config["time_step_minutes"]
            ),
            'ml_predicted_distance': min_distances[min_idx],
            'ml_collision_probability': max(collision_probabilities),
            'prediction_uncertainty': {
                'position_std_km': float(np.mean([
                    predictions1['std'][min_idx],
                    predictions2['std'][min_idx]
                ])),
            }
        }
    
    def generate_avoidance_recommendations(
        self,
        collision_risk: Dict,
        delta_v_budget_m_s: float = 10.0
    ) -> Dict:
        """Generate collision avoidance maneuver recommendations."""
        
        tca_time = collision_risk['tca_time']
        distance = collision_risk['tca_distance_km']
        probability = collision_risk['collision_probability']
        
        # Calculate required separation increase
        target_distance = self.config["min_distance_km"] * 2
        required_separation = target_distance - distance
        
        # Estimate delta-v requirement (simplified)
        # In reality, this would use optimal control theory
        maneuver_lead_time_hours = 24
        orbital_velocity_km_s = 7.5  # Approximate for LEO
        
        delta_v_required = (required_separation / orbital_velocity_km_s / 
                          (maneuver_lead_time_hours * 3600)) * 1000  # m/s
        
        recommendations = {
            'risk_level': 'HIGH' if probability > 0.01 else 'MEDIUM',
            'recommended_action': 'MANEUVER' if delta_v_required < delta_v_budget_m_s else 'MONITOR',
            'maneuver_details': {
                'execute_time': tca_time - timedelta(hours=maneuver_lead_time_hours),
                'delta_v_m_s': delta_v_required,
                'direction': 'RADIAL_OUT',  # Simplified
                'expected_miss_distance_km': target_distance
            },
            'alternative_options': [
                {
                    'option': 'EARLY_MANEUVER',
                    'execute_time': tca_time - timedelta(hours=48),
                    'delta_v_m_s': delta_v_required / 2
                },
                {
                    'option': 'LATE_MANEUVER', 
                    'execute_time': tca_time - timedelta(hours=12),
                    'delta_v_m_s': delta_v_required * 2
                }
            ]
        }
        
        return recommendations


def demo_collision_detector():
    """Demo collision detection system."""
    
    # Import TLE fetcher
    from ..data.tle_fetcher import TLEFetcher
    
    print("Fetching satellite data...")
    fetcher = TLEFetcher()
    
    # Get a small subset for demo
    stations = fetcher.fetch_tle_data("space_stations")
    starlink = fetcher.fetch_tle_data("starlink")
    
    # Combine into DataFrame
    satellites_df = pd.DataFrame(stations[:5] + starlink[:10])
    
    print(f"\nAnalyzing {len(satellites_df)} satellites...")
    
    # Initialize detector
    detector = CollisionDetector()
    
    # Screen for collisions
    collision_risks = detector.batch_collision_screening(
        satellites_df,
        duration_hours=24,  # Just 24 hours for demo
        max_workers=2
    )
    
    print(f"\nFound {len(collision_risks)} potential collision risks")
    
    if collision_risks:
        # Show top risk
        top_risk = collision_risks[0]
        print(f"\nHighest risk collision:")
        print(f"  {top_risk['sat1_name']} vs {top_risk['sat2_name']}")
        print(f"  TCA: {top_risk['tca_time']}")
        print(f"  Distance: {top_risk['tca_distance_km']:.2f} km")
        print(f"  Probability: {top_risk['collision_probability']:.2e}")
        
        # Generate recommendations
        recommendations = detector.generate_avoidance_recommendations(top_risk)
        print(f"\nRecommendations:")
        print(f"  Risk Level: {recommendations['risk_level']}")
        print(f"  Action: {recommendations['recommended_action']}")


if __name__ == "__main__":
    demo_collision_detector()
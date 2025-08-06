"""Quick start demo for the Orbital Collision Predictor."""

import sys
sys.path.append('..')

from datetime import datetime
import pandas as pd

from src.data.tle_fetcher import TLEFetcher
from src.models.collision_detector import CollisionDetector
from src.visualization.orbit_visualizer import OrbitVisualizer
from src.utils.orbital_mechanics import OrbitalMechanics


def main():
    """Run a quick demonstration of the system."""
    
    print("=" * 60)
    print("🛰️  Orbital Collision Prediction System - Quick Demo")
    print("=" * 60)
    
    # Step 1: Fetch real satellite data
    print("\n1. Fetching real satellite data from NORAD...")
    fetcher = TLEFetcher()
    
    # Get space stations and a sample of Starlink satellites
    stations = fetcher.fetch_tle_data("space_stations")
    starlink = fetcher.fetch_tle_data("starlink")
    
    # Combine into DataFrame (limit for demo)
    satellites_df = pd.DataFrame(stations + starlink[:20])
    print(f"   ✓ Loaded {len(satellites_df)} satellites")
    
    # Step 2: Show some interesting satellites
    print("\n2. Sample satellites in our database:")
    for _, sat in satellites_df.head(5).iterrows():
        print(f"   - {sat['name']} (NORAD ID: {sat['norad_id']})")
    
    # Step 3: Run collision detection
    print("\n3. Running collision detection algorithm...")
    detector = CollisionDetector()
    
    collision_risks = detector.batch_collision_screening(
        satellites_df,
        duration_hours=24,  # Next 24 hours
        max_workers=2
    )
    
    print(f"   ✓ Found {len(collision_risks)} potential collision risks")
    
    # Step 4: Display top risks
    if collision_risks:
        print("\n4. Top collision risks detected:")
        for i, risk in enumerate(collision_risks[:3]):
            print(f"\n   Risk #{i+1}:")
            print(f"   - Satellites: {risk['sat1_name']} vs {risk['sat2_name']}")
            print(f"   - Time of Closest Approach: {risk['tca_time'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"   - Minimum Distance: {risk['tca_distance_km']:.2f} km")
            print(f"   - Collision Probability: {risk['collision_probability']:.2e}")
            
            # Generate recommendations
            recommendations = detector.generate_avoidance_recommendations(risk)
            print(f"   - Recommended Action: {recommendations['recommended_action']}")
    else:
        print("\n4. No collision risks detected in the next 24 hours!")
    
    # Step 5: Visualize ISS orbit
    print("\n5. Generating ISS orbit visualization...")
    iss_data = satellites_df[satellites_df['name'].str.contains('ISS', case=False)]
    
    if not iss_data.empty:
        iss = iss_data.iloc[0]
        om = OrbitalMechanics()
        viz = OrbitVisualizer()
        
        # Propagate ISS orbit
        satrec = om.tle_to_satrec(iss['tle_line1'], iss['tle_line2'])
        orbit_data = om.propagate_orbit(satrec, datetime.utcnow(), 1.5)
        
        # Create visualization
        import plotly.graph_objects as go
        fig = go.Figure()
        
        # Add Earth
        fig.add_trace(viz.create_earth_sphere())
        
        # Add ISS orbit
        iss_traces = viz.plot_orbit_3d(orbit_data, iss['name'])
        for trace in iss_traces:
            fig.add_trace(trace)
        
        fig.update_layout(
            title='ISS Current Orbit',
            scene=dict(
                xaxis_title='X (km)',
                yaxis_title='Y (km)',
                zaxis_title='Z (km)',
                aspectmode='data'
            ),
            width=800,
            height=600
        )
        
        # Save visualization
        output_file = 'iss_orbit_demo.html'
        fig.write_html(output_file)
        print(f"   ✓ Visualization saved to {output_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Demo complete! Key capabilities demonstrated:")
    print("- Real-time satellite data fetching from NORAD")
    print("- Collision risk detection with probability estimation")
    print("- 3D orbit visualization")
    print("- Automated collision avoidance recommendations")
    print("\nTo see the full dashboard, run:")
    print("  streamlit run src/visualization/dashboard.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
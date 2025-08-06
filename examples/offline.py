"""Demo with sample data for offline testing."""

import sys
sys.path.append('..')

from datetime import datetime
import pandas as pd
import numpy as np

from src.utils.orbital_mechanics import OrbitalMechanics
from src.visualization.orbit_visualizer import OrbitVisualizer


def create_sample_data():
    """Create sample satellite data for demo purposes."""
    
    # Sample TLE data for ISS and a few other satellites
    sample_satellites = [
        {
            'name': 'ISS (ZARYA)',
            'norad_id': 25544,
            'tle_line1': '1 25544U 98067A   24001.50000000  .00016717  00000-0  29565-3 0  9993',
            'tle_line2': '2 25544  51.6416 339.6916 0002719  55.9686  85.6604 15.50381554428603',
            'source': 'space_stations',
            'inclination': 51.6416,
            'eccentricity': 0.0002719,
            'mean_motion': 15.50381554
        },
        {
            'name': 'STARLINK-1234',
            'norad_id': 45678,
            'tle_line1': '1 45678U 20019A   24001.50000000  .00001000  00000-0  10000-3 0  9990',
            'tle_line2': '2 45678  53.0000 100.0000 0001000  90.0000 270.0000 15.00000000100000',
            'source': 'starlink',
            'inclination': 53.0,
            'eccentricity': 0.0001,
            'mean_motion': 15.0
        },
        {
            'name': 'COSMOS 2251 DEB',
            'norad_id': 34567,
            'tle_line1': '1 34567U 09013A   24001.50000000  .00000500  00000-0  50000-4 0  9995',
            'tle_line2': '2 34567  74.0000 200.0000 0010000 180.0000 180.0000 14.50000000200000',
            'source': 'debris',
            'inclination': 74.0,
            'eccentricity': 0.001,
            'mean_motion': 14.5
        }
    ]
    
    return pd.DataFrame(sample_satellites)


def main():
    """Run demo with sample data."""
    
    print("=" * 60)
    print("🛰️  Orbital Collision Prediction System - Offline Demo")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Using sample satellite data...")
    satellites_df = create_sample_data()
    print(f"   ✓ Loaded {len(satellites_df)} sample satellites")
    
    # Show satellites
    print("\n2. Sample satellites in our database:")
    for _, sat in satellites_df.iterrows():
        print(f"   - {sat['name']} (NORAD ID: {sat['norad_id']})")
    
    # Visualize ISS orbit
    print("\n3. Generating ISS orbit visualization...")
    iss = satellites_df[satellites_df['name'].str.contains('ISS', case=False)].iloc[0]
    
    om = OrbitalMechanics()
    viz = OrbitVisualizer()
    
    # Propagate ISS orbit
    satrec = om.tle_to_satrec(iss['tle_line1'], iss['tle_line2'])
    orbit_data = om.propagate_orbit(satrec, datetime.utcnow(), 1.5)
    
    print(f"   ✓ Propagated {len(orbit_data['times'])} orbital positions")
    print(f"   ✓ Current altitude: {orbit_data['altitudes'][0]:.1f} km")
    
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
        title='ISS Current Orbit - Demo Visualization',
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
    output_file = 'iss_orbit_demo_offline.html'
    fig.write_html(output_file)
    print(f"\n   ✓ Visualization saved to {output_file}")
    print("   ✓ Open the HTML file in your browser to see the 3D orbit!")
    
    # Summary
    print("\n" + "=" * 60)
    print("Demo complete! This demonstrated:")
    print("- Orbital mechanics calculations")
    print("- 3D orbit visualization")
    print("- Working with TLE data")
    print("\nNote: This used sample data. For real satellite tracking,")
    print("ensure you have internet access to fetch live TLE data.")
    print("=" * 60)


if __name__ == "__main__":
    main()
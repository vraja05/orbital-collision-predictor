"""3D visualization of orbits and collision risks."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

from ..utils.config import VIZ_CONFIG
from ..utils.orbital_mechanics import OrbitalMechanics


class OrbitVisualizer:
    """Creates interactive 3D visualizations of orbits."""
    
    def __init__(self):
        self.earth_radius = VIZ_CONFIG["earth_radius_km"]
        self.colors = VIZ_CONFIG["color_scheme"]
        self.om = OrbitalMechanics()
        
    def create_earth_sphere(self) -> go.Surface:
        """Create a 3D sphere representing Earth."""
        
        # Create sphere coordinates
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        
        x = self.earth_radius * np.outer(np.cos(u), np.sin(v))
        y = self.earth_radius * np.outer(np.sin(u), np.sin(v))
        z = self.earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        earth = go.Surface(
            x=x, y=y, z=z,
            colorscale='Blues',
            showscale=False,
            opacity=0.7,
            name='Earth',
            hovertemplate='Earth<br>Radius: 6371 km<extra></extra>'
        )
        
        return earth
    
    def plot_orbit_3d(
        self,
        orbit_data: Dict,
        satellite_name: str,
        color: Optional[str] = None,
        show_trajectory: bool = True
    ) -> List[go.Scatter3d]:
        """Plot a single orbit in 3D."""
        
        traces = []
        positions = orbit_data['positions']
        
        # Filter out NaN values
        valid_mask = ~np.any(np.isnan(positions), axis=1)
        valid_positions = positions[valid_mask]
        
        if len(valid_positions) == 0:
            return traces
        
        if color is None:
            color = self.colors['active_satellite']
        
        # Orbit trajectory
        if show_trajectory:
            orbit_trace = go.Scatter3d(
                x=valid_positions[:, 0],
                y=valid_positions[:, 1],
                z=valid_positions[:, 2],
                mode='lines',
                name=f'{satellite_name} orbit',
                line=dict(color=color, width=2),
                hovertemplate='%{text}<extra></extra>',
                text=[f'{satellite_name}<br>Time: {t}<br>Alt: {alt:.1f} km'
                      for t, alt in zip(np.array(orbit_data['times'])[valid_mask], 
                                      orbit_data['altitudes'][valid_mask])]
            )
            traces.append(orbit_trace)
        
        # Current position (last point)
        current_pos = go.Scatter3d(
            x=[valid_positions[-1, 0]],
            y=[valid_positions[-1, 1]],
            z=[valid_positions[-1, 2]],
            mode='markers',
            name=f'{satellite_name} position',
            marker=dict(
                color=color,
                size=8,
                symbol='diamond'
            ),
            hovertemplate=f'{satellite_name}<br>Current Position<extra></extra>'
        )
        traces.append(current_pos)
        
        return traces
    
    def visualize_collision_risk(
        self,
        collision_risk: Dict,
        sat1_orbit: Dict,
        sat2_orbit: Dict
    ) -> go.Figure:
        """Visualize a potential collision scenario."""
        
        fig = go.Figure()
        
        # Add Earth
        fig.add_trace(self.create_earth_sphere())
        
        # Add satellite orbits
        sat1_traces = self.plot_orbit_3d(
            sat1_orbit,
            collision_risk['sat1_name'],
            color=self.colors['active_satellite']
        )
        for trace in sat1_traces:
            fig.add_trace(trace)
        
        sat2_traces = self.plot_orbit_3d(
            sat2_orbit,
            collision_risk['sat2_name'],
            color=self.colors['debris'] if 'debris' in collision_risk['sat2_name'].lower() 
            else self.colors['active_satellite']
        )
        for trace in sat2_traces:
            fig.add_trace(trace)
        
        # Add close approach points
        for approach in collision_risk['close_approaches']:
            # Connection line at closest approach
            fig.add_trace(go.Scatter3d(
                x=[approach['sat1_position'][0], approach['sat2_position'][0]],
                y=[approach['sat1_position'][1], approach['sat2_position'][1]],
                z=[approach['sat1_position'][2], approach['sat2_position'][2]],
                mode='lines+markers',
                name=f'Close approach {approach["distance_km"]:.1f} km',
                line=dict(color=self.colors['collision_risk'], width=4, dash='dash'),
                marker=dict(size=10, color=self.colors['collision_risk']),
                hovertemplate=f'Distance: {approach["distance_km"]:.1f} km<br>' +
                            f'Time: {approach["time"]}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Collision Risk: {collision_risk["sat1_name"]} vs {collision_risk["sat2_name"]}<br>' +
                     f'Min Distance: {collision_risk["tca_distance_km"]:.1f} km | ' +
                     f'Probability: {collision_risk["collision_probability"]:.2e}',
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis_title='X (km)',
                yaxis_title='Y (km)',
                zaxis_title='Z (km)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            width=1000,
            height=800
        )
        
        return fig
    
    def create_risk_dashboard(
        self,
        collision_risks: List[Dict],
        time_range_hours: int = 72
    ) -> go.Figure:
        """Create a dashboard showing multiple collision risks."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Collision Risk Timeline',
                'Risk by Satellite Type',
                'Distance Distribution',
                'Top 10 Highest Risks'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'histogram'}, {'type': 'bar'}]]
        )
        
        # 1. Risk Timeline
        times = []
        probabilities = []
        names = []
        
        for risk in collision_risks:
            times.append(risk['tca_time'])
            probabilities.append(risk['collision_probability'])
            names.append(f"{risk['sat1_name'][:20]} vs {risk['sat2_name'][:20]}")
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=probabilities,
                mode='markers',
                marker=dict(
                    size=10,
                    color=probabilities,
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Probability")
                ),
                text=names,
                hovertemplate='%{text}<br>Time: %{x}<br>Probability: %{y:.2e}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Risk by Satellite Type
        sat_types = {}
        for risk in collision_risks:
            for sat_name in [risk['sat1_name'], risk['sat2_name']]:
                sat_type = 'Debris' if 'DEB' in sat_name else \
                          'Station' if 'ISS' in sat_name or 'STATION' in sat_name else \
                          'Starlink' if 'STARLINK' in sat_name else \
                          'Other'
                sat_types[sat_type] = sat_types.get(sat_type, 0) + 1
        
        fig.add_trace(
            go.Bar(
                x=list(sat_types.keys()),
                y=list(sat_types.values()),
                marker_color=['red', 'blue', 'orange', 'green'][:len(sat_types)]
            ),
            row=1, col=2
        )
        
        # 3. Distance Distribution
        distances = [risk['tca_distance_km'] for risk in collision_risks]
        
        fig.add_trace(
            go.Histogram(
                x=distances,
                nbinsx=20,
                marker_color='lightblue',
                name='Distance Distribution'
            ),
            row=2, col=1
        )
        
        # 4. Top 10 Risks
        top_risks = sorted(collision_risks, 
                          key=lambda x: x['collision_probability'], 
                          reverse=True)[:10]
        
        risk_names = [f"{r['sat1_name'][:15]}..." for r in top_risks]
        risk_probs = [r['collision_probability'] for r in top_risks]
        
        fig.add_trace(
            go.Bar(
                y=risk_names,
                x=risk_probs,
                orientation='h',
                marker_color='red',
                text=[f"{p:.2e}" for p in risk_probs],
                textposition='outside'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Collision Probability", type="log", row=1, col=1)
        fig.update_xaxes(title_text="Satellite Type", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_xaxes(title_text="Distance (km)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Probability", row=2, col=2)
        
        fig.update_layout(
            title_text=f"Orbital Collision Risk Dashboard - {len(collision_risks)} Risks Detected",
            showlegend=False,
            height=800,
            width=1200
        )
        
        return fig
    
    def create_constellation_view(
        self,
        satellites_df: pd.DataFrame,
        constellation_name: str = "starlink",
        time_point: Optional[datetime] = None
    ) -> go.Figure:
        """Visualize an entire constellation."""
        
        if time_point is None:
            time_point = datetime.utcnow()
        
        fig = go.Figure()
        
        # Add Earth
        fig.add_trace(self.create_earth_sphere())
        
        # Filter constellation satellites
        const_sats = satellites_df[
            satellites_df['source'].str.contains(constellation_name, case=False)
        ].head(50)  # Limit for performance
        
        print(f"Visualizing {len(const_sats)} {constellation_name} satellites...")
        
        # Add each satellite
        for _, sat in const_sats.iterrows():
            try:
                # Create satrec and get current position
                satrec = self.om.tle_to_satrec(sat['tle_line1'], sat['tle_line2'])
                
                # Get position at time_point
                jd, fr = self.om.jday(
                    time_point.year, time_point.month, time_point.day,
                    time_point.hour, time_point.minute, time_point.second
                )
                
                e, r, v = satrec.sgp4(jd, fr)
                
                if e == 0:  # Success
                    fig.add_trace(go.Scatter3d(
                        x=[r[0]],
                        y=[r[1]],
                        z=[r[2]],
                        mode='markers',
                        name=sat['name'],
                        marker=dict(
                            size=4,
                            color=self.colors['starlink'],
                            symbol='circle'
                        ),
                        hovertemplate=f"{sat['name']}<br>NORAD: {sat['norad_id']}<extra></extra>"
                    ))
            except Exception as e:
                continue
        
        # Update layout
        fig.update_layout(
            title=f'{constellation_name.title()} Constellation View',
            scene=dict(
                xaxis_title='X (km)',
                yaxis_title='Y (km)',
                zaxis_title='Z (km)',
                aspectmode='data'
            ),
            showlegend=False,
            width=1000,
            height=800
        )
        
        return fig
    
    def animate_collision_scenario(
        self,
        collision_risk: Dict,
        sat1_tle: Tuple[str, str],
        sat2_tle: Tuple[str, str],
        hours_before_tca: float = 2.0,
        hours_after_tca: float = 2.0,
        time_step_minutes: float = 1.0
    ) -> go.Figure:
        """Create animated visualization of collision scenario."""
        
        tca_time = collision_risk['tca_time']
        start_time = tca_time - timedelta(hours=hours_before_tca)
        duration = hours_before_tca + hours_after_tca
        
        # Propagate orbits
        satrec1 = self.om.tle_to_satrec(sat1_tle[0], sat1_tle[1])
        satrec2 = self.om.tle_to_satrec(sat2_tle[0], sat2_tle[1])
        
        orbit1 = self.om.propagate_orbit(satrec1, start_time, duration, time_step_minutes)
        orbit2 = self.om.propagate_orbit(satrec2, start_time, duration, time_step_minutes)
        
        # Create figure with animation
        fig = go.Figure()
        
        # Add Earth
        fig.add_trace(self.create_earth_sphere())
        
        # Create frames for animation
        frames = []
        
        for i in range(len(orbit1['times'])):
            frame_data = []
            
            # Earth (static)
            frame_data.append(self.create_earth_sphere())
            
            # Satellite 1 trail and position
            trail_start = max(0, i - 20)
            frame_data.append(go.Scatter3d(
                x=orbit1['positions'][trail_start:i+1, 0],
                y=orbit1['positions'][trail_start:i+1, 1],
                z=orbit1['positions'][trail_start:i+1, 2],
                mode='lines',
                line=dict(color=self.colors['active_satellite'], width=2),
                name=collision_risk['sat1_name']
            ))
            
            # Satellite 2 trail and position  
            frame_data.append(go.Scatter3d(
                x=orbit2['positions'][trail_start:i+1, 0],
                y=orbit2['positions'][trail_start:i+1, 1],
                z=orbit2['positions'][trail_start:i+1, 2],
                mode='lines',
                line=dict(color=self.colors['debris'], width=2),
                name=collision_risk['sat2_name']
            ))
            
            # Current positions
            if not np.any(np.isnan(orbit1['positions'][i])) and \
               not np.any(np.isnan(orbit2['positions'][i])):
                
                # Distance line
                distance = self.om.calculate_distance(
                    orbit1['positions'][i],
                    orbit2['positions'][i]
                )
                
                frame_data.append(go.Scatter3d(
                    x=[orbit1['positions'][i, 0], orbit2['positions'][i, 0]],
                    y=[orbit1['positions'][i, 1], orbit2['positions'][i, 1]],
                    z=[orbit1['positions'][i, 2], orbit2['positions'][i, 2]],
                    mode='lines+markers',
                    line=dict(
                        color='red' if distance < 50 else 'yellow',
                        width=3,
                        dash='dot'
                    ),
                    marker=dict(size=8),
                    name=f'Distance: {distance:.1f} km'
                ))
            
            frames.append(go.Frame(
                data=frame_data,
                name=str(i),
                layout=go.Layout(
                    title=f"Time: {orbit1['times'][i].strftime('%Y-%m-%d %H:%M:%S')} UTC<br>" +
                          f"Distance: {distance:.1f} km"
                )
            ))
        
        # Set up animation
        fig.frames = frames
        
        # Add play/pause buttons
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'y': 0,
                'x': 0.1,
                'xanchor': 'right',
                'yanchor': 'bottom',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 50, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'active': 0,
                'steps': [{
                    'args': [[f.name], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': orbit1['times'][i].strftime('%H:%M'),
                    'method': 'animate'
                } for i, f in enumerate(fig.frames[::10])]  # Show every 10th frame in slider
            }]
        )
        
        # Set initial frame
        fig.add_traces(frames[0].data)
        
        fig.update_layout(
            title=f"Collision Scenario Animation: {collision_risk['sat1_name']} vs {collision_risk['sat2_name']}",
            scene=dict(
                xaxis_title='X (km)',
                yaxis_title='Y (km)', 
                zaxis_title='Z (km)',
                aspectmode='data'
            ),
            width=1000,
            height=800
        )
        
        return fig


def demo_orbit_visualizer():
    """Demo the orbit visualizer."""
    from ..data.tle_fetcher import TLEFetcher
    
    # Fetch some satellite data
    fetcher = TLEFetcher()
    stations = fetcher.fetch_tle_data("space_stations")
    
    if not stations:
        print("No satellite data available")
        return
    
    # Get ISS data
    iss = None
    for sat in stations:
        if 'ISS' in sat['name']:
            iss = sat
            break
    
    if not iss:
        print("ISS data not found")
        return
    
    # Create visualizer
    viz = OrbitVisualizer()
    om = OrbitalMechanics()
    
    # Propagate ISS orbit
    satrec = om.tle_to_satrec(iss['tle_line1'], iss['tle_line2'])
    orbit_data = om.propagate_orbit(satrec, datetime.utcnow(), 2.0)
    
    # Create 3D plot
    fig = go.Figure()
    fig.add_trace(viz.create_earth_sphere())
    
    iss_traces = viz.plot_orbit_3d(orbit_data, iss['name'])
    for trace in iss_traces:
        fig.add_trace(trace)
    
    fig.update_layout(
        title='ISS Orbit Visualization',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data'
        ),
        width=1000,
        height=800
    )
    
    # Save figure
    fig.write_html("iss_orbit.html")
    print("Visualization saved to iss_orbit.html")
    
    return fig


if __name__ == "__main__":
    demo_orbit_visualizer()
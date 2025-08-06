"""Streamlit dashboard for real-time collision monitoring."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

from src.data.tle_fetcher import TLEFetcher
from src.models.collision_detector import CollisionDetector
from src.models.orbital_predictor import OrbitalPredictor
from src.visualization.orbit_visualizer import OrbitVisualizer
from src.utils.config import VIZ_CONFIG
from src.utils.orbital_mechanics import OrbitalMechanics


# Page configuration
st.set_page_config(
    page_title="Orbital Collision Predictor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .risk-high { color: #ff4444; font-weight: bold; }
    .risk-medium { color: #ffaa00; font-weight: bold; }
    .risk-low { color: #44ff44; }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


class CollisionDashboard:
    """Streamlit dashboard for collision monitoring."""
    
    def __init__(self):
        self.fetcher = TLEFetcher()
        self.detector = CollisionDetector()
        self.visualizer = OrbitVisualizer()
        self.predictor = OrbitalPredictor()
        
        # Initialize session state
        if 'satellites_df' not in st.session_state:
            st.session_state.satellites_df = None
        if 'collision_risks' not in st.session_state:
            st.session_state.collision_risks = []
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
    
    def run(self):
        """Run the dashboard."""
        st.title("🛰️ Orbital Collision Prediction System")
        st.markdown("*Real-time tracking and ML-powered collision prediction*")
        
        # Sidebar
        with st.sidebar:
            st.header("Configuration")
            
            # Data sources
            st.subheader("Data Sources")
            sources = st.multiselect(
                "Select satellite groups:",
                ["active_satellites", "space_stations", "starlink", "debris", "oneweb"],
                default=["space_stations", "starlink"]
            )
            
            # Analysis parameters
            st.subheader("Analysis Parameters")
            duration_hours = st.slider(
                "Prediction horizon (hours):",
                min_value=6,
                max_value=168,
                value=72,
                step=6
            )
            
            min_distance = st.number_input(
                "Minimum distance threshold (km):",
                min_value=1.0,
                max_value=50.0,
                value=10.0,
                step=1.0
            )
            
            prob_threshold = st.number_input(
                "Probability threshold:",
                min_value=0.0001,
                max_value=0.1,
                value=0.01,
                format="%.4f"
            )
            
            # Update button
            if st.button("🔄 Update Analysis", type="primary"):
                self.update_analysis(sources, duration_hours, min_distance, prob_threshold)
            
            # Auto-refresh
            auto_refresh = st.checkbox("Auto-refresh (every 5 min)")
            
            if st.session_state.last_update:
                st.caption(f"Last update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        # Main content
        if st.session_state.satellites_df is None:
            st.info("👈 Click 'Update Analysis' to start tracking satellites")
            return
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "⚠️ Collision Risks", 
            "🌍 3D Visualization",
            "📈 Predictions",
            "📡 Satellite Data"
        ])
        
        with tab1:
            self.show_overview()
        
        with tab2:
            self.show_collision_risks()
        
        with tab3:
            self.show_3d_visualization()
        
        with tab4:
            self.show_predictions()
        
        with tab5:
            self.show_satellite_data()
        
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(300)  # 5 minutes
            st.rerun()
    
    def update_analysis(self, sources, duration_hours, min_distance, prob_threshold):
        """Update the collision analysis."""
        with st.spinner("Fetching satellite data..."):
            # Fetch TLE data
            satellites_df = self.fetcher.fetch_all_sources(sources)
            st.session_state.satellites_df = satellites_df
        
        with st.spinner("Analyzing collision risks..."):
            # Update detector config
            self.detector.config["min_distance_km"] = min_distance
            self.detector.config["probability_threshold"] = prob_threshold
            
            # Run collision screening
            collision_risks = self.detector.batch_collision_screening(
                satellites_df,
                duration_hours=duration_hours,
                max_workers=4
            )
            st.session_state.collision_risks = collision_risks
            st.session_state.last_update = datetime.utcnow()
        
        st.success(f"Analysis complete! Found {len(collision_risks)} potential collisions.")
    
    def show_overview(self):
        """Show overview metrics and statistics."""
        st.header("Mission Overview")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Satellites",
                len(st.session_state.satellites_df),
                delta=None
            )
        
        with col2:
            high_risk = sum(1 for r in st.session_state.collision_risks 
                          if r['collision_probability'] > 0.01)
            st.metric(
                "High Risk Events",
                high_risk,
                delta=None,
                delta_color="inverse"
            )
        
        with col3:
            if st.session_state.collision_risks:
                min_dist = min(r['tca_distance_km'] for r in st.session_state.collision_risks)
                st.metric(
                    "Closest Approach",
                    f"{min_dist:.1f} km",
                    delta=None
                )
            else:
                st.metric("Closest Approach", "N/A")
        
        with col4:
            if st.session_state.collision_risks:
                max_prob = max(r['collision_probability'] for r in st.session_state.collision_risks)
                st.metric(
                    "Max Probability",
                    f"{max_prob:.2e}",
                    delta=None
                )
            else:
                st.metric("Max Probability", "0")
        
        # Risk timeline
        st.subheader("Risk Timeline")
        if st.session_state.collision_risks:
            fig = self.create_risk_timeline()
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No collision risks detected")
        
        # Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Satellites by Type")
            sat_counts = st.session_state.satellites_df['source'].value_counts()
            fig = go.Figure(data=[
                go.Pie(labels=sat_counts.index, values=sat_counts.values)
            ])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Altitude Distribution")
            # Calculate mean motion to altitude (simplified)
            altitudes = 550 + (15.5 - st.session_state.satellites_df['mean_motion']) * 50
            fig = go.Figure(data=[
                go.Histogram(x=altitudes, nbinsx=30, name='Altitude Distribution')
            ])
            fig.update_layout(
                xaxis_title="Approximate Altitude (km)",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def show_collision_risks(self):
        """Show detailed collision risk information."""
        st.header("⚠️ Collision Risk Analysis")
        
        if not st.session_state.collision_risks:
            st.info("No collision risks detected within the specified parameters")
            return
        
        # Risk level filter
        risk_level = st.selectbox(
            "Filter by risk level:",
            ["All", "High (>1%)", "Medium (0.1-1%)", "Low (<0.1%)"]
        )
        
        # Filter risks
        filtered_risks = self.filter_risks_by_level(st.session_state.collision_risks, risk_level)
        
        st.write(f"Showing {len(filtered_risks)} collision risks")
        
        # Display risks
        for i, risk in enumerate(filtered_risks[:20]):  # Show top 20
            with st.expander(
                f"{risk['sat1_name']} ↔️ {risk['sat2_name']} - "
                f"P: {risk['collision_probability']:.2e} | "
                f"D: {risk['tca_distance_km']:.1f} km"
            ):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Collision Details:**")
                    st.write(f"- TCA: {risk['tca_time'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
                    st.write(f"- Minimum Distance: {risk['tca_distance_km']:.2f} km")
                    st.write(f"- Collision Probability: {risk['collision_probability']:.2e}")
                    st.write(f"- Satellite 1 NORAD ID: {risk['sat1_norad_id']}")
                    st.write(f"- Satellite 2 NORAD ID: {risk['sat2_norad_id']}")
                
                with col2:
                    # Generate avoidance recommendations
                    recommendations = self.detector.generate_avoidance_recommendations(risk)
                    st.write("**Recommendations:**")
                    st.write(f"- Risk Level: {recommendations['risk_level']}")
                    st.write(f"- Action: {recommendations['recommended_action']}")
                    if recommendations['recommended_action'] == 'MANEUVER':
                        st.write(f"- ΔV Required: {recommendations['maneuver_details']['delta_v_m_s']:.1f} m/s")
                        st.write(f"- Execute Time: {recommendations['maneuver_details']['execute_time'].strftime('%Y-%m-%d %H:%M')} UTC")
                
                # Show close approach history
                if st.checkbox(f"Show approach history #{i+1}", key=f"hist_{i}"):
                    self.plot_approach_history(risk)
    
    def show_3d_visualization(self):
        """Show 3D visualization of orbits and risks."""
        st.header("🌍 3D Orbit Visualization")
        
        viz_type = st.selectbox(
            "Visualization type:",
            ["Top Collision Risk", "Constellation View", "Custom Selection"]
        )
        
        if viz_type == "Top Collision Risk" and st.session_state.collision_risks:
            # Show the highest risk collision
            risk = st.session_state.collision_risks[0]
            
            # Get satellite TLEs
            sat1 = st.session_state.satellites_df[
                st.session_state.satellites_df['norad_id'] == risk['sat1_norad_id']
            ].iloc[0]
            sat2 = st.session_state.satellites_df[
                st.session_state.satellites_df['norad_id'] == risk['sat2_norad_id']
            ].iloc[0]
            
            # Propagate orbits
            om = OrbitalMechanics()
            
            satrec1 = om.tle_to_satrec(sat1['tle_line1'], sat1['tle_line2'])
            satrec2 = om.tle_to_satrec(sat2['tle_line1'], sat2['tle_line2'])
            
            start_time = risk['tca_time'] - timedelta(hours=1)
            orbit1 = om.propagate_orbit(satrec1, start_time, 2.0)
            orbit2 = om.propagate_orbit(satrec2, start_time, 2.0)
            
            # Create visualization
            fig = self.visualizer.visualize_collision_risk(risk, orbit1, orbit2)
            st.plotly_chart(fig, use_container_width=True)
            
            # Animation option
            if st.checkbox("Show animated scenario"):
                anim_fig = self.visualizer.animate_collision_scenario(
                    risk,
                    (sat1['tle_line1'], sat1['tle_line2']),
                    (sat2['tle_line1'], sat2['tle_line2']),
                    hours_before_tca=1.0,
                    hours_after_tca=1.0
                )
                st.plotly_chart(anim_fig, use_container_width=True)
        
        elif viz_type == "Constellation View":
            constellation = st.selectbox(
                "Select constellation:",
                ["starlink", "oneweb", "iridium"]
            )
            
            if constellation in st.session_state.satellites_df['source'].str.lower().values:
                fig = self.visualizer.create_constellation_view(
                    st.session_state.satellites_df,
                    constellation
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No {constellation} satellites in current data")
        
        elif viz_type == "Custom Selection":
            # Let user select specific satellites
            satellite_names = st.session_state.satellites_df['name'].unique()
            selected = st.multiselect(
                "Select satellites to visualize:",
                satellite_names,
                max_selections=5
            )
            
            if selected:
                self.visualize_selected_satellites(selected)
    
    def show_predictions(self):
        """Show ML predictions and analysis."""
        st.header("📈 Machine Learning Predictions")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Prediction Accuracy")
            
            # Check if model exists
            try:
                self.predictor.load_model()
                model_loaded = True
            except:
                model_loaded = False
                st.warning("No trained model found. Train a model first.")
            
            if model_loaded and st.button("Generate Test Predictions"):
                with st.spinner("Generating predictions..."):
                    # Test on a sample satellite
                    test_sat = st.session_state.satellites_df.iloc[0]
                    
                    om = OrbitalMechanics()
                    
                    satrec = om.tle_to_satrec(test_sat['tle_line1'], test_sat['tle_line2'])
                    orbit_data = om.propagate_orbit(satrec, datetime.utcnow(), 2.0)
                    
                    # Use first 24 points for prediction
                    initial_positions = orbit_data['positions'][:24]
                    initial_velocities = orbit_data['velocities'][:24]
                    
                    # Predict next hour
                    predictions = self.predictor.predict_with_uncertainty(
                        initial_positions,
                        initial_velocities,
                        prediction_steps=60,
                        num_samples=50
                    )
                    
                    # Compare with actual
                    actual_positions = orbit_data['positions'][24:84]
                    metrics = self.predictor.evaluate_predictions(
                        actual_positions,
                        predictions['mean']
                    )
                    
                    # Display metrics
                    st.write("**Prediction Metrics:**")
                    for key, value in metrics.items():
                        st.write(f"- {key}: {value:.2f}")
                    
                    # Plot predictions
                    fig = self.plot_prediction_comparison(
                        actual_positions,
                        predictions,
                        test_sat['name']
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Model Information")
            
            if model_loaded:
                st.write("**Model Architecture:**")
                st.write("- Type: LSTM Neural Network")
                st.write("- Layers: 3 LSTM + 2 Dense")
                st.write("- Parameters: ~500K")
                st.write("- Training samples: 10K+")
                
                st.write("\n**Capabilities:**")
                st.write("- Predict up to 7 days ahead")
                st.write("- Uncertainty quantification")
                st.write("- Works best for LEO objects")
            else:
                if st.button("Train New Model"):
                    with st.spinner("Training model... This may take several minutes."):
                        # Generate training data from current satellites
                        self.train_predictor_model()
    
    def show_satellite_data(self):
        """Show raw satellite data."""
        st.header("📡 Satellite Database")
        
        # Search and filter
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_term = st.text_input("Search satellites:", "")
        
        with col2:
            source_filter = st.selectbox(
                "Filter by source:",
                ["All"] + list(st.session_state.satellites_df['source'].unique())
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort by:",
                ["name", "norad_id", "mean_motion", "inclination", "eccentricity"]
            )
        
        # Apply filters
        df = st.session_state.satellites_df.copy()
        
        if search_term:
            df = df[df['name'].str.contains(search_term, case=False)]
        
        if source_filter != "All":
            df = df[df['source'] == source_filter]
        
        df = df.sort_values(by=sort_by)
        
        # Display data
        st.write(f"Showing {len(df)} satellites")
        
        # Pagination
        items_per_page = 20
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=max(1, len(df) // items_per_page + 1),
            value=1
        )
        
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(df))
        
        # Show data table
        display_columns = [
            'name', 'norad_id', 'source', 'inclination', 
            'eccentricity', 'mean_motion', 'fetch_time'
        ]
        st.dataframe(
            df[display_columns].iloc[start_idx:end_idx],
            use_container_width=True
        )
        
        # Export option
        if st.button("Export to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"satellite_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def create_risk_timeline(self):
        """Create risk timeline visualization."""
        risks = st.session_state.collision_risks
        
        fig = go.Figure()
        
        # Group by time bins
        times = [r['tca_time'] for r in risks]
        probs = [r['collision_probability'] for r in risks]
        distances = [r['tca_distance_km'] for r in risks]
        names = [f"{r['sat1_name'][:20]} vs {r['sat2_name'][:20]}" for r in risks]
        
        fig.add_trace(go.Scatter(
            x=times,
            y=probs,
            mode='markers',
            marker=dict(
                size=[d/2 for d in distances],  # Size by distance
                color=probs,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Probability"),
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=names,
            hovertemplate='%{text}<br>Time: %{x}<br>Probability: %{y:.2e}<br>Distance: %{marker.size:.1f} km<extra></extra>'
        ))
        
        fig.update_layout(
            title="Collision Risk Timeline",
            xaxis_title="Time (UTC)",
            yaxis_title="Collision Probability",
            yaxis_type="log",
            height=400
        )
        
        return fig
    
    def filter_risks_by_level(self, risks, level):
        """Filter risks by severity level."""
        if level == "All":
            return risks
        elif level == "High (>1%)":
            return [r for r in risks if r['collision_probability'] > 0.01]
        elif level == "Medium (0.1-1%)":
            return [r for r in risks if 0.001 <= r['collision_probability'] <= 0.01]
        elif level == "Low (<0.1%)":
            return [r for r in risks if r['collision_probability'] < 0.001]
        return risks
    
    def plot_approach_history(self, risk):
        """Plot the approach history for a collision risk."""
        times = risk['probability_history']['times']
        distances = risk['probability_history']['distances']
        probabilities = risk['probability_history']['probabilities']
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Distance Over Time", "Collision Probability Over Time"),
            shared_xaxes=True
        )
        
        # Distance plot
        fig.add_trace(
            go.Scatter(
                x=times,
                y=distances,
                mode='lines',
                name='Distance',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Probability plot
        fig.add_trace(
            go.Scatter(
                x=times,
                y=probabilities,
                mode='lines',
                name='Probability',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)
        fig.update_yaxes(title_text="Distance (km)", row=1, col=1)
        fig.update_yaxes(title_text="Probability", type="log", row=2, col=1)
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_prediction_comparison(self, actual, predictions, sat_name):
        """Plot actual vs predicted positions."""
        fig = go.Figure()
        
        # Actual trajectory
        fig.add_trace(go.Scatter3d(
            x=actual[:, 0],
            y=actual[:, 1],
            z=actual[:, 2],
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=3)
        ))
        
        # Predicted trajectory
        fig.add_trace(go.Scatter3d(
            x=predictions['mean'][:, 0],
            y=predictions['mean'][:, 1],
            z=predictions['mean'][:, 2],
            mode='lines',
            name='Predicted',
            line=dict(color='red', width=3)
        ))
        
        # Uncertainty bounds (simplified - just show a few points)
        for i in range(0, len(predictions['mean']), 10):
            fig.add_trace(go.Scatter3d(
                x=[predictions['lower_95'][i, 0], predictions['upper_95'][i, 0]],
                y=[predictions['lower_95'][i, 1], predictions['upper_95'][i, 1]],
                z=[predictions['lower_95'][i, 2], predictions['upper_95'][i, 2]],
                mode='lines',
                line=dict(color='pink', width=1),
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"Prediction vs Actual: {sat_name}",
            scene=dict(
                xaxis_title='X (km)',
                yaxis_title='Y (km)',
                zaxis_title='Z (km)'
            ),
            height=600
        )
        
        return fig
    
    def visualize_selected_satellites(self, selected_names):
        """Visualize selected satellites."""
        fig = go.Figure()
        
        # Add Earth
        fig.add_trace(self.visualizer.create_earth_sphere())
        
        # Add each selected satellite
        om = OrbitalMechanics()
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, name in enumerate(selected_names):
            sat = st.session_state.satellites_df[
                st.session_state.satellites_df['name'] == name
            ].iloc[0]
            
            # Propagate orbit
            satrec = om.tle_to_satrec(sat['tle_line1'], sat['tle_line2'])
            orbit = om.propagate_orbit(satrec, datetime.utcnow(), 1.5)
            
            # Add traces
            traces = self.visualizer.plot_orbit_3d(
                orbit,
                name,
                color=colors[i % len(colors)]
            )
            for trace in traces:
                fig.add_trace(trace)
        
        fig.update_layout(
            title="Selected Satellites Visualization",
            scene=dict(
                xaxis_title='X (km)',
                yaxis_title='Y (km)',
                zaxis_title='Z (km)',
                aspectmode='data'
            ),
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def train_predictor_model(self):
        """Train the orbital predictor model."""
        # Generate training data from current satellites
        om = OrbitalMechanics()
        
        training_data = []
        sample_sats = st.session_state.satellites_df.head(20)  # Use first 20 for demo
        
        progress_bar = st.progress(0)
        
        for idx, (_, sat) in enumerate(sample_sats.iterrows()):
            try:
                satrec = om.tle_to_satrec(sat['tle_line1'], sat['tle_line2'])
                orbit = om.propagate_orbit(satrec, datetime.utcnow() - timedelta(days=1), 24)
                
                training_data.append({
                    'positions': orbit['positions'],
                    'velocities': orbit['velocities']
                })
                
                progress_bar.progress((idx + 1) / len(sample_sats))
            except:
                continue
        
        # Train model
        if training_data:
            history = self.predictor.train(training_data, epochs=10)
            st.success("Model trained successfully!")
            
            # Show training history
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history.history['loss'],
                mode='lines',
                name='Training Loss'
            ))
            if 'val_loss' in history.history:
                fig.add_trace(go.Scatter(
                    y=history.history['val_loss'],
                    mode='lines',
                    name='Validation Loss'
                ))
            
            fig.update_layout(
                title="Model Training History",
                xaxis_title="Epoch",
                yaxis_title="Loss"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Failed to generate training data")


def main():
    """Main entry point for the dashboard."""
    dashboard = CollisionDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
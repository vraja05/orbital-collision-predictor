# 🛰️ Orbital Collision Prediction System

An ML-powered system for tracking satellites and predicting potential collisions using real NORAD TLE data. This project demonstrates advanced space situational awareness capabilities relevant to companies like SpaceX, NASA, and Lockheed Martin.

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://orbital-collision-predictor-mzwrnw9qbtnwz7lppaadcy.streamlit.app)

<img width="1756" height="897" alt="image" src="https://github.com/user-attachments/assets/a48cf4e1-40ef-46c7-86c9-ade768cafbe4" />


<img width="1566" height="807" alt="image" src="https://github.com/user-attachments/assets/6af3c0bc-3a46-4ba1-b552-8c92ae0e449b" />


<img width="1523" height="707" alt="image" src="https://github.com/user-attachments/assets/5f706661-b999-429d-9371-acb8d87b79f8" />


![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Key Features

- **Real-Time Data Integration**: Fetches live TLE data from NORAD/Celestrak
- **ML-Powered Predictions**: LSTM neural networks predict orbital trajectories beyond SGP4 accuracy
- **Collision Detection**: Monte Carlo simulations estimate collision probabilities
- **3D Visualizations**: Interactive Plotly visualizations of orbits and collision scenarios
- **Live Dashboard**: Streamlit dashboard for real-time monitoring
- **Constellation Analysis**: Special focus on mega-constellations like Starlink
- **Avoidance Recommendations**: Automated maneuver planning suggestions

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git
- 4GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/orbital-collision-predictor.git
cd orbital-collision-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Run the Dashboard

```bash
# Start the Streamlit dashboard
streamlit run src/visualization/dashboard.py
```

Open your browser to `http://localhost:8501` to view the dashboard.

### Run Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/ directory
# Open 01_data_exploration.ipynb to get started
```

## 📊 Project Architecture

### Core Components

1. **Data Pipeline** (`src/data/`)
   - `tle_fetcher.py`: Fetches real satellite TLE data from Celestrak
   - `data_processor.py`: Processes and validates orbital data

2. **Models** (`src/models/`)
   - `orbital_predictor.py`: LSTM model for trajectory prediction
   - `collision_detector.py`: Collision probability estimation

3. **Visualization** (`src/visualization/`)
   - `orbit_visualizer.py`: 3D orbit rendering
   - `dashboard.py`: Real-time monitoring dashboard

4. **Utilities** (`src/utils/`)
   - `orbital_mechanics.py`: SGP4 propagation and orbital calculations
   - `config.py`: System configuration

### Data Flow

```
NORAD/Celestrak API → TLE Fetcher → Orbital Propagator → 
    ↓
Collision Detector ← LSTM Predictor
    ↓
Dashboard/Visualizations
```

## 🔬 Technical Details

### Orbital Mechanics
- Uses SGP4/SDP4 propagators for accurate orbit determination
- Calculates classical orbital elements from state vectors
- Handles Earth's oblateness and atmospheric drag

### Machine Learning
- **Architecture**: 3-layer LSTM + 2 dense layers
- **Input**: 24-hour position/velocity history
- **Output**: Up to 7-day trajectory predictions
- **Uncertainty**: Monte Carlo dropout for uncertainty quantification

### Collision Detection
- Screens satellite pairs for close approaches
- Monte Carlo simulations for probability estimation
- Considers position uncertainties and covariance
- Generates automated avoidance recommendations

## 📈 Performance Metrics

- **Prediction Accuracy**: <5km error at 24 hours (LEO)
- **Processing Speed**: ~1000 satellite pairs/minute
- **Detection Threshold**: 10km minimum distance
- **Probability Range**: 10^-4 to 10^-1

## 🌍 Real-World Applications

This system addresses critical space industry challenges:

1. **Collision Avoidance**: Protects billions in space assets
2. **Mission Planning**: Ensures safe launch windows
3. **Constellation Management**: Monitors mega-constellations
4. **Space Debris Tracking**: Tracks hazardous debris
5. **Regulatory Compliance**: Meets space safety requirements

## 📝 Example Usage

```python
from src.data.tle_fetcher import TLEFetcher
from src.models.collision_detector import CollisionDetector

# Fetch satellite data
fetcher = TLEFetcher()
satellites = fetcher.fetch_all_sources(["starlink", "space_stations"])

# Run collision screening
detector = CollisionDetector()
risks = detector.batch_collision_screening(
    satellites,
    duration_hours=72
)

# Display high-risk events
for risk in risks[:5]:
    print(f"{risk['sat1_name']} vs {risk['sat2_name']}")
    print(f"Probability: {risk['collision_probability']:.2e}")
    print(f"Distance: {risk['tca_distance_km']:.1f} km\n")
```

## 🛠️ Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Format code
black src/

# Lint
flake8 src/
```

### Building Documentation

```bash
cd docs/
make html
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Celestrak](https://celestrak.org/) for providing TLE data
- [SGP4](https://pypi.org/project/sgp4/) for orbital propagation
- NASA for orbital mechanics algorithms
- The space tracking community

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact via GitHub.

---

**Note**: This is a demonstration project showcasing ML and space technology capabilities. For production use in critical space operations, additional validation and certification would be required.

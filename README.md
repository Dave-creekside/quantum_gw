# Quantum Gravitational Wave Detector

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

*Advanced relativistic quantum algorithms in a package so efficient your phone could probably run it*

<p align="center">
  <img src="data/assets/qgw_logo.png" alt="QGW Detector Logo" width="300" />
</p>

## 🌊 What is this?

A quantum simulation framework that lets you detect gravitational waves by modeling how they affect entangled quantum states. This project demonstrates how quantum entanglement across relativistic reference frames can be used to create an entirely new class of gravitational wave detectors.

The QGW Detector provides both a command-line interface and a web-based dashboard for analyzing gravitational wave data through quantum circuits. It transforms complex strain data from LIGO into phase shifts on entangled qubits, then measures how much gravitational information is encoded in the resulting quantum states.

## 🌟 Key Features

- **Blazing Fast Performance**: Processes gravitational wave data through quantum circuits in milliseconds, with minimal resource usage (seriously, it barely touches the GPU)
- **Compositional Architecture**: Revolutionary pipeline approach that chains multiple quantum topologies for up to 2× improvement in detection sensitivity
- **Interactive Web Dashboard**: Experiment with different detector designs through an intuitive UI with real-time visualizations
- **Multi-Stage Pipelines**: Create complex detection pipelines combining different qubit counts and topologies
- **Parameter Sweeps**: Automatically run and compare multiple configurations to find optimal settings
- **Project Management**: Save, load, and manage experiment configurations and results
- **Responsive Design**: Use the interface on desktop or mobile devices
- **Dark/Light Mode**: Choose your preferred visual theme for comfortable extended use

## 🔍 The "Conventional Approach" vs. Our Quantum Shenanigans

| LIGO | This Quantum Detector |
|------|----------------------|
| 4 km long laser interferometers | Runs on your desktop |
| Costs over $1 billion | GPU barely notices it's running |
| Mechanical isolation systems | Pure quantum mathematics |
| Took decades to develop | Hacked together in a few days |
| Huge team of scientists | Can be operated by one person |
| Requires extensive facilities | Packaged in a simple web app |

## 📊 The Magic: Pipeline Architecture

We discovered that chaining quantum detectors in specific configurations dramatically improves gravitational wave detection sensitivity. Top performers include:

```
4-linear → 4-full → 4-linear  (SNR: 5.61, 1.85× improvement)
4-star → 4-full → 4-linear    (SNR: 5.61, 1.85× improvement)
```

These configurations follow a quantum encoder→transformer→decoder pattern that might represent a fundamental principle in quantum information processing.

<p align="center">
  <img src="data/assets/pipeline_diagram.png" alt="Pipeline Architecture" width="600" />
</p>

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for larger quantum circuits)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quantum_gw_detector.git
cd quantum_gw_detector

# Create and activate virtual environment
python -m venv qgw_env
source qgw_env/bin/activate  # On Windows: qgw_env\Scripts\activate

# Install package with basic dependencies
pip install -e .
```

### Web Interface Installation

To use the web dashboard, install with the web extra:

```bash
pip install -e ".[web]"
```

### ZX-Calculus Optimization

For circuit optimization using ZX-calculus:

```bash
pip install -e ".[zx]"
```

### Full Installation (All Features)

```bash
pip install -e ".[web,zx]"
```

### Frontend Dependencies

The web interface uses:
- Vanilla JavaScript (no frameworks required)
- Font Awesome icons (loaded via CDN)
- Custom CSS

No npm/yarn dependencies are needed, making setup simple.

## 🎮 Quick Start

### Command Line Interface

```bash
# Run interactive menu
qgw-menu

# Or run a specific pipeline configuration
python -m qgw_detector.run_detector --mode single --event GW150914 --qubits 4 --entanglement linear
```

### Web Dashboard

```bash
# Start the web interface
python run_web_interface.py

# Then open your browser to http://localhost:8000
```

<p align="center">
  <img src="data/assets/web_dashboard.png" alt="Web Dashboard" width="700" />
</p>

## 💻 Web Interface Guide

The web interface provides an intuitive way to interact with the QGW Detector.

### Dashboard

The dashboard provides:
- Quick access to run preset pipelines
- System resource monitoring (CPU, RAM, GPU, VRAM)
- Recent activity summary

### Configuration

Set global parameters that apply to all pipeline runs:
- Gravitational wave event selection
- Downsampling and scale factors
- GPU acceleration toggle
- ZX-calculus optimization settings

### Pipelines

Design and run quantum detection pipelines:
- Select from preset pipeline configurations
- Build custom multi-stage pipelines
- View detailed results with collapsible sections
- Visualize quantum circuit performance

### Results

Browse and analyze saved experiment results:
- Sort and filter by date, configuration, or metrics
- Compare performance metrics across different runs
- View detailed SNR and QFI metrics for each stage

### Parameter Sweeps

Run systematic experiments across parameter ranges:
- Qubit count sweeps (test different qubit counts with fixed topology)
- Topology sweeps (test different topologies with fixed qubit count)
- Scale factor sweeps (find optimal scale factor for given configuration)

### Projects

Organize your work into project workspaces:
- Save configurations and results together
- Switch between different projects
- Track experiment history

### Visualizations

View detailed visual representations of results:
- Pipeline performance graphs
- Comparative metric visualizations
- Stage-by-stage analysis

## 🔧 How It Works

1. **Quantum Encoding**: Convert gravitational wave strain into phase shifts on entangled qubits
2. **Topology Specialization**: Different entanglement patterns (star, linear, full) excel at different aspects of signal processing
3. **Pipeline Processing**: Feed the output of one quantum detector into another for enhanced sensitivity
4. **Quantum Fisher Information**: Measure how much gravitational information is encoded in the quantum states

<p align="center">
  <img src="data/assets/qgw_process_flow.png" alt="Process Flow" width="700" />
</p>

## 🏗️ Project Structure

```
quantum_gw_detector/
├── qgw_detector/          # Main package
│   ├── quantum/          # Quantum circuit implementations
│   ├── data/             # Data handling & LIGO dataset interface
│   ├── utils/            # Utility functions
│   ├── visualization/    # Plotting and visualization tools
│   ├── api.py            # Core API for detector operations
│   └── web_api.py        # FastAPI web interface
├── frontend/             # Web interface
│   ├── css/              # Stylesheets
│   ├── js/               # JavaScript modules
│   └── assets/           # Images and backgrounds
├── data/                 # Data storage
│   ├── experiments/      # Saved experiment results
│   └── projects/         # Project workspaces
└── scripts/              # Utility scripts
```

## 🔬 Advanced Features

### Performance Optimization

- **GPU Acceleration**: Enable for up to 10x performance on larger circuits
- **ZX-Calculus Optimization**: Reduce circuit complexity before execution
- **Downsampling**: Adjust to balance between precision and performance

### Resource Considerations

- **Memory Usage**: 
  - 4-qubit circuits: ~100MB RAM
  - 6-qubit circuits: ~500MB RAM
  - 8-qubit circuits: ~2GB RAM
  - "Red Circle" (8-qubit full): >4GB VRAM when using GPU

- **GPU Memory**:
  - Linear and Star topologies are memory-efficient
  - Full connectivity increases memory requirements exponentially with qubit count

### Custom Pipeline Design Principles

For optimal results:
1. Start with an "encoder" stage (Linear or Star topology)
2. Use a "transformer" middle stage (Full topology works best)
3. End with a "decoder" stage (Linear topology often performs best)

## 🐞 Troubleshooting

### Common Issues

**Problem**: Out of memory errors with large circuits  
**Solution**: Reduce qubit count, avoid full topology, or use a machine with more RAM/VRAM

**Problem**: Web interface displays "Error fetching system stats"  
**Solution**: Ensure psutil is installed (`pip install psutil`)

**Problem**: GPU acceleration not working  
**Solution**: Check that PyTorch is installed with CUDA support (`torch.cuda.is_available()` should return True)

**Problem**: ZX optimization errors with 8-qubit full circuits  
**Solution**: These circuits are too complex for ZX optimization, disable this feature for large circuits

## 📚 Development Guide

### Adding New Topologies

1. Extend the `QuantumGWDetector` class in `qgw_detector/quantum/circuits.py`
2. Implement the entanglement pattern in the `_create_circuit` method
3. Update UI elements in both CLI and web interface

### Creating Custom Visualization Plots

Add new plotting functions to `qgw_detector/visualization/plots.py` and wire them to the API methods.

### Extending the Web Interface

The frontend follows a modular structure with separate JavaScript files for each functional area:
- `api.js`: API communication layer
- `app.js`: Core UI functionality
- `config.js`, `pipelines.js`, etc.: Feature-specific modules

## 🔜 Coming Soon™

- Web dashboard real-time data streaming
- Noise resilience testing
- Experiments with actual quantum hardware
- Integration with more gravitational wave datasets
- Multi-user collaboration features
- Pipeline icon representation system with safety controls

## ✨ Contributors

- Quantum GW Team

## 📄 License

MIT License, because we believe in open science (and who else would believe this actually works?)

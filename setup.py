# setup.py
from setuptools import setup, find_packages

setup(
    name="qgw_detector",
    version="1.0.0",
    description="Quantum Gravitational Wave Detector",
    author="Quantum GW Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pennylane>=0.28.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.61.0",
        "gwpy>=2.0.0",
        "h5py>=3.1.0",
        "tabulate>=0.8.9",
        "torch>=1.9.0",
        "psutil>=5.8.0", # Added for system stats
    ],
    extras_require={
        "web": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "pydantic>=1.8.2",
        ],
        "zx": [
            "pyzx>=0.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qgw-menu=qgw_detector.cli_menu:main",
        ],
    },
)

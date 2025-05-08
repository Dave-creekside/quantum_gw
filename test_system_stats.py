#!/usr/bin/env python3
"""
Test script for the system stats functionality.
"""

from qgw_detector.api import QuantumGWAPI

def test_system_stats():
    """Test the system stats functionality."""
    api = QuantumGWAPI()
    
    print("\n=== Testing System Stats ===")
    stats = api.get_system_stats()
    
    print("\nSystem Stats Results:")
    print(f"CPU Usage: {stats['cpu_percent']}%")
    print(f"RAM: {stats['ram_used_gb']:.2f}GB / {stats['ram_total_gb']:.2f}GB ({stats['ram_percent']}%)" 
          if isinstance(stats['ram_used_gb'], float) else f"RAM: {stats['ram_used_gb']} / {stats['ram_total_gb']} ({stats['ram_percent']}%)")
    print(f"GPU: {stats['gpu_name']}")
    print(f"GPU Utilization: {stats['gpu_utilization_percent']}%")
    print(f"GPU Temperature: {stats['gpu_temperature_c']}°C")
    print(f"VRAM: {stats['vram_used_mb']} MB / {stats['vram_total_mb']} MB")
    print(f"VRAM Percentage: {stats['vram_percent']}%")
    
    return stats

if __name__ == "__main__":
    test_system_stats()

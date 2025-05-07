#!/usr/bin/env python
"""
Test script for Quantum Gravitational Wave Detector Web API
"""
import time
import json
import subprocess
import threading
import requests
from pprint import pprint

def start_web_api_server():
    """Start the web API server in a separate process"""
    print("Starting web API server...")
    process = subprocess.Popen(["python", "-m", "qgw_detector.web_api"])
    
    # Give server more time to start
    time.sleep(5)
    return process

def test_config_endpoints():
    """Test configuration management endpoints"""
    print("\n=== Testing Configuration Endpoints ===")
    
    # Get current config
    response = requests.get("http://localhost:8000/config")
    assert response.status_code == 200, "Failed to get config"
    config = response.json()
    print("Initial config:")
    pprint(config)
    
    # Update config
    new_config = {
        "event_name": "GW151226",
        "downsample_factor": 300,
        "scale_factor": 2e21,
        "use_gpu": False,
        "use_zx_opt": False
    }
    response = requests.post("http://localhost:8000/config", json=new_config)
    assert response.status_code == 200, "Failed to update config"
    updated_config = response.json()
    print("\nUpdated config:")
    pprint(updated_config)
    
    # Verify changes were applied
    assert updated_config['event_name'] == "GW151226"
    assert updated_config['downsample_factor'] == 300
    assert updated_config['scale_factor'] == 2e21
    assert updated_config['use_gpu'] is False
    
    # Reset config for other tests
    response = requests.post("http://localhost:8000/config", json={
        "event_name": "GW150914",
        "downsample_factor": 500,  # Higher value for faster tests
        "scale_factor": 1e21,
        "use_gpu": True
    })
    assert response.status_code == 200, "Failed to reset config"
    
    print("Configuration endpoints test passed!")

def test_preset_endpoints():
    """Test preset management endpoints"""
    print("\n=== Testing Preset Endpoints ===")
    
    # List presets
    response = requests.get("http://localhost:8000/presets")
    assert response.status_code == 200, "Failed to get presets"
    presets = response.json()
    print("Available presets:")
    for name, config in presets.items():
        pipeline_str = " → ".join([f"{q}-{t}" for q, t in config])
        print(f"  {name}: {pipeline_str}")
    
    # Add a preset
    test_preset_name = "api_test_preset_" + str(int(time.time()))
    test_config = {
        "name": test_preset_name,
        "config": [(4, "star"), (6, "linear")]
    }
    response = requests.post("http://localhost:8000/presets", json=test_config)
    assert response.status_code == 200, "Failed to add preset"
    result = response.json()
    print(f"\nAdded preset: {test_preset_name}")
    
    # Verify preset was added by listing again
    response = requests.get("http://localhost:8000/presets")
    assert response.status_code == 200, "Failed to get updated presets"
    updated_presets = response.json()
    assert test_preset_name in updated_presets, f"Preset {test_preset_name} not found in updated presets"
    
    print("Preset endpoints test passed!")

def test_run_pipeline_endpoint():
    """Test running a pipeline via the API"""
    print("\n=== Testing Pipeline Execution Endpoint ===")
    
    # Run a simple one-stage pipeline
    pipeline_request = {
        "stages": [(4, "star")],
        "save_results": True,
        "save_visualization": True
    }
    print(f"Running pipeline: {pipeline_request['stages']}")
    
    start_time = time.time()
    response = requests.post("http://localhost:8000/run", json=pipeline_request)
    end_time = time.time()
    
    assert response.status_code == 200, f"Failed to run pipeline: {response.text}"
    result = response.json()
    
    print(f"Pipeline executed in {end_time - start_time:.2f} seconds")
    print(f"Final SNR: {result['summary']['final_stage_snr']:.4f}")
    
    # Verify result structure
    assert 'stages' in result
    assert 'summary' in result
    assert len(result['stages']) == 1
    assert result['stages'][0]['qubits'] == 4
    assert result['stages'][0]['topology'] == 'star'
    
    print("Pipeline execution endpoint test passed!")

def test_compare_pipelines_endpoint():
    """Test comparing pipelines via the API"""
    print("\n=== Testing Pipeline Comparison Endpoint ===")
    
    # Compare two simple pipelines
    comparison_request = {
        "pipeline_configs": [
            [(4, "star")],
            [(4, "linear")]
        ],
        "names": ["Star Config", "Linear Config"]
    }
    print(f"Comparing pipelines: {comparison_request['names']}")
    
    response = requests.post("http://localhost:8000/compare", json=comparison_request)
    assert response.status_code == 200, f"Failed to compare pipelines: {response.text}"
    result = response.json()
    
    # Verify result structure
    assert 'pipelines' in result
    assert len(result['pipelines']) == 2
    
    # Print comparison results
    for pipeline in result['pipelines']:
        print(f"Pipeline {pipeline['name']}: SNR = {pipeline['final_snr']:.4f}, " +
              f"Improvement = {pipeline['improvement']:.2f}x")
    
    print("Pipeline comparison endpoint test passed!")

def test_sweep_endpoints():
    """Test parameter sweep endpoints"""
    print("\n=== Testing Parameter Sweep Endpoints ===")
    
    # Test qubit count sweep
    print("Testing qubit count sweep...")
    sweep_request = {
        "topology": "star",
        "qubit_counts": [4, 6],
        "base_config_params": {"downsample_factor": 500}
    }
    
    response = requests.post("http://localhost:8000/sweep/qubit_count", json=sweep_request)
    assert response.status_code == 200, f"Failed to run qubit count sweep: {response.text}"
    results = response.json()
    
    print(f"Qubit count sweep completed with {len(results)} results")
    assert len(results) == 2
    
    # Test topology sweep
    print("\nTesting topology sweep...")
    sweep_request = {
        "qubit_count": 4,
        "topologies": ["star", "linear"],
        "base_config_params": {"downsample_factor": 500}
    }
    
    response = requests.post("http://localhost:8000/sweep/topology", json=sweep_request)
    assert response.status_code == 200, f"Failed to run topology sweep: {response.text}"
    results = response.json()
    
    print(f"Topology sweep completed with {len(results)} results")
    assert len(results) == 2
    
    # Test scale factor sweep
    print("\nTesting scale factor sweep...")
    sweep_request = {
        "pipeline_config": [(4, "star")],
        "scale_factors": [1e20, 1e21],
        "base_config_params": {"downsample_factor": 500}
    }
    
    response = requests.post("http://localhost:8000/sweep/scale_factor", json=sweep_request)
    assert response.status_code == 200, f"Failed to run scale factor sweep: {response.text}"
    results = response.json()
    
    print(f"Scale factor sweep completed with {len(results)} results")
    assert len(results) == 2
    
    print("Parameter sweep endpoints test passed!")

def test_results_endpoints():
    """Test results management endpoints"""
    print("\n=== Testing Results Management Endpoints ===")
    
    # List saved results
    response = requests.get("http://localhost:8000/results")
    assert response.status_code == 200, "Failed to list results"
    results = response.json()
    
    print(f"Found {len(results)} saved results")
    
    if results:
        # Get the first result
        result_id = results[0]['identifier']
        print(f"Retrieving result: {result_id}")
        
        response = requests.get(f"http://localhost:8000/results/{result_id}")
        assert response.status_code == 200, f"Failed to get result: {response.text}"
        result = response.json()
        
        print(f"Retrieved result from {result.get('timestamp', 'unknown')}")
        print(f"Pipeline: {result.get('pipeline_config', 'unknown')}")
        
        # Verify result structure
        assert 'stages' in result
        assert 'summary' in result
        
        print("Results management endpoints test passed!")
    else:
        print("No saved results found - run pipeline tests first")

def test_api_server():
    """Run all web API tests"""
    # Start the server
    server_process = start_web_api_server()
    
    try:
        print("\n" + "="*72)
        print("QUANTUM GRAVITATIONAL WAVE DETECTOR WEB API TESTS")
        print("="*72)
        
        # Run tests
        test_config_endpoints()
        test_preset_endpoints()
        test_run_pipeline_endpoint()
        test_compare_pipelines_endpoint()
        test_sweep_endpoints()
        test_results_endpoints()
        
        print("\n" + "="*72)
        print("WEB API TESTS COMPLETE")
        print("="*72)
    finally:
        # Clean up server process
        print("\nStopping web API server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    test_api_server()

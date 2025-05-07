#!/usr/bin/env python
"""
Test script for Quantum Gravitational Wave Detector API
"""
import os
import json
import time
from pprint import pprint

from qgw_detector.api import QuantumGWAPI

def test_config():
    """Test configuration management"""
    print("\n=== Testing Configuration ===")
    api = QuantumGWAPI()
    
    # Get current config
    config = api.get_config()
    print("Initial config:")
    pprint(config)
    
    # Update config
    new_config = api.set_config(
        event_name="GW151226",
        downsample_factor=300,
        scale_factor=2e21,
        use_gpu=False,
        use_zx_opt=False
    )
    print("\nUpdated config:")
    pprint(new_config)
    
    # Verify changes were applied
    assert new_config['event_name'] == "GW151226"
    assert new_config['downsample_factor'] == 300
    assert new_config['scale_factor'] == 2e21
    assert new_config['use_gpu'] is False
    
    # Reset config for other tests
    api.set_config(
        event_name="GW150914",
        downsample_factor=200,
        scale_factor=1e21,
        use_gpu=True
    )
    print("\nConfig test passed!")

def test_presets():
    """Test preset management"""
    print("\n=== Testing Preset Management ===")
    api = QuantumGWAPI()
    
    # List presets
    presets = api.list_presets()
    print("Available presets:")
    for name, config in presets.items():
        pipeline_str = " → ".join([f"{q}-{t}" for q, t in config])
        print(f"  {name}: {pipeline_str}")
    
    # Test adding a preset
    test_preset_name = "test_preset_" + str(int(time.time()))
    test_config = [(4, "star"), (6, "linear")]
    try:
        result = api.add_preset(test_preset_name, test_config)
        print(f"\nAdded preset: {test_preset_name}")
        
        # Verify preset was added
        updated_presets = api.list_presets()
        assert test_preset_name in updated_presets
        assert updated_presets[test_preset_name] == test_config
        print("Preset test passed!")
    except Exception as e:
        print(f"Error adding preset: {e}")

def test_run_pipeline():
    """Test running a pipeline"""
    print("\n=== Testing Pipeline Execution ===")
    api = QuantumGWAPI()
    
    # Set minimal config to speed up test
    api.set_config(downsample_factor=500)
    
    # Run a simple one-stage pipeline
    pipeline_config = [(4, "star")]
    print(f"Running pipeline: {pipeline_config}")
    
    try:
        start_time = time.time()
        result = api.run_pipeline(pipeline_config)
        end_time = time.time()
        
        print(f"Pipeline executed in {end_time - start_time:.2f} seconds")
        print(f"Final SNR: {result['summary']['final_stage_snr']:.4f}")
        
        # Verify result structure
        assert 'stages' in result
        assert 'summary' in result
        assert len(result['stages']) == 1
        assert result['stages'][0]['qubits'] == 4
        assert result['stages'][0]['topology'] == 'star'
        
        print("Pipeline execution test passed!")
    except Exception as e:
        print(f"Error running pipeline: {e}")

def test_parameter_sweeps():
    """Test parameter sweeps"""
    print("\n=== Testing Parameter Sweeps ===")
    api = QuantumGWAPI()
    
    # Set minimal config to speed up test
    api.set_config(downsample_factor=500)
    
    # Test qubit count sweep
    print("Testing qubit count sweep...")
    try:
        sweep_results = api.sweep_qubit_count("star", [4, 6])
        print(f"Qubit count sweep completed with {len(sweep_results)} results")
        assert len(sweep_results) == 2
        
        # Test topology sweep
        print("\nTesting topology sweep...")
        sweep_results = api.sweep_topology(4, ["star", "linear"])
        print(f"Topology sweep completed with {len(sweep_results)} results")
        assert len(sweep_results) == 2
        
        # Test scale factor sweep
        print("\nTesting scale factor sweep...")
        sweep_results = api.sweep_scale_factor([(4, "star")], [1e20, 1e21])
        print(f"Scale factor sweep completed with {len(sweep_results)} results")
        assert len(sweep_results) == 2
        
        print("Parameter sweeps test passed!")
    except Exception as e:
        print(f"Error in parameter sweeps: {e}")

def test_results_management():
    """Test results management"""
    print("\n=== Testing Results Management ===")
    api = QuantumGWAPI()
    
    # List saved results
    results = api.list_saved_results()
    print(f"Found {len(results)} saved results")
    
    if results:
        # Test retrieving a result
        result_id = results[0]['identifier']
        print(f"Retrieving result: {result_id}")
        try:
            result = api.get_saved_result(result_id)
            print(f"Retrieved result from {result.get('timestamp', 'unknown')}")
            print(f"Pipeline: {result.get('pipeline_config', 'unknown')}")
            
            # Verify result structure
            assert 'stages' in result
            assert 'summary' in result
            print("Results management test passed!")
        except Exception as e:
            print(f"Error retrieving result: {e}")
    else:
        print("No saved results found - run pipeline tests first")

def main():
    """Run all tests"""
    print("\n" + "="*72)
    print("QUANTUM GRAVITATIONAL WAVE DETECTOR API TESTS")
    print("="*72)
    
    # Run tests
    test_config()
    test_presets()
    test_run_pipeline()
    test_parameter_sweeps()
    test_results_management()
    
    print("\n" + "="*72)
    print("API TESTS COMPLETE")
    print("="*72)

if __name__ == "__main__":
    main()

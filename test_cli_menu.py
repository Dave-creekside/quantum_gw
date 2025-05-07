#!/usr/bin/env python
"""
Test script for Quantum Gravitational Wave Detector CLI Menu
"""
import sys
import os
import time
from io import StringIO
from unittest.mock import patch

from cli_menu import QuantumGWMenu

def test_cli_init():
    """Test CLI menu initialization"""
    print("\n=== Testing CLI Menu Initialization ===")
    menu = QuantumGWMenu()
    
    # Verify menu has API instance
    assert hasattr(menu, 'api'), "Menu does not have API instance"
    print("CLI menu initialized successfully!")
    return menu

def test_preset_listing():
    """Test CLI menu preset listing"""
    print("\n=== Testing CLI Menu Preset Listing ===")
    menu = QuantumGWMenu()
    
    # Mock input for 'Run Preset Pipeline' and then return to menu
    with patch('builtins.input', side_effect=['0']):
        # Capture stdout to verify output
        with patch('sys.stdout', new=StringIO()) as fake_out:
            menu.run_preset_pipeline()
            output = fake_out.getvalue()
            
            # Check if presets are listed
            print("Verifying preset listing...")
            assert "Available Preset Pipelines:" in output, "Preset listing not found in output"
            assert "star" in output or "linear" in output or "full" in output, "No preset topologies found in output"
    
    print("CLI menu preset listing test passed!")

def test_settings_menu():
    """Test CLI menu settings interface"""
    print("\n=== Testing CLI Menu Settings Interface ===")
    menu = QuantumGWMenu()
    
    # Mock input to enter settings menu and exit
    with patch('builtins.input', side_effect=['0']):
        # Capture stdout to verify output
        with patch('sys.stdout', new=StringIO()) as fake_out:
            menu.change_settings()
            output = fake_out.getvalue()
            
            # Check if settings options are displayed
            print("Verifying settings menu...")
            assert "Current Settings:" in output, "Settings header not found in output"
            # Check if API values are used for settings display
            config = menu.api.get_config()
            assert f"Event: {config['event_name']}" in output, "Event setting not found in output"
            assert f"Downsample factor: {config['downsample_factor']}" in output, "Downsample factor not found in output"
    
    print("CLI menu settings interface test passed!")

def test_view_results():
    """Test CLI menu result viewing interface"""
    print("\n=== Testing CLI Menu Results Viewer ===")
    menu = QuantumGWMenu()
    
    # Mock input to view results and exit
    with patch('builtins.input', side_effect=['0']):
        # Capture stdout to verify output
        with patch('sys.stdout', new=StringIO()) as fake_out:
            menu.view_results()
            output = fake_out.getvalue()
            
            # Check if it properly uses the API
            print("Verifying results viewer...")
            assert "VIEW PREVIOUS RESULTS" in output, "Results viewer header not found in output"
            
            # We can't guarantee there are results, but the menu should show either results
            # or a "no results" message
            assert "Available results:" in output or "No saved results found" in output, "Results listing not found in output"
    
    print("CLI menu results viewer test passed!")

def test_parameter_sweep_menu():
    """Test CLI menu parameter sweep interface"""
    print("\n=== Testing CLI Menu Parameter Sweep Interface ===")
    menu = QuantumGWMenu()
    
    # Mock input to enter parameter sweep menu and exit
    with patch('builtins.input', side_effect=['4']):  # Choose "Return to main menu"
        # Capture stdout to verify output
        with patch('sys.stdout', new=StringIO()) as fake_out:
            menu.parameter_sweep()
            output = fake_out.getvalue()
            
            # Check if parameter sweep options are displayed
            print("Verifying parameter sweep menu...")
            assert "PARAMETER SWEEP" in output, "Parameter sweep header not found in output"
            assert "Qubit count" in output, "Qubit count sweep option not found in output"
            assert "Topology" in output, "Topology sweep option not found in output"
            assert "Strain scale factor" in output, "Scale factor sweep option not found in output"
    
    print("CLI menu parameter sweep interface test passed!")

def run_all_cli_tests():
    """Run all CLI menu tests"""
    print("\n" + "="*72)
    print("QUANTUM GRAVITATIONAL WAVE DETECTOR CLI MENU TESTS")
    print("="*72)
    
    test_cli_init()
    test_preset_listing()
    test_settings_menu()
    test_view_results()
    test_parameter_sweep_menu()
    
    print("\n" + "="*72)
    print("CLI MENU TESTS COMPLETE")
    print("="*72)

if __name__ == "__main__":
    run_all_cli_tests()

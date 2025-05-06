# qgw_detector/cli.py
import os
import sys
import argparse
import json
from tabulate import tabulate

from qgw_detector.api import QuantumGWAPI

def main():
    """Command-line interface for the Quantum GW Detector API"""
    parser = argparse.ArgumentParser(
        description="Quantum Gravitational Wave Detector API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Run pipeline command
    run_parser = subparsers.add_parser('run', help='Run a pipeline')
    run_parser.add_argument('--preset', help='Use a preset pipeline')
    run_parser.add_argument('--custom', help='Custom pipeline configuration (format: "4-star,6-full,4-linear")')
    run_parser.add_argument('--event', default='GW150914', help='Gravitational wave event name')
    run_parser.add_argument('--downsample', type=int, default=200, help='Downsample factor')
    run_parser.add_argument('--scale', type=float, default=1e21, help='Scale factor')
    run_parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    run_parser.add_argument('--no-save', action='store_true', help='Don\'t save results')
    run_parser.add_argument('--no-viz', action='store_true', help='Don\'t generate visualization')
    
    # List presets command
    subparsers.add_parser('list-presets', help='List available preset pipelines')
    
    # List events command
    subparsers.add_parser('list-events', help='List available gravitational wave events')
    
    # Compare pipelines command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple pipelines')
    compare_parser.add_argument('--presets', help='Comma-separated list of preset names to compare')
    compare_parser.add_argument('--custom', help='Comma-separated list of custom pipelines to compare')
    compare_parser.add_argument('--event', default='GW150914', help='Gravitational wave event name')
    
    # Interactive menu command
    subparsers.add_parser('menu', help='Launch interactive menu')
    
    args = parser.parse_args()
    
    # Create API instance
    api = QuantumGWAPI()
    
    # No command or menu command - launch interactive menu
    if args.command is None or args.command == 'menu':
        # Import the menu and run it
        from qgw_detector.cli_menu import QuantumGWMenu
        menu = QuantumGWMenu()
        menu.main_menu()
        return
    
    # List presets command
    if args.command == 'list-presets':
        presets = api.list_presets()
        print("\nAvailable preset pipelines:")
        for name, config in presets.items():
            pipeline_str = " → ".join([f"{q}-{t}" for q, t in config])
            print(f"  {name}: {pipeline_str}")
        return
    
    # List events command
    if args.command == 'list-events':
        events = api.get_available_events()
        print("\nAvailable gravitational wave events:")
        for event in events:
            print(f"  {event}")
        return
    
    # Run pipeline command
    if args.command == 'run':
        # Set configuration
        api.set_config(
            event_name=args.event,
            downsample_factor=args.downsample,
            scale_factor=args.scale,
            use_gpu=args.gpu
        )
        
        # Determine pipeline configuration
        pipeline_config = None
        
        if args.preset:
            presets = api.list_presets()
            if args.preset in presets:
                pipeline_config = presets[args.preset]
            else:
                print(f"Error: Preset '{args.preset}' not found. Use 'list-presets' to see available presets.")
                return
        elif args.custom:
            try:
                stages = args.custom.split(',')
                pipeline_config = []
                
                for stage in stages:
                    parts = stage.split('-')
                    if len(parts) != 2:
                        raise ValueError(f"Invalid stage format: {stage}")
                    
                    qubits = int(parts[0])
                    topology = parts[1]
                    
                    if topology not in ['star', 'linear', 'full']:
                        raise ValueError(f"Invalid topology: {topology}")
                    
                    pipeline_config.append((qubits, topology))
            except Exception as e:
                print(f"Error parsing custom pipeline: {e}")
                return
        else:
            print("Error: Either --preset or --custom must be specified.")
            return
        
        # Run the pipeline
        print(f"\nRunning pipeline: {' → '.join([f'{q}-{t}' for q, t in pipeline_config])}")
        result = api.run_pipeline(
            pipeline_config=pipeline_config,
            save_results=not args.no_save,
            save_visualization=not args.no_viz
        )
        
        # Print stage results
        print("\nStage Results:")
        headers = ["Stage", "Configuration", "SNR", "Max QFI", "Time (s)"]
        table_data = []
        
        for stage in result['stages']:
            config = f"{stage['qubits']}-{stage['topology']}"
            table_data.append([
                stage['stage'],
                config,
                f"{stage['qfi_snr']:.4f}",
                f"{stage['max_qfi']:.2f}",
                f"{stage['execution_time']:.2f}"
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Print summary
        summary = result['summary']
        print("\nSummary:")
        print(f"  First stage SNR: {summary['first_stage_snr']:.4f}")
        print(f"  Final stage SNR: {summary['final_stage_snr']:.4f}")
        print(f"  Improvement factor: {summary['improvement_factor']:.2f}x")
        print(f"  Total execution time: {summary['total_execution_time']:.2f} seconds")
        
        # Print file paths
        if 'file_paths' in result:
            print("\nFiles:")
            for key, path in result['file_paths'].items():
                print(f"  {key}: {path}")
        
        return
    
    # Compare pipelines command
    if args.command == 'compare':
        # Set configuration
        api.set_config(event_name=args.event)
        
        pipeline_configs = []
        names = []
        
        # Handle preset pipelines
        if args.presets:
            preset_names = args.presets.split(',')
            presets = api.list_presets()
            
            for name in preset_names:
                if name in presets:
                    pipeline_configs.append(presets[name])
                    names.append(name)
                else:
                    print(f"Warning: Preset '{name}' not found. Skipping.")
        
        # Handle custom pipelines
        if args.custom:
            custom_configs = args.custom.split(';')
            
            for i, config_str in enumerate(custom_configs):
                try:
                    stages = config_str.split(',')
                    config = []
                    
                    for stage in stages:
                        parts = stage.split('-')
                        if len(parts) != 2:
                            raise ValueError(f"Invalid stage format: {stage}")
                        
                        qubits = int(parts[0])
                        topology = parts[1]
                        
                        if topology not in ['star', 'linear', 'full']:
                            raise ValueError(f"Invalid topology: {topology}")
                        
                        config.append((qubits, topology))
                    
                    pipeline_configs.append(config)
                    names.append(f"Custom {i+1}")
                except Exception as e:
                    print(f"Warning: Error parsing custom pipeline {i+1}: {e}. Skipping.")
        
        if not pipeline_configs:
            print("Error: No valid pipelines specified for comparison.")
            return
        
        # Run the comparison
        print(f"\nComparing {len(pipeline_configs)} pipelines...")
        
        result = api.compare_pipelines(pipeline_configs, names)
        
        # Print results
        print("\nComparison Results (sorted by SNR):")
        headers = ["Rank", "Pipeline", "SNR", "Improvement", "Time (s)"]
        table_data = []
        
        for i, pipeline in enumerate(result['pipelines']):
            table_data.append([
                i+1,
                pipeline['name'],
                f"{pipeline['final_snr']:.4f}",
                f"{pipeline['improvement']:.2f}x",
                f"{pipeline['execution_time']:.2f}"
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Print file paths
        if 'file_paths' in result:
            print("\nFiles:")
            for key, path in result['file_paths'].items():
                print(f"  {key}: {path}")
        
        return

if __name__ == "__main__":
    main()
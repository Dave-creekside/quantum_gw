# qgw_detector/run_pipeline.py
import os
import argparse
import time
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Import our modules
from qgw_detector.data.ligo import fetch_gw_data, preprocess_gw_data
from qgw_detector.quantum.circuits import QuantumGWDetector
from qgw_detector.visualization.plots import GWDetectionVisualizer

def run_detector_pipeline(event_name="GW150914", pipeline=None, 
                        downsample_factor=200, scale_factor=1e21,
                        use_gpu=True, save_results=True, plot_results=True):
    """
    Run a pipeline of quantum GW detectors
    
    Args:
        event_name: Name of gravitational wave event
        pipeline: List of detector configurations ['qubits-topology', ...]
        downsample_factor: Factor to downsample data
        scale_factor: Scale factor to amplify strain values
        use_gpu: Whether to use GPU acceleration
        save_results: Whether to save results to file
        plot_results: Whether to visualize results
        
    Returns:
        dict: Results dictionary
    """
    # Default pipeline: LIGO -> 4-star -> 6-full -> 4-linear
    if pipeline is None:
        pipeline = ['4-star', '6-full', '4-linear']
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*80)
    print(f" QUANTUM GRAVITATIONAL WAVE DETECTOR - Pipeline {timestamp} ")
    print("="*80)
    print(f"Event: {event_name}")
    print(f"Pipeline: LIGO -> {' -> '.join(pipeline)}")
    print(f"Downsample factor: {downsample_factor}")
    print(f"Strain scale factor: {scale_factor:.2e}")
    print(f"GPU acceleration: {use_gpu}")
    print("="*80 + "\n")
    
    # Create results directory if needed
    if save_results:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
    
    # Create visualizer if needed
    if plot_results:
        visualizer = GWDetectionVisualizer(save_dir="data/visualization")
        os.makedirs("data/visualization", exist_ok=True)
    
    # Parse pipeline configuration
    pipeline_stages = []
    for stage_config in pipeline:
        parts = stage_config.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid stage config '{stage_config}'. Format should be 'qubits-topology'")
        
        try:
            qubits = int(parts[0])
            topology = parts[1]
            
            if topology not in ['star', 'linear', 'full']:
                raise ValueError(f"Unknown topology '{topology}'. Must be 'star', 'linear', or 'full'")
            
            pipeline_stages.append({
                'qubits': qubits,
                'topology': topology,
                'results': {}
            })
            
        except ValueError as e:
            print(f"Error parsing '{stage_config}': {e}")
            return None
    
    # 1. Fetch LIGO data
    print("Fetching LIGO data...")
    times, strain, sample_rate = fetch_gw_data(event_name)
    
    if times is None or strain is None:
        print("Failed to fetch LIGO data. Exiting.")
        return None
    
    # 2. Preprocess data
    print("\nPreprocessing LIGO data...")
    proc_times, proc_strain = preprocess_gw_data(times, strain, sample_rate)
    
    # Save initial data visualization
    if plot_results:
        plt.figure(figsize=(12, 6))
        plt.plot(proc_times, proc_strain * 1e21)
        plt.title(f"LIGO Strain Data: {event_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Strain (×10²¹)")
        plt.grid(True)
        plt.savefig(f"data/visualization/{event_name}_pipeline_input_{timestamp}.png", 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Process through the pipeline
    # Start with LIGO data
    current_times = proc_times
    current_strain = proc_strain
    
    # For visualization
    all_stages_data = [{
        'name': 'LIGO',
        'times': current_times,
        'data': current_strain * 1e21,
        'label': 'Strain (×10²¹)'
    }]
    
    # Process through each stage in the pipeline
    for i, stage in enumerate(pipeline_stages):
        print(f"\n=== Pipeline Stage {i+1}: {stage['qubits']}-qubit {stage['topology']} detector ===\n")
        
        # Create detector for this stage
        detector = QuantumGWDetector(
            n_qubits=stage['qubits'],
            entanglement_type=stage['topology'],
            use_gpu=use_gpu
        )
        
        # Apply stage-specific downsample factor
        # (We only downsample in first stage, then process at full resolution)
        stage_downsample = downsample_factor if i == 0 else 1
        
        # Apply stage-specific scale factor (amplify at each stage)
        stage_scale = scale_factor * (i+1)
        
        # Process data through this detector
        print(f"Processing data through detector...")
        start_time = time.time()
        ds_times, ds_strain, quantum_states = detector.process_gw_data(
            current_times, current_strain, 
            downsample_factor=stage_downsample,
            scale_factor=stage_scale
        )
        process_time = time.time() - start_time
        
        # Calculate quantum Fisher information
        qfi_values = detector.calculate_qfi(quantum_states, ds_times)
        
        # Calculate detector response (our "detection metric")
        detection_metric = detector.calculate_detection_metric(quantum_states)
        
        # Store results for this stage
        stage['results'] = {
            'times': ds_times,
            'qfi': qfi_values,
            'detection_metric': detection_metric,
            'process_time': process_time,
            'max_qfi': float(np.max(qfi_values)),
            'mean_qfi': float(np.mean(qfi_values)),
            'qfi_snr': float(np.max(qfi_values) / np.std(qfi_values)) if np.std(qfi_values) > 0 else 0
        }
        
        # Store data for visualization
        all_stages_data.append({
            'name': f"Stage {i+1}: {stage['qubits']}-{stage['topology']}",
            'times': ds_times,
            'data': detection_metric,
            'label': 'Detection Metric'
        })
        
        # Use this stage's output as input to the next stage
        current_times = ds_times
        current_strain = detection_metric  # Feed detection metric to the next stage
        
        print(f"Stage {i+1} complete:")
        print(f"  QFI SNR: {stage['results']['qfi_snr']:.3f}")
        print(f"  Process time: {process_time:.2f} seconds")
    
    # 4. Create visualization of the complete pipeline
    if plot_results:
        # Create multi-panel plot for all stages
        n_stages = len(all_stages_data)
        plt.figure(figsize=(12, 4 * n_stages))
        
        for i, stage_data in enumerate(all_stages_data):
            plt.subplot(n_stages, 1, i+1)
            plt.plot(stage_data['times'], stage_data['data'])
            plt.title(stage_data['name'])
            plt.ylabel(stage_data['label'])
            plt.grid(True)
            
            # Only add x label to bottom plot
            if i == n_stages - 1:
                plt.xlabel("Time (s)")
        
        plt.tight_layout()
        pipeline_viz_path = f"data/visualization/{event_name}_pipeline_{timestamp}.png"
        plt.savefig(pipeline_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nPipeline visualization saved to: {pipeline_viz_path}")
    
    # 5. Save results
    if save_results:
        # Create results dictionary
        results = {
            'event_name': event_name,
            'pipeline': pipeline,
            'downsample_factor': downsample_factor,
            'scale_factor': float(scale_factor),
            'timestamp': timestamp,
            'stages': []
        }
        
        # Add each stage's results
        for i, stage in enumerate(pipeline_stages):
            stage_results = {
                'index': i+1,
                'qubits': stage['qubits'],
                'topology': stage['topology'],
                'process_time': stage['results']['process_time'],
                'max_qfi': stage['results']['max_qfi'],
                'mean_qfi': stage['results']['mean_qfi'],
                'qfi_snr': stage['results']['qfi_snr']
            }
            results['stages'].append(stage_results)
        
        # Save to JSON file
        results_path = f"{results_dir}/{event_name}_pipeline_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Pipeline results saved to: {results_path}")
    
    # 6. Print summary table
    print("\n" + "="*80)
    print(" PIPELINE RESULTS SUMMARY ")
    print("="*80)
    print(f"Event: {event_name}")
    print(f"Pipeline: LIGO -> {' -> '.join(pipeline)}")
    print("\nStage Performance:")
    print("-" * 78)
    print(f"{'Stage':^12} | {'Config':^14} | {'QFI SNR':^10} | {'Max QFI':^10} | {'Time (s)':^10}")
    print("-" * 78)
    
    # Print LIGO as first stage
    print(f"{'LIGO':^12} | {'N/A':^14} | {'N/A':^10} | {'N/A':^10} | {'N/A':^10}")
    
    # Print each detector stage
    total_time = 0
    for i, stage in enumerate(pipeline_stages):
        config = f"{stage['qubits']}-{stage['topology']}"
        process_time = stage['results']['process_time']
        total_time += process_time
        
        print(f"{f'Stage {i+1}':^12} | {config:^14} | "
              f"{stage['results']['qfi_snr']:^10.2f} | "
              f"{stage['results']['max_qfi']:^10.1f} | "
              f"{process_time:^10.2f}")
    
    print("-" * 78)
    print(f"{'Total':^12} | {' ':^14} | {' ':^10} | {' ':^10} | {total_time:^10.2f}")
    print("="*80 + "\n")
    
    return results

def main():
    """Command-line interface for the pipeline"""
    parser = argparse.ArgumentParser(
        description="Quantum Gravitational Wave Detector Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Event selection
    parser.add_argument("--event", default="GW150914", 
                      help="Gravitational wave event name")
    
    # Pipeline configuration
    parser.add_argument("--pipeline", type=str, nargs='+', 
                      default=['4-star', '6-full', '4-linear'],
                      help="Pipeline stages (format: 'qubits-topology')")
    
    # Processing parameters
    parser.add_argument("--downsample", type=int, default=200, 
                      help="Downsample factor for data")
    parser.add_argument("--scale", type=float, default=1e21, 
                      help="Scale factor for strain amplification")
    
    # Hardware options
    parser.add_argument("--cpu", action="store_true", 
                      help="Disable GPU acceleration (use CPU)")
    
    # Output options
    parser.add_argument("--no-save", action="store_true", 
                      help="Don't save results")
    parser.add_argument("--no-plot", action="store_true", 
                      help="Don't generate plots")
    
    args = parser.parse_args()
    
    # Run the pipeline
    run_detector_pipeline(
        event_name=args.event,
        pipeline=args.pipeline,
        downsample_factor=args.downsample,
        scale_factor=args.scale,
        use_gpu=not args.cpu,
        save_results=not args.no_save,
        plot_results=not args.no_plot
    )

if __name__ == "__main__":
    main()
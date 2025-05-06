# qgw_detector/run_detector.py
import os
import argparse
import time
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt  # Add missing matplotlib import

# Import our modules
from qgw_detector.data.ligo import fetch_gw_data, preprocess_gw_data
from qgw_detector.quantum.circuits import QuantumGWDetector
from qgw_detector.quantum.error_corrected_detector import ErrorCorrectedQuantumGWDetector
from qgw_detector.visualization.plots import GWDetectionVisualizer

def run_quantum_gw_detector(event_name="GW150914", n_qubits=8, 
                          entanglement_type="star", downsample_factor=100,
                          scale_factor=1e21, use_gpu=True, 
                          save_results=True, plot_results=True,
                          error_correction=False):
    """
    Run the complete quantum gravitational wave detection pipeline
    
    Args:
        event_name: Name of gravitational wave event
        n_qubits: Number of qubits to use
        entanglement_type: Type of entanglement ('linear', 'star', or 'full')
        downsample_factor: Factor to downsample data
        scale_factor: Scale factor to amplify strain values
        use_gpu: Whether to use GPU acceleration
        save_results: Whether to save results to file
        plot_results: Whether to visualize results
        error_correction: Whether to use error correction
        
    Returns:
        dict: Results dictionary
    """
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*80)
    print(f" QUANTUM GRAVITATIONAL WAVE DETECTOR - Run {timestamp} ")
    print("="*80)
    print(f"Event: {event_name}")
    print(f"Qubits: {n_qubits}")
    print(f"Entanglement: {entanglement_type}")
    print(f"Error correction: {error_correction}")
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
        # Make sure visualization directory exists
        os.makedirs("data/visualization", exist_ok=True)
    
    # 1. Fetch LIGO data
    print("Fetching LIGO data...")
    start_time = time.time()
    times, strain, sample_rate = fetch_gw_data(event_name)
    
    if times is None or strain is None:
        print("Failed to fetch data. Exiting.")
        return None
    
    fetch_time = time.time() - start_time
    print(f"Data fetched in {fetch_time:.2f} seconds")
    
    # 2. Preprocess data
    print("\nPreprocessing data...")
    proc_times, proc_strain = preprocess_gw_data(times, strain, sample_rate)
    
    # Plot raw and processed data if requested
    if plot_results:
        visualizer.plot_strain_data(
            times, strain, 
            event_name=event_name,
            processed=(proc_times, proc_strain),
            save_filename=f"{event_name}_strain_data.png",
            display=False  # Don't display the plot
        )
    
    # 3. Create quantum detector with or without error correction
    print("\nInitializing quantum detector...")
    if error_correction:
        # Use error-corrected detector
            detector = ErrorCorrectedQuantumGWDetector(
            n_logical_qubits=n_qubits,
            entanglement_type=entanglement_type,
            use_gpu=use_gpu
        )
    else:
        # Use standard detector with ZX options
            detector = QuantumGWDetector(
            n_qubits=n_qubits,
            entanglement_type=entanglement_type,
            use_gpu=use_gpu,
            use_zx_opt=args.use_zx_opt if hasattr(args, 'use_zx_opt') else False,
            zx_opt_level=args.zx_opt_level if hasattr(args, 'zx_opt_level') else 1
    )
    
    # 4. Process through quantum circuit
    print("\nProcessing gravitational wave data through quantum circuit...")
    start_time = time.time()
    ds_times, ds_strain, quantum_states = detector.process_gw_data(
        proc_times, proc_strain, 
        downsample_factor=downsample_factor,
        scale_factor=scale_factor
    )
    process_time = time.time() - start_time
    print(f"Quantum processing completed in {process_time:.2f} seconds")
    
    # 5. Calculate quantum Fisher information
    print("\nCalculating detection metrics...")
    qfi = detector.calculate_qfi(quantum_states, ds_times)
    
    # Calculate additional detection metric
    detection_metric = detector.calculate_detection_metric(quantum_states)
    
    # 6. Visualize results if requested
    if plot_results:
        print("\nGenerating visualizations...")
        ec_tag = "ec_" if error_correction else ""  # Add tag for error correction
        detection_stats = visualizer.plot_quantum_detection(
            ds_times, ds_strain, quantum_states,
            qfi=qfi, detection_metric=detection_metric,
            event_name=f"{event_name} {'(with error correction)' if error_correction else ''}",
            n_qubits=n_qubits,
            save_filename=f"{event_name}_{n_qubits}qubits_{ec_tag}{entanglement_type}.png",
            display=False  # Don't display the plot
        )
    else:
        # Calculate stats without plotting
        max_qfi = np.max(qfi)
        mean_qfi = np.mean(qfi)
        std_qfi = np.std(qfi)
        qfi_snr = max_qfi / std_qfi if std_qfi > 0 else 0
        
        detection_stats = {
            "max_qfi": max_qfi,
            "mean_qfi": mean_qfi,
            "std_qfi": std_qfi,
            "qfi_snr": qfi_snr
        }
    
    # 7. Save results if requested
    if save_results:
        # Create filename with EC tag if using error correction
        ec_tag = "ec_" if error_correction else ""
        result_filename = f"{results_dir}/{event_name}_{n_qubits}qubits_{ec_tag}{entanglement_type}_{timestamp}"
        
        # Save quantum states and metrics as NPZ
        np.savez(
            result_filename + ".npz",
            times=ds_times,
            strain=ds_strain,
            quantum_states=quantum_states,
            qfi=qfi,
            detection_metric=detection_metric
        )
        
        # Save metadata and stats as JSON
        metadata = {
            "event_name": event_name,
            "n_qubits": n_qubits,
            "entanglement_type": entanglement_type,
            "error_correction": error_correction,
            "downsample_factor": downsample_factor,
            "scale_factor": float(scale_factor),
            "sample_rate": float(sample_rate) if sample_rate else None,
            "original_samples": len(times) if times is not None else 0,
            "processed_samples": len(ds_times) if ds_times is not None else 0,
            "execution_time": process_time,
            "timestamp": timestamp,
            "stats": detection_stats
        }
        
        with open(result_filename + ".json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  {result_filename}.npz")
        print(f"  {result_filename}.json")
    
    # 8. Return results dictionary
    results = {
        "event_name": event_name,
        "n_qubits": n_qubits,
        "entanglement_type": entanglement_type,
        "error_correction": error_correction,
        "downsample_factor": downsample_factor,
        "scale_factor": scale_factor,
        "times": ds_times,
        "strain": ds_strain,
        "quantum_states": quantum_states,
        "qfi": qfi,
        "detection_metric": detection_metric,
        "stats": detection_stats,
        "execution_time": process_time
    }
    
    print("\n" + "="*80)
    print(" DETECTION RESULTS ")
    print("="*80)
    print(f"Event: {event_name}")
    print(f"Detection configuration: {n_qubits} qubits, {entanglement_type} entanglement")
    print(f"Error correction: {error_correction}")
    print(f"Max QFI: {detection_stats['max_qfi']:.4f}")
    print(f"QFI SNR: {detection_stats['qfi_snr']:.4f}")
    print(f"Processing time: {process_time:.2f} seconds")
    print("="*80 + "\n")
    
    return results

def run_error_correction_comparison(event_name="GW150914", n_qubits=6,
                                   entanglement_type="star", downsample_factor=100,
                                   use_gpu=True, save_results=True, plot_results=True):
    """
    Run comparison between standard and error-corrected quantum detection
    
    Args:
        event_name: Name of gravitational wave event
        n_qubits: Number of qubits to use
        entanglement_type: Type of entanglement ('linear', 'star', or 'full')
        downsample_factor: Factor to downsample data
        use_gpu: Whether to use GPU acceleration
        save_results: Whether to save results to file
        plot_results: Whether to visualize results
        
    Returns:
        dict: Results dictionary for both runs
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*80)
    print(f" QUANTUM GRAVITATIONAL WAVE DETECTOR - Error Correction Comparison {timestamp} ")
    print("="*80)
    print(f"Event: {event_name}")
    print(f"Qubits: {n_qubits}")
    print(f"Entanglement: {entanglement_type}")
    print(f"Downsample factor: {downsample_factor}")
    print(f"GPU acceleration: {use_gpu}")
    print("="*80 + "\n")
    
    # Create visualizer if needed
    if plot_results:
        visualizer = GWDetectionVisualizer(save_dir="data/visualization")
        os.makedirs("data/visualization", exist_ok=True)
    
    # Create results directory if needed
    if save_results:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
    
    # Store results for both runs
    comparison_results = {}
    
    # Run standard detector
    print("\n=== Running standard quantum detector ===\n")
    standard_results = run_quantum_gw_detector(
        event_name=event_name,
        n_qubits=n_qubits,
        entanglement_type=entanglement_type,
        downsample_factor=downsample_factor,
        use_gpu=use_gpu,
        save_results=save_results,
        plot_results=plot_results,
        error_correction=False
    )
    
    if standard_results:
        comparison_results['standard'] = standard_results
    
    # Run error-corrected detector
    print("\n=== Running error-corrected quantum detector ===\n")
    ec_results = run_quantum_gw_detector(
        event_name=event_name,
        n_qubits=n_qubits,
        entanglement_type=entanglement_type,
        downsample_factor=downsample_factor,
        use_gpu=use_gpu,
        save_results=save_results,
        plot_results=plot_results,
        error_correction=True
    )
    
    if ec_results:
        comparison_results['error_corrected'] = ec_results
    
    # Create comparison visualization
    if plot_results and len(comparison_results) == 2:
        print("\nGenerating comparison visualization...")
        
        # Set up figure
        plt.figure(figsize=(12, 12))
        
        # Plot strain data
        plt.subplot(3, 1, 1)
        times = standard_results['times']
        strain = standard_results['strain']
        plt.plot(times, strain * 1e21, 'k-')
        plt.title(f"Gravitational Wave Strain: {event_name}")
        plt.ylabel("Strain (×10²¹)")
        plt.grid(True)
        
        # Plot QFI comparison
        plt.subplot(3, 1, 2)
        
        # Standard QFI
        std_qfi = standard_results['qfi']
        std_qfi_times = standard_results['times'][:-1]
        std_snr = standard_results['stats']['qfi_snr']
        
        # Error-corrected QFI
        ec_qfi = ec_results['qfi']
        ec_qfi_times = ec_results['times'][:-1]
        ec_snr = ec_results['stats']['qfi_snr']
        
        # Plot both QFI curves
        plt.plot(std_qfi_times, std_qfi, 'b-', label=f"Standard (SNR: {std_snr:.2f})")
        plt.plot(ec_qfi_times, ec_qfi, 'r-', label=f"Error-corrected (SNR: {ec_snr:.2f})")
        
        plt.title("QFI Comparison: Standard vs. Error-Corrected")
        plt.ylabel("Quantum Fisher Information")
        plt.legend()
        plt.grid(True)
        
        # Plot detection metric comparison
        plt.subplot(3, 1, 3)
        
        # Standard detection metric
        std_metric = standard_results['detection_metric']
        
        # Error-corrected detection metric
        ec_metric = ec_results['detection_metric']
        
        # Plot both metric curves
        plt.plot(times, std_metric, 'b-', label="Standard")
        plt.plot(times, ec_metric, 'r-', label="Error-corrected")
        
        plt.title("Detection Metric Comparison")
        plt.xlabel("Time (s)")
        plt.ylabel("Detection Metric")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the comparison plot
        save_path = f"data/visualization/{event_name}_error_correction_comparison_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison visualization saved to {save_path}")
    
    # Save comparison results if requested
    if save_results and len(comparison_results) == 2:
        # Create comparison metadata
        comparison_metadata = {
            "event_name": event_name,
            "n_qubits": n_qubits,
            "entanglement_type": entanglement_type,
            "downsample_factor": downsample_factor,
            "timestamp": timestamp,
            "standard": {
                "execution_time": standard_results["execution_time"],
                "stats": standard_results["stats"]
            },
            "error_corrected": {
                "execution_time": ec_results["execution_time"],
                "stats": ec_results["stats"]
            },
            "improvements": {
                "qfi_snr": ec_results["stats"]["qfi_snr"] / standard_results["stats"]["qfi_snr"] if standard_results["stats"]["qfi_snr"] > 0 else 0,
                "max_qfi": ec_results["stats"]["max_qfi"] / standard_results["stats"]["max_qfi"] if standard_results["stats"]["max_qfi"] > 0 else 0
            }
        }
        
        # Save as JSON
        result_filename = f"{results_dir}/{event_name}_error_correction_comparison_{timestamp}.json"
        with open(result_filename, 'w') as f:
            json.dump(comparison_metadata, f, indent=2)
        
        print(f"\nComparison results saved to:")
        print(f"  {result_filename}")
    
    # Print comparison summary
    if len(comparison_results) == 2:
        std_snr = standard_results['stats']['qfi_snr']
        ec_snr = ec_results['stats']['qfi_snr']
        improvement = ec_snr / std_snr if std_snr > 0 else 0
        
        print("\n" + "="*80)
        print(" ERROR CORRECTION COMPARISON RESULTS ")
        print("="*80)
        print(f"Event: {event_name}")
        print(f"Configuration: {n_qubits} qubits, {entanglement_type} entanglement")
        print(f"Standard QFI SNR: {std_snr:.4f}")
        print(f"Error-corrected QFI SNR: {ec_snr:.4f}")
        print(f"Improvement factor: {improvement:.2f}x")
        print(f"Standard execution time: {standard_results['execution_time']:.2f} seconds")
        print(f"Error-corrected execution time: {ec_results['execution_time']:.2f} seconds")
        print(f"Execution time overhead: {ec_results['execution_time']/standard_results['execution_time']:.2f}x")
        print("="*80 + "\n")
    
    return comparison_results

def run_comparative_analysis(event_name="GW150914", analysis_type="entanglement",
                           n_qubits=8, downsample_factor=100, use_gpu=True,
                           save_results=True, plot_results=True):
    """
    Run a comparative analysis varying either entanglement type or qubit count
    
    Args:
        event_name: Name of gravitational wave event
        analysis_type: Type of analysis ('entanglement' or 'qubits')
        n_qubits: Base number of qubits (for entanglement analysis)
        downsample_factor: Factor to downsample data
        use_gpu: Whether to use GPU acceleration
        save_results: Whether to save results to file
        plot_results: Whether to visualize results
        
    Returns:
        dict: Results dictionary for all runs
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*80)
    print(f" QUANTUM GRAVITATIONAL WAVE DETECTOR - Comparative Analysis {timestamp} ")
    print("="*80)
    print(f"Event: {event_name}")
    print(f"Analysis type: {analysis_type}")
    if analysis_type == "entanglement":
        print(f"Qubits: {n_qubits}")
        print("Comparing: linear, star, and full entanglement")
    else:  # "qubits"
        print("Comparing different qubit counts")
    print(f"Downsample factor: {downsample_factor}")
    print(f"GPU acceleration: {use_gpu}")
    print("="*80 + "\n")
    
    # Create visualizer if needed
    if plot_results:
        visualizer = GWDetectionVisualizer(save_dir="data/visualization")
        # Make sure visualization directory exists
        os.makedirs("data/visualization", exist_ok=True)
    
    # Create results directory if needed
    if save_results:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
    
    # Store all results
    all_results = {}
    
    # Run appropriate type of analysis
    if analysis_type == "entanglement":
        # Test different entanglement topologies
        entanglement_types = ["linear", "star", "full"]
        
        for entanglement_type in entanglement_types:
            print(f"\n--- Running with {entanglement_type} entanglement ---\n")
            results = run_quantum_gw_detector(
                event_name=event_name,
                n_qubits=n_qubits,
                entanglement_type=entanglement_type,
                downsample_factor=downsample_factor,
                use_gpu=use_gpu,
                save_results=save_results,
                plot_results=False  # Skip individual plots
            )
            
            if results:
                all_results[entanglement_type] = results
        
        # Create comparison plot if requested
        if plot_results and len(all_results) > 0:
            visualizer.plot_entanglement_comparison(
                all_results, event_name=event_name,
                save_filename=f"{event_name}_entanglement_comparison_{timestamp}.png",
                display=False  # Don't display the plot
            )
        
    else:  # "qubits"
        # Test different qubit counts
        qubit_counts = [4, 6, 8, 10]  # Adjust based on available resources
        
        for n_qubits in qubit_counts:
            print(f"\n--- Running with {n_qubits} qubits ---\n")
            results = run_quantum_gw_detector(
                event_name=event_name,
                n_qubits=n_qubits,
                entanglement_type="star",  # Use star as default
                downsample_factor=downsample_factor,
                use_gpu=use_gpu,
                save_results=save_results,
                plot_results=False  # Skip individual plots
            )
            
            if results:
                all_results[str(n_qubits)] = results
        
        # Create scaling plots if requested
        if plot_results and len(all_results) > 0:
            # Plot QFI SNR scaling
            visualizer.plot_qubit_scaling(
                all_results, metric="qfi_snr", event_name=event_name,
                save_filename=f"{event_name}_qubit_scaling_snr_{timestamp}.png",
                display=False  # Don't display the plot
            )
            
            # Plot execution time scaling
            visualizer.plot_qubit_scaling(
                all_results, metric="time", event_name=event_name,
                save_filename=f"{event_name}_qubit_scaling_time_{timestamp}.png",
                display=False  # Don't display the plot
            )
    
    # Save combined results if requested
    if save_results:
        # Create combined results metadata
        metadata = {
            "event_name": event_name,
            "analysis_type": analysis_type,
            "downsample_factor": downsample_factor,
            "use_gpu": use_gpu,
            "timestamp": timestamp,
            "results": {}
        }
        
        # Add stats from each run
        for key, result in all_results.items():
            metadata["results"][key] = {
                "n_qubits": result["n_qubits"],
                "entanglement_type": result["entanglement_type"],
                "stats": result["stats"],
                "execution_time": result["execution_time"]
            }
        
        # Save as JSON
        result_filename = f"{results_dir}/{event_name}_{analysis_type}_analysis_{timestamp}.json"
        with open(result_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nCombined analysis results saved to:")
        print(f"  {result_filename}")
    
    print("\n" + "="*80)
    print(" COMPARATIVE ANALYSIS COMPLETE ")
    print("="*80 + "\n")
    
    return all_results

def main():
    """Command-line interface for the quantum GW detector"""
    parser = argparse.ArgumentParser(
        description="Quantum Gravitational Wave Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Analysis mode selection
    parser.add_argument("--mode", default="single", 
                      choices=["single", "entanglement", "qubits", "errorcorrection"],
                      help="Analysis mode: single run, entanglement comparison, qubit scaling, or error correction comparison")
    
    # Event selection
    parser.add_argument("--event", default="GW150914", 
                      help="Gravitational wave event name")
    
    # Detector parameters
    parser.add_argument("--qubits", type=int, default=8, 
                      help="Number of qubits")
    parser.add_argument("--entanglement", default="star", 
                      choices=["linear", "star", "full"], 
                      help="Entanglement topology")
    parser.add_argument("--downsample", type=int, default=100, 
                      help="Downsample factor for data")
    parser.add_argument("--scale", type=float, default=1e21, 
                      help="Scale factor for strain amplification")
    
    # Error correction option
    parser.add_argument("--ec", action="store_true",
                      help="Enable error correction for single mode")
    
    # Hardware options
    parser.add_argument("--cpu", action="store_true", 
                      help="Disable GPU acceleration (use CPU)")
    
    # Output options
    parser.add_argument("--no-save", action="store_true", 
                      help="Don't save results")
    parser.add_argument("--no-plot", action="store_true", 
                      help="Don't generate plots")
    
    # ZX Optimization options
    parser.add_argument("--use-zx-opt", action="store_true",
                    help="Enable ZX-calculus circuit optimization")
    parser.add_argument("--zx-opt-level", type=int, choices=[1, 2, 3], default=1,
                    help="ZX optimization level (1=basic, 2=standard, 3=aggressive)")
    
    args = parser.parse_args()
    
    # Run the appropriate analysis
    if args.mode == "single":
        # Run single detector with specified parameters
        run_quantum_gw_detector(
            event_name=args.event,
            n_qubits=args.qubits,
            entanglement_type=args.entanglement,
            downsample_factor=args.downsample,
            scale_factor=args.scale,
            use_gpu=not args.cpu,
            save_results=not args.no_save,
            plot_results=not args.no_plot,
            error_correction=args.ec  # Pass the EC flag
        )
    elif args.mode == "errorcorrection":
        # Run error correction comparison
        run_error_correction_comparison(
            event_name=args.event,
            n_qubits=args.qubits,
            entanglement_type=args.entanglement,
            downsample_factor=args.downsample,
            use_gpu=not args.cpu,
            save_results=not args.no_save,
            plot_results=not args.no_plot
        )
    else:
        # Run entanglement or qubit comparison
        run_comparative_analysis(
            event_name=args.event,
            analysis_type="entanglement" if args.mode == "entanglement" else "qubits",
            n_qubits=args.qubits,
            downsample_factor=args.downsample,
            use_gpu=not args.cpu,
            save_results=not args.no_save,
            plot_results=not args.no_plot
        )

if __name__ == "__main__":
    main()
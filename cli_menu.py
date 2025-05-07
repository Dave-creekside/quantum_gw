# qgw_detector/cli_menu.py
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from tabulate import tabulate
import subprocess

# Import our API
from qgw_detector.api import QuantumGWAPI

# Try to import PyZX if available
try:
    import pyzx
    PYZX_AVAILABLE = True
except ImportError:
    PYZX_AVAILABLE = False

class QuantumGWMenu:
    """Interactive CLI menu for quantum gravitational wave detector experiments"""
    
    def __init__(self):
        """Initialize the menu system"""
        # Create an instance of the API
        self.api = QuantumGWAPI()
        
        # Store current result for certain operations
        self.current_result = None
    
    def main_menu(self):
        """Display the main menu"""
        while True:
            self._clear_screen()
            print("\n" + "="*72)
            print("  QUANTUM GRAVITATIONAL WAVE DETECTOR - EXPERIMENTAL WORKBENCH")
            print("=" * 72)
            
            # Get current config from API
            config = self.api.get_config()
            
            print("\nCurrent Settings:")
            print(f"  Event: {config['event_name']}")
            print(f"  Downsample factor: {config['downsample_factor']}")
            print(f"  Scale factor: {config['scale_factor']:.2e}")
            print(f"  GPU acceleration: {config['use_gpu']}")
            print(f"  ZX-calculus optimization: {config['use_zx_opt']}")
            if config['use_zx_opt']:
                print(f"  ZX optimization level: {config['zx_opt_level']}")
            
            print("\nMain Menu:")
            print("  1. Run Preset Pipeline")
            print("  2. Create Custom Pipeline")
            print("  3. Compare Multiple Pipelines")
            print("  4. Perform Parameter Sweep")
            print("  5. View Previous Results")
            print("  6. Change Settings")
            print("  7. Advanced Tools")
            print("  8. Exit")
            
            choice = input("\nEnter your choice (1-8): ")
            
            if choice == '1':
                self.run_preset_pipeline()
            elif choice == '2':
                self.create_custom_pipeline()
            elif choice == '3':
                self.compare_pipelines()
            elif choice == '4':
                self.parameter_sweep()
            elif choice == '5':
                self.view_results()
            elif choice == '6':
                self.change_settings()
            elif choice == '7':
                self.advanced_tools()
            elif choice == '8':
                print("\nExiting Quantum GW Detector Workbench. Goodbye!")
                sys.exit(0)
            else:
                input("Invalid selection. Press Enter to continue...")
    
    def run_preset_pipeline(self):
        """Run a preset pipeline configuration"""
        self._clear_screen()
        print("\n" + "="*72)
        print("  RUN PRESET PIPELINE")
        print("=" * 72)
        
        # Get presets from the API
        presets = self.api.list_presets()
        
        print("\nAvailable Preset Pipelines:")
        for i, (name, config) in enumerate(presets.items(), 1):
            pipeline_str = " → ".join([f"{q}-{t}" for q, t in config])
            print(f"  {i}. {name}: {pipeline_str}")
        
        print("\n  0. Return to Main Menu")
        
        choice = input("\nEnter your choice (0-{}): ".format(len(presets)))
        
        if choice == '0':
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(presets):
                name = list(presets.keys())[idx]
                config = presets[name]
                
                print(f"\nRunning preset pipeline: {name}")
                
                # Use the API to run the pipeline
                self.current_result = self.api.run_pipeline(config)
                self._view_pipeline_result(self.current_result)
                
                input("\nPress Enter to continue...")
            else:
                input("Invalid selection. Press Enter to continue...")
        except ValueError:
            input("Invalid input. Press Enter to continue...")
    
    def create_custom_pipeline(self):
        """Create and run a custom pipeline configuration"""
        self._clear_screen()
        print("\n" + "="*72)
        print("  CREATE CUSTOM PIPELINE")
        print("=" * 72)
        
        # Initialize empty pipeline
        pipeline = []
        
        print("\nBuilding custom pipeline. Enter 'done' when finished.")
        stage = 1
        
        while True:
            print(f"\nStage {stage}:")
            print("  Available qubit counts: 4, 6, 8")
            print("  Available topologies: star, linear, full")
            
            # Warn about 8-qubit full topology
            if stage == 1:
                print("\n  Note: 8-qubit full topology may exceed computational limits")
            
            qubits_input = input("  Enter qubit count (or 'done' to finish, 'cancel' to abort): ")
            
            if qubits_input.lower() == 'done':
                if len(pipeline) > 0:
                    break
                else:
                    print("  Pipeline must have at least one stage!")
                    continue
            
            if qubits_input.lower() == 'cancel':
                return
            
            try:
                qubits = int(qubits_input)
                if qubits not in [4, 6, 8]:
                    print("  Invalid qubit count! Must be 4, 6, or, 8")
                    continue
                
                topology = input("  Enter topology (star, linear, full): ").lower()
                if topology not in ['star', 'linear', 'full']:
                    print("  Invalid topology! Must be star, linear, or full")
                    continue
                
                # Extra warning for 8-qubit full topology
                if qubits == 8 and topology == 'full':
                    confirm = input("  Warning: 8-qubit full topology may crash. Proceed? (y/n): ")
                    if confirm.lower() != 'y':
                        continue
                
                # Add to pipeline
                pipeline.append((qubits, topology))
                stage += 1
                
                # Show current pipeline
                print("\n  Current pipeline:")
                pipeline_str = " → ".join([f"{q}-{t}" for q, t in pipeline])
                print(f"  {pipeline_str}")
                
            except ValueError:
                print("  Invalid input! Qubit count must be a number")
        
        # Run the custom pipeline
        print(f"\nRunning custom pipeline with {len(pipeline)} stages...")
        
        # Use the API to run the pipeline
        self.current_result = self.api.run_pipeline(pipeline)
        self._view_pipeline_result(self.current_result)
        
        # Ask if user wants to save this as a preset
        save_preset = input("\nSave this pipeline as a preset? (y/n): ")
        if save_preset.lower() == 'y':
            preset_name = input("Enter a name for this preset: ")
            try:
                if preset_name:
                    # Use the API to add a preset
                    result = self.api.add_preset(preset_name, pipeline)
                    print(f"Preset '{preset_name}' saved!")
            except ValueError as e:
                print(f"Error: {e}")
        
        input("\nPress Enter to continue...")
    
    def compare_pipelines(self):
        """Compare multiple pipeline configurations"""
        self._clear_screen()
        print("\n" + "="*72)
        print("  COMPARE MULTIPLE PIPELINES")
        print("=" * 72)
        
        # Get presets from the API
        presets = self.api.list_presets()
        
        # Select pipelines to compare
        pipelines_to_compare = []
        pipelines_configs = []
        pipelines_names = []
        
        print("\nSelect pipelines to compare (max 5):")
        print("  Available preset pipelines:")
        for i, (name, config) in enumerate(presets.items(), 1):
            pipeline_str = " → ".join([f"{q}-{t}" for q, t in config])
            print(f"  {i}. {name}: {pipeline_str}")
        
        print("\nEnter pipeline numbers separated by commas (e.g., 1,3,4)")
        print("  Or enter 'custom' to add a custom pipeline")
        print("  Or enter 'done' when finished")
        
        while len(pipelines_to_compare) < 5:
            choice = input("\nSelection: ")
            
            if choice.lower() == 'done':
                if len(pipelines_to_compare) > 0:
                    break
                else:
                    print("  You must select at least one pipeline!")
                    continue
            
            if choice.lower() == 'custom':
                # Create a custom pipeline
                print("\nBuilding custom pipeline for comparison:")
                pipeline = []
                stage = 1
                
                while True:
                    print(f"  Stage {stage}:")
                    qubits_input = input("    Enter qubit count (or 'done' to finish): ")
                    
                    if qubits_input.lower() == 'done':
                        if len(pipeline) > 0:
                            break
                        else:
                            print("    Pipeline must have at least one stage!")
                            continue
                    
                    try:
                        qubits = int(qubits_input)
                        if qubits not in [4, 6, 8]:
                            print("    Invalid qubit count! Must be 4, 6, or 8")
                            continue
                        
                        topology = input("    Enter topology (star, linear, full): ").lower()
                        if topology not in ['star', 'linear', 'full']:
                            print("    Invalid topology! Must be star, linear, or full")
                            continue
                        
                        # Add to pipeline
                        pipeline.append((qubits, topology))
                        stage += 1
                        
                    except ValueError:
                        print("    Invalid input! Qubit count must be a number")
                
                # Add the custom pipeline
                pipeline_str = " → ".join([f"{q}-{t}" for q, t in pipeline])
                pipelines_to_compare.append({
                    'name': f"Custom {len(pipelines_to_compare) + 1}",
                    'config': pipeline
                })
                print(f"  Added custom pipeline: {pipeline_str}")
                
            else:
                try:
                    indices = [int(idx.strip()) - 1 for idx in choice.split(',')]
                    for idx in indices:
                        if 0 <= idx < len(self.preset_pipelines):
                            name = list(self.preset_pipelines.keys())[idx]
                            config = self.preset_pipelines[name]
                            
                            # Check if already added
                            if any(p['name'] == name for p in pipelines_to_compare):
                                print(f"  Pipeline '{name}' already selected!")
                                continue
                            
                            pipelines_to_compare.append({
                                'name': name,
                                'config': config
                            })
                            
                            if len(pipelines_to_compare) >= 5:
                                print("  Maximum number of pipelines selected (5)")
                                break
                        else:
                            print(f"  Invalid pipeline index: {idx + 1}")
                except ValueError:
                    print("  Invalid input! Please enter numbers separated by commas")
        
        if not pipelines_to_compare:
            input("No pipelines selected. Press Enter to continue...")
            return
        
        # Run comparison
        if pipelines_to_compare:
            print("\nRunning comparison of selected pipelines...")
            
            # Extract configs and names for API call
            for pipeline in pipelines_to_compare:
                pipelines_configs.append(pipeline['config'])
                pipelines_names.append(pipeline['name'])
            
            # Use the API to run the comparison
            comparison_result = self.api.compare_pipelines(pipelines_configs, pipelines_names)
            
            # Display comparison
            self._display_comparison_results(comparison_result)
        
        input("\nPress Enter to continue...")
    
    def parameter_sweep(self):
        """Perform a parameter sweep"""
        self._clear_screen()
        print("\n" + "="*72)
        print("  PARAMETER SWEEP")
        print("=" * 72)
        
        print("\nSelect parameter to sweep:")
        print("  1. Qubit count")
        print("  2. Topology")
        print("  3. Strain scale factor")
        print("  4. Return to main menu")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            self._sweep_qubit_count()
        elif choice == '2':
            self._sweep_topology()
        elif choice == '3':
            self._sweep_scale_factor()
        elif choice == '4':
            return
        else:
            input("Invalid selection. Press Enter to continue...")
    
    def view_results(self):
        """View previously saved results"""
        self._clear_screen()
        print("\n" + "="*72)
        print("  VIEW PREVIOUS RESULTS")
        print("=" * 72)
        
        # Use the API to list saved results
        result_files = self.api.list_saved_results()
        
        if not result_files:
            input("\nNo saved results found. Press Enter to continue...")
            return
        
        # Display available results
        print("\nAvailable results:")
        for i, result_info in enumerate(result_files, 1):
            # Display result information
            timestamp = result_info['timestamp']
            pipeline_name = result_info['pipeline_name']
            print(f"  {i}. [{timestamp}] {pipeline_name}")
        
        print("\n  0. Return to Main Menu")
        
        choice = input("\nEnter your choice (0-{}): ".format(len(result_files)))
        
        if choice == '0':
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(result_files):
                result_info = result_files[idx]
                
                # Use the API to get the saved result
                result = self.api.get_saved_result(result_info['identifier'])
                
                # Display the result
                self._view_pipeline_result(result)
                
                input("\nPress Enter to continue...")
            else:
                input("Invalid selection. Press Enter to continue...")
        except ValueError:
            input("Invalid input. Press Enter to continue...")
    
    def change_settings(self):
        """Change global settings"""
        self._clear_screen()
        print("\n" + "="*72)
        print("  CHANGE SETTINGS")
        print("=" * 72)
        
        # Get current config from API
        config = self.api.get_config()
        
        print("\nCurrent Settings:")
        print(f"  1. Event: {config['event_name']}")
        print(f"  2. Downsample factor: {config['downsample_factor']}")
        print(f"  3. Scale factor: {config['scale_factor']:.2e}")
        print(f"  4. GPU acceleration: {config['use_gpu']}")
        print(f"  5. ZX-calculus optimization: {config['use_zx_opt']}")
        if config['use_zx_opt']:
            print(f"  6. ZX optimization level: {config['zx_opt_level']}")
        
        print("\n  0. Return to Main Menu")
        
        choice = input("\nEnter setting to change (0-6): ")
        
        if choice == '0':
            return
        elif choice == '1':
            print("\nAvailable events:")
            print("  1. GW150914 (First detection)")
            print("  2. GW151226 (Boxing Day event)")
            print("  3. GW170104 (January 2017 event)")
            print("  4. GW170814 (First three-detector observation)")
            print("  5. GW170817 (Binary neutron star merger)")
            
            event_choice = input("Select event (1-5): ")
            events = ["GW150914", "GW151226", "GW170104", "GW170814", "GW170817"]
            
            try:
                idx = int(event_choice) - 1
                if 0 <= idx < len(events):
                    # Use the API to update the event
                    self.api.set_config(event_name=events[idx])
                    print(f"\nEvent changed to: {events[idx]}")
                else:
                    print("\nInvalid selection!")
            except ValueError:
                print("\nInvalid input!")
        
        elif choice == '2':
            try:
                new_value = int(input("\nEnter new downsample factor (recommended: 100-500): "))
                if new_value > 0:
                    # Use the API to update the downsample factor
                    self.api.set_config(downsample_factor=new_value)
                    print(f"\nDownsample factor set to: {new_value}")
                else:
                    print("\nValue must be positive!")
            except ValueError:
                print("\nInvalid input!")
        
        elif choice == '3':
            try:
                new_value = float(input("\nEnter new scale factor (scientific notation ok, e.g. 1e21): "))
                if new_value > 0:
                    # Use the API to update the scale factor
                    self.api.set_config(scale_factor=new_value)
                    print(f"\nScale factor set to: {new_value:.2e}")
                else:
                    print("\nValue must be positive!")
            except ValueError:
                print("\nInvalid input!")
        
        elif choice == '4':
            new_value = input("\nEnable GPU acceleration? (y/n): ").lower()
            # Use the API to update the GPU setting
            use_gpu = new_value == 'y'
            self.api.set_config(use_gpu=use_gpu)
            print(f"\nGPU acceleration: {use_gpu}")
        
        elif choice == '5':
            new_value = input("\nEnable ZX-calculus optimization? (y/n): ").lower()
            # Use the API to update the ZX optimization setting
            use_zx_opt = new_value == 'y'
            self.api.set_config(use_zx_opt=use_zx_opt)
            print(f"\nZX-calculus optimization: {use_zx_opt}")
            
            if use_zx_opt and not PYZX_AVAILABLE:
                print("\nWarning: PyZX not found! Please install with 'pip install pyzx'")
                self.api.set_config(use_zx_opt=False)
        
        elif choice == '6':
            # Get updated config to check if ZX is enabled
            config = self.api.get_config()
            if not config['use_zx_opt']:
                print("\nZX-calculus optimization is disabled!")
            else:
                try:
                    print("\nZX optimization levels:")
                    print("  1: Basic Clifford optimization")
                    print("  2: Full reductions")
                    print("  3: Aggressive optimization")
                    
                    new_value = int(input("\nEnter ZX optimization level (1-3): "))
                    if 1 <= new_value <= 3:
                        # Use the API to update the ZX optimization level
                        self.api.set_config(zx_opt_level=new_value)
                        print(f"\nZX optimization level set to: {new_value}")
                    else:
                        print("\nValue must be between 1 and 3!")
                except ValueError:
                    print("\nInvalid input!")
        
        input("\nPress Enter to continue...")
    
    def advanced_tools(self):
        """Advanced tools menu"""
        self._clear_screen()
        print("\n" + "="*72)
        print("  ADVANCED TOOLS")
        print("=" * 72)
        
        print("\nAvailable Tools:")
        print("  1. Raw Quantum State Analyzer")
        print("  2. Circuit Visualization")
        print("  3. Batch Export Results")
        print("  4. Noise Analysis")
        
        if PYZX_AVAILABLE:
            print("  5. ZX-Calculus Circuit Optimization")
        
        print("\n  0. Return to Main Menu")
        
        choice = input("\nEnter your choice (0-5): ")
        
        if choice == '0':
            return
        elif choice == '1':
            self._quantum_state_analyzer()
        elif choice == '2':
            self._circuit_visualization()
        elif choice == '3':
            self._batch_export()
        elif choice == '4':
            self._noise_analysis()
        elif choice == '5' and PYZX_AVAILABLE:
            self._zx_circuit_optimizer()
        else:
            input("Invalid selection. Press Enter to continue...")
    
    def _ensure_data_loaded(self):
        """Ensure LIGO data is loaded"""
        if self.data is None:
            print(f"\nLoading LIGO data for {self.event_name}...")
            times, strain, sample_rate = fetch_gw_data(self.event_name)
            proc_times, proc_strain = preprocess_gw_data(times, strain, sample_rate)
            
            self.data = {
                'times': proc_times,
                'strain': proc_strain,
                'sample_rate': sample_rate
            }
            
            print(f"Data loaded: {len(proc_times)} samples")
    
    def _run_pipeline(self, pipeline_config):
        """
        Run a pipeline configuration
        
        Args:
            pipeline_config: List of (qubit, topology) tuples
            
        Returns:
            dict: Pipeline results
        """
        pipeline_str = "_".join([f"{q}-{t}" for q, t in pipeline_config])
        
        # Check if already in cache
        cache_key = f"{self.event_name}_{pipeline_str}_{self.downsample_factor}_{self.scale_factor}"
        if cache_key in self.results_cache:
            print(f"Using cached results for {pipeline_str}")
            return self.results_cache[cache_key]
        
        print(f"\nRunning pipeline: {pipeline_str}")
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # For visualization
        all_stages_data = [{
            'name': 'LIGO',
            'times': self.data['times'],
            'data': self.data['strain'] * 1e21,
            'label': 'Strain (×10²¹)'
        }]
        
        # Store results for each stage
        pipeline_results = {
            'event_name': self.event_name,
            'pipeline_config': pipeline_str,
            'timestamp': timestamp,
            'stages': []
        }
        
        # Get initial data
        current_times = self.data['times']
        current_strain = self.data['strain']
        
        # Process through each stage
        for i, (n_qubits, topology) in enumerate(pipeline_config):
            print(f"\n=== Stage {i+1}: {n_qubits}-qubit {topology} detector ===")
            
            # Create detector
            detector = QuantumGWDetector(
                n_qubits=n_qubits,
                entanglement_type=topology,
                use_gpu=self.use_gpu,
                use_zx_opt=self.use_zx_opt,
                zx_opt_level=self.zx_opt_level
            )
            
            # Apply stage-specific downsample factor (only first stage)
            stage_downsample = self.downsample_factor if i == 0 else 1
            
            # Process data
            start_time = time.time()
            ds_times, ds_strain, quantum_states = detector.process_gw_data(
                current_times, current_strain, 
                downsample_factor=stage_downsample,
                scale_factor=self.scale_factor
            )
            process_time = time.time() - start_time
            
            # Calculate metrics
            qfi_values = detector.calculate_qfi(quantum_states, ds_times)
            detection_metric = detector.calculate_detection_metric(quantum_states)
            
            # Calculate SNR and other stats
            max_qfi = float(np.max(qfi_values))
            mean_qfi = float(np.mean(qfi_values))
            std_qfi = float(np.std(qfi_values))
            snr = max_qfi / std_qfi if std_qfi > 0 else 0
            
            # Store stage results
            stage_results = {
                'stage': i+1,
                'qubits': n_qubits,
                'topology': topology,
                'execution_time': process_time,
                'max_qfi': max_qfi,
                'mean_qfi': mean_qfi,
                'qfi_snr': snr,
                'zx_stats': detector.zx_stats if hasattr(detector, 'zx_stats') else {}
            }
            pipeline_results['stages'].append(stage_results)
            
            # Store data for visualization
            all_stages_data.append({
                'name': f"Stage {i+1}: {n_qubits}-{topology}",
                'times': ds_times,
                'data': detection_metric,
                'label': 'Detection Metric'
            })
            
            # Use this stage's output as input to the next stage
            current_times = ds_times
            current_strain = detection_metric
            
            print(f"Stage {i+1} complete:")
            print(f"  QFI SNR: {snr:.4f}")
            print(f"  Process time: {process_time:.2f} seconds")
        
        # Calculate pipeline-level metrics
        first_stage_snr = pipeline_results['stages'][0]['qfi_snr']
        final_stage_snr = pipeline_results['stages'][-1]['qfi_snr']
        improvement = final_stage_snr / first_stage_snr if first_stage_snr > 0 else 0
        
        pipeline_results['summary'] = {
            'first_stage_snr': first_stage_snr,
            'final_stage_snr': final_stage_snr,
            'improvement_factor': improvement,
            'total_execution_time': sum(stage['execution_time'] for stage in pipeline_results['stages'])
        }
        
        # Create visualization
        output_dir = f"data/experiments/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        fig_path = os.path.join(output_dir, f"pipeline_{pipeline_str}.png")
        
        # Create multi-panel plot
        n_stages = len(all_stages_data)
        plt.figure(figsize=(14, 4 * n_stages))
        
        for i, stage_data in enumerate(all_stages_data):
            plt.subplot(n_stages, 1, i+1)
            plt.plot(stage_data['times'], stage_data['data'])
            plt.title(stage_data['name'])
            plt.ylabel(stage_data['label'])
            plt.grid(True)
            
            if i == n_stages - 1:
                plt.xlabel("Time (s)")
        
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save results as JSON
        results_path = os.path.join(output_dir, f"results_{pipeline_str}.json")
        with open(results_path, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        print(f"\nPipeline results saved to: {results_path}")
        print(f"Pipeline visualization saved to: {fig_path}")
        
        # Store in cache
        self.results_cache[cache_key] = pipeline_results
        
        return pipeline_results
    
    def _view_pipeline_result(self, result):
        """Display a pipeline result"""
        self._clear_screen()
        print("\n" + "="*72)
        print("  PIPELINE RESULTS")
        print("=" * 72)
        
        print(f"\nEvent: {result['event_name']}")
        print(f"Pipeline: {result['pipeline_config']}")
        print(f"Timestamp: {result.get('timestamp', 'N/A')}")
        
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
        if 'summary' in result:
            summary = result['summary']
            print("\nSummary:")
            print(f"  First stage SNR: {summary['first_stage_snr']:.4f}")
            print(f"  Final stage SNR: {summary['final_stage_snr']:.4f}")
            print(f"  Improvement factor: {summary['improvement_factor']:.2f}x")
            print(f"  Total execution time: {summary['total_execution_time']:.2f} seconds")
        
        # Ask if user wants to view the visualization
        if 'timestamp' in result:
            view_viz = input("\nView visualization? (y/n): ")
            if view_viz.lower() == 'y':
                pipeline_str = result['pipeline_config']
                timestamp = result['timestamp']
                fig_path = f"data/experiments/{timestamp}/pipeline_{pipeline_str}.png"
                
                if os.path.exists(fig_path):
                    # Try to open the image with default viewer
                    try:
                        import subprocess
                        if sys.platform.startswith('darwin'):  # macOS
                            subprocess.call(('open', fig_path))
                        elif sys.platform.startswith('linux'):  # Linux
                            subprocess.call(('xdg-open', fig_path))
                        elif sys.platform.startswith('win'):  # Windows
                            subprocess.call(('start', fig_path), shell=True)
                        else:
                            print(f"\nVisualization available at: {fig_path}")
                    except:
                        print(f"\nVisualization available at: {fig_path}")
                else:
                    print(f"\nVisualization not found at: {fig_path}")
    
    def _compare_pipeline_results(self, results):
        """Compare multiple pipeline results"""
        self._clear_screen()
        print("\n" + "="*72)
        print("  PIPELINE COMPARISON")
        print("=" * 72)
        
        # Create comparison table
        headers = ["Pipeline", "Final SNR", "Improvement", "Time (s)"]
        table_data = []
        
        for result in results:
            table_data.append([
                result['name'],
                f"{result['summary']['final_stage_snr']:.4f}",
                f"{result['summary']['improvement_factor']:.2f}x",
                f"{result['summary']['total_execution_time']:.2f}"
            ])
        
        # Sort by final SNR
        table_data.sort(key=lambda x: float(x[1]), reverse=True)
        
        print("\nRanked by Final SNR:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Plot SNR comparison
        pipelines = [r['name'] for r in results]
        snrs = [r['summary']['final_stage_snr'] for r in results]
        improvements = [r['summary']['improvement_factor'] for r in results]
        
        # Sort by SNR
        sorted_indices = np.argsort(snrs)[::-1]
        pipelines = [pipelines[i] for i in sorted_indices]
        snrs = [snrs[i] for i in sorted_indices]
        improvements = [improvements[i] for i in sorted_indices]
        
        plt.subplot(1, 2, 1)
        plt.barh(range(len(pipelines)), snrs, color='skyblue')
        plt.yticks(range(len(pipelines)), pipelines)
        plt.xlabel('Final SNR')
        plt.title('SNR Comparison')
        plt.grid(True, axis='x')
        
        plt.subplot(1, 2, 2)
        plt.barh(range(len(pipelines)), improvements, color='lightgreen')
        plt.yticks(range(len(pipelines)), pipelines)
        plt.xlabel('Improvement Factor')
        plt.title('Improvement Comparison')
        plt.grid(True, axis='x')
        
        plt.tight_layout()
        
        # Save the comparison visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"data/experiments/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        fig_path = os.path.join(output_dir, "pipeline_comparison.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nComparison visualization saved to: {fig_path}")
        
        # Ask if user wants to view the visualization
        view_viz = input("\nView visualization? (y/n): ")
        if view_viz.lower() == 'y':
            # Try to open the image with default viewer
            try:
                import subprocess
                if sys.platform.startswith('darwin'):  # macOS
                    subprocess.call(('open', fig_path))
                elif sys.platform.startswith('linux'):  # Linux
                    subprocess.call(('xdg-open', fig_path))
                elif sys.platform.startswith('win'):  # Windows
                    subprocess.call(('start', fig_path), shell=True)
                else:
                    print(f"\nVisualization available at: {fig_path}")
            except:
                print(f"\nVisualization available at: {fig_path}")
    
    def _display_comparison_results(self, comparison_result):
        """Display comparison results from the API"""
        self._clear_screen()
        print("\n" + "="*72)
        print("  PIPELINE COMPARISON")
        print("=" * 72)
        
        # Create comparison table
        headers = ["Pipeline", "Final SNR", "Improvement", "Time (s)"]
        table_data = []
        
        for pipeline in comparison_result['pipelines']:
            table_data.append([
                pipeline['name'],
                f"{pipeline['final_snr']:.4f}",
                f"{pipeline['improvement']:.2f}x",
                f"{pipeline['execution_time']:.2f}"
            ])
        
        print("\nRanked by Final SNR:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Check if there's a visualization to show
        if 'file_paths' in comparison_result and 'visualization' in comparison_result['file_paths']:
            viz_path = comparison_result['file_paths']['visualization']
            print(f"\nComparison visualization saved to: {viz_path}")
            
            # Ask if user wants to view the visualization
            view_viz = input("\nView visualization? (y/n): ")
            if view_viz.lower() == 'y':
                # Try to open the image with default viewer
                try:
                    if sys.platform.startswith('darwin'):  # macOS
                        subprocess.call(('open', viz_path))
                    elif sys.platform.startswith('linux'):  # Linux
                        subprocess.call(('xdg-open', viz_path))
                    elif sys.platform.startswith('win'):  # Windows
                        subprocess.call(('start', viz_path), shell=True)
                    else:
                        print(f"\nVisualization available at: {viz_path}")
                except Exception:
                    print(f"\nVisualization available at: {viz_path}")
    
    def _sweep_qubit_count(self):
        """Perform a qubit count sweep"""
        self._clear_screen()
        print("\n" + "="*72)
        print("  QUBIT COUNT SWEEP")
        print("=" * 72)
        
        print("\nSelect topology for sweep:")
        print("  1. Star")
        print("  2. Linear")
        print("  3. Full")
        
        topology_choice = input("\nEnter your choice (1-3): ")
        
        if topology_choice == '1':
            topology = 'star'
        elif topology_choice == '2':
            topology = 'linear'
        elif topology_choice == '3':
            topology = 'full'
        else:
            input("Invalid selection. Press Enter to continue...")
            return
        
        print("\nSelect qubit counts to test:")
        print("  Available qubit counts: 4, 6, 8")
        print("  Note: 8-qubit full topology may exceed computational limits")
        
        qubits_input = input("\nEnter qubit counts separated by commas (e.g., 4,6,8): ")
        
        try:
            qubit_counts = [int(q.strip()) for q in qubits_input.split(',')]
            valid_counts = [q for q in qubit_counts if q in [4, 6, 8]]
            
            if not valid_counts:
                input("No valid qubit counts specified. Press Enter to continue...")
                return
            
            if 8 in valid_counts and topology == 'full':
                confirm = input("\nWarning: 8-qubit full topology may crash. Proceed? (y/n): ")
                if confirm.lower() != 'y':
                    valid_counts.remove(8)
                    if not valid_counts:
                        input("No valid qubit counts remaining. Press Enter to continue...")
                        return
            
            # Run sweep
            print(f"\nRunning qubit count sweep for {topology} topology...")
            
            # Set up base config params if needed
            base_config = None
            
            # Use the API to run the sweep
            results = self.api.sweep_qubit_count(topology, valid_counts, base_config)
            
            # Display results
            if isinstance(results, list) and results:
                # If it's a list of individual results, compare them
                self._compare_pipeline_results(results)
            else:
                # If it's already a comparison result
                print("\nSweep completed successfully!")
            
            input("\nPress Enter to continue...")
            
        except ValueError:
            input("Invalid input. Press Enter to continue...")
    
    def _sweep_topology(self):
        """Perform a topology sweep"""
        self._clear_screen()
        print("\n" + "="*72)
        print("  TOPOLOGY SWEEP")
        print("=" * 72)
        
        print("\nSelect qubit count for sweep:")
        print("  1. 4 qubits")
        print("  2. 6 qubits")
        print("  3. 8 qubits")
        
        qubits_choice = input("\nEnter your choice (1-3): ")
        
        if qubits_choice == '1':
            qubits = 4
        elif qubits_choice == '2':
            qubits = 6
        elif qubits_choice == '3':
            qubits = 8
        else:
            input("Invalid selection. Press Enter to continue...")
            return
        
        print("\nSelect topologies to test:")
        topologies = []
        
        print("  Include star topology? (y/n): ")
        if input().lower() == 'y':
            topologies.append('star')
        
        print("  Include linear topology? (y/n): ")
        if input().lower() == 'y':
            topologies.append('linear')
        
        print("  Include full topology? (y/n): ")
        if qubits == 8:
            confirm = input("  Warning: 8-qubit full topology may crash. Proceed? (y/n): ")
            if confirm.lower() == 'y':
                topologies.append('full')
        else:
            if input().lower() == 'y':
                topologies.append('full')
        
        if not topologies:
            input("No topologies selected. Press Enter to continue...")
            return
        
        # Run sweep
        print(f"\nRunning topology sweep for {qubits} qubits...")
        
        # Set up base config params if needed
        base_config = None
        
        # Use the API to run the sweep
        results = self.api.sweep_topology(qubits, topologies, base_config)
        
        # Display results
        if isinstance(results, list) and results:
            # If it's a list of individual results, compare them
            self._compare_pipeline_results(results)
        else:
            # If it's already a comparison result
            print("\nSweep completed successfully!")
        
        input("\nPress Enter to continue...")
    
    def _sweep_scale_factor(self):
        """Perform a scale factor sweep"""
        self._clear_screen()
        print("\n" + "="*72)
        print("  SCALE FACTOR SWEEP")
        print("=" * 72)
        
        print("\nSelect detector configuration:")
        print("  1. 4-qubit star")
        print("  2. 4-qubit linear")
        print("  3. 4-qubit full")
        print("  4. 6-qubit star")
        print("  5. 6-qubit linear")
        print("  6. 6-qubit full")
        print("  7. Custom configuration")
        
        config_choice = input("\nEnter your choice (1-7): ")
        
        if config_choice == '1':
            config = [(4, 'star')]
        elif config_choice == '2':
            config = [(4, 'linear')]
        elif config_choice == '3':
            config = [(4, 'full')]
        elif config_choice == '4':
            config = [(6, 'star')]
        elif config_choice == '5':
            config = [(6, 'linear')]
        elif config_choice == '6':
            config = [(6, 'full')]
        elif config_choice == '7':
            # Custom configuration
            print("\nBuilding custom configuration:")
            qubits_input = input("  Enter qubit count (4, 6, or 8): ")
            try:
                qubits = int(qubits_input)
                if qubits not in [4, 6, 8]:
                    input("  Invalid qubit count. Press Enter to continue...")
                    return
                
                topology = input("  Enter topology (star, linear, or full): ").lower()
                if topology not in ['star', 'linear', 'full']:
                    input("  Invalid topology. Press Enter to continue...")
                    return
                
                if qubits == 8 and topology == 'full':
                    confirm = input("  Warning: 8-qubit full topology may crash. Proceed? (y/n): ")
                    if confirm.lower() != 'y':
                        input("  Operation cancelled. Press Enter to continue...")
                        return
                
                config = [(qubits, topology)]
                
            except ValueError:
                input("  Invalid input. Press Enter to continue...")
                return
        else:
            input("Invalid selection. Press Enter to continue...")
            return
        
        print("\nSelect scale factors to test:")
        print("  Recommended range: 1e19 to 1e23")
        print("  Enter values in scientific notation separated by commas (e.g., 1e19,1e20,1e21)")
        
        factors_input = input("\nEnter scale factors: ")
        
        try:
            scale_factors = [float(f.strip()) for f in factors_input.split(',')]
            
            if not scale_factors:
                input("No scale factors specified. Press Enter to continue...")
                return
            
            # Run sweep
            print(f"\nRunning scale factor sweep for {config[0][0]}-qubit {config[0][1]} detector...")
            
            # Set up base config params if needed
            base_config = None
            
            # Use the API to run the sweep
            results = self.api.sweep_scale_factor(config, scale_factors, base_config)
            
            # Display results
            if isinstance(results, list) and results:
                # If it's a list of individual results, compare them
                self._compare_pipeline_results(results)
            else:
                # If it's already a comparison result
                print("\nSweep completed successfully!")
            
            input("\nPress Enter to continue...")
            
        except ValueError:
            input("Invalid input. Press Enter to continue...")
    
    def _quantum_state_analyzer(self):
        """Tool for analyzing quantum states"""
        self._clear_screen()
        print("\n" + "="*72)
        print("  QUANTUM STATE ANALYZER")
        print("=" * 72)
        
        # First, need a result to analyze
        if self.current_result is None:
            print("\nNo current result to analyze. Please run a pipeline first.")
            print("\n1. Run a preset pipeline")
            print("2. Return to Advanced Tools menu")
            
            choice = input("\nEnter your choice (1-2): ")
            
            if choice == '1':
                self.run_preset_pipeline()
                if self.current_result is None:
                    return
            else:
                return
        
        # Simplified version - implement details as needed
        print("\nQuantum State Analyzer functionality to be implemented.")
        print("This will allow detailed analysis of quantum states at each pipeline stage.")
        print("Stay tuned for this feature in a future update!")
        
        input("\nPress Enter to continue...")
    
    def _circuit_visualization(self):
        """Tool for visualizing quantum circuits"""
        self._clear_screen()
        print("\n" + "="*72)
        print("  CIRCUIT VISUALIZATION")
        print("=" * 72)
        
        # Simplified version - implement details as needed
        print("\nCircuit Visualization functionality to be implemented.")
        print("This will create detailed visualizations of the quantum circuits used.")
        print("Stay tuned for this feature in a future update!")
        
        input("\nPress Enter to continue...")
    
    def _batch_export(self):
        """Tool for batch exporting results"""
        self._clear_screen()
        print("\n" + "="*72)
        print("  BATCH EXPORT RESULTS")
        print("=" * 72)
        
        # Simplified version - implement details as needed
        print("\nBatch Export functionality to be implemented.")
        print("This will allow exporting multiple results to various formats.")
        print("Stay tuned for this feature in a future update!")
        
        input("\nPress Enter to continue...")
    
    def _noise_analysis(self):
        """Tool for noise analysis"""
        self._clear_screen()
        print("\n" + "="*72)
        print("  NOISE ANALYSIS")
        print("=" * 72)
        
        # Simplified version - implement details as needed
        print("\nNoise Analysis functionality to be implemented.")
        print("This will simulate detector performance with various noise models.")
        print("Stay tuned for this feature in a future update!")
        
        input("\nPress Enter to continue...")
    
    def _zx_circuit_optimizer(self):
        """Tool for ZX-calculus circuit optimization"""
        if not PYZX_AVAILABLE:
            input("\nPyZX not found! Please install with 'pip install pyzx'. Press Enter to continue...")
            return
        
        self._clear_screen()
        print("\n" + "="*72)
        print("  ZX-CALCULUS CIRCUIT OPTIMIZER")
        print("=" * 72)
        
        # Simplified version - implement details as needed
        print("\nZX-Calculus Optimizer functionality to be implemented.")
        print("This will provide detailed ZX-calculus optimization of quantum circuits.")
        print("Stay tuned for this feature in a future update!")
        
        input("\nPress Enter to continue...")
    
    def _clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """Main function to run the CLI menu"""
    menu = QuantumGWMenu()
    menu.main_menu()

if __name__ == "__main__":
    main()

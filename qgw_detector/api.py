# qgw_detector/api.py
import os
import json
import numpy as np
import time
from datetime import datetime

from qgw_detector.data.ligo import fetch_gw_data, preprocess_gw_data
from qgw_detector.quantum.circuits import QuantumGWDetector

class QuantumGWAPI:
    """API for the Quantum Gravitational Wave Detector"""
    
    def __init__(self):
        """Initialize the API"""
        # Configuration defaults
        self.config = {
            'event_name': 'GW150914',
            'downsample_factor': 200,
            'scale_factor': 1e21,
            'use_gpu': True,
            'use_zx_opt': False,
            'zx_opt_level': 1
        }
        
        # Internal state
        self.data_cache = {}
        self.results_cache = {}
        self.presets = {
            "best_3stage": [
                (4, "linear"), (4, "full"), (4, "linear")
            ],
            "best_2stage": [
                (4, "linear"), (4, "linear")
            ],
            "star_full_linear": [
                (4, "star"), (4, "full"), (4, "linear")
            ],
            "full_star": [
                (6, "full"), (6, "star")
            ],
            "star_linear": [
                (4, "star"), (4, "linear")
            ],
            "8qubit_pipeline": [
                (8, "star"), (6, "full"), (8, "linear")
            ]
        }
        
        # Create directories
        os.makedirs("data/experiments", exist_ok=True)
        self.results_dir = "data/experiments" # Define results directory for later use
    
    def get_config(self):
        """Get current configuration"""
        return self.config
    
    def set_config(self, **kwargs):
        """Update configuration settings"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
        return self.config
    
    def list_presets(self):
        """List available preset pipelines"""
        return self.presets

    def add_preset(self, name: str, config: list):
        """Add a new preset pipeline configuration."""
        if name in self.presets:
            raise ValueError(f"Preset with name '{name}' already exists.")
        if not isinstance(config, list) or not all(isinstance(stage, tuple) and len(stage) == 2 for stage in config):
            raise ValueError("Invalid pipeline configuration format. Expected list of (qubit, topology) tuples.")
        self.presets[name] = config
        return {"message": f"Preset '{name}' added successfully.", "presets": self.presets}
    
    def run_pipeline(self, pipeline_config, save_results=True, save_visualization=True):
        """
        Run a pipeline configuration
        
        Args:
            pipeline_config: List of (qubit, topology) tuples
            save_results: Whether to save results to disk
            save_visualization: Whether to generate and save visualization
            
        Returns:
            dict: Pipeline results
        """
        pipeline_str = "_".join([f"{q}-{t}" for q, t in pipeline_config])
        
        # Check if already in cache
        cache_key = f"{self.config['event_name']}_{pipeline_str}_{self.config['downsample_factor']}_{self.config['scale_factor']}"
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]
        
        # Ensure data is loaded
        self._ensure_data_loaded()
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize result storage
        pipeline_results = {
            'event_name': self.config['event_name'],
            'pipeline_config': pipeline_str,
            'timestamp': timestamp,
            'stages': []
        }
        
        # Get initial data
        current_times = self.data_cache[self.config['event_name']]['times']
        current_strain = self.data_cache[self.config['event_name']]['strain']
        
        # Process through each stage
        for i, (n_qubits, topology) in enumerate(pipeline_config):
            # Create detector
            detector = QuantumGWDetector(
                n_qubits=n_qubits,
                entanglement_type=topology,
                use_gpu=self.config['use_gpu'],
                use_zx_opt=self.config['use_zx_opt'],
                zx_opt_level=self.config['zx_opt_level']
            )
            
            # Apply stage-specific downsample factor (only first stage)
            stage_downsample = self.config['downsample_factor'] if i == 0 else 1
            
            # Process data
            start_time = time.time()
            ds_times, ds_strain, quantum_states = detector.process_gw_data(
                current_times, current_strain, 
                downsample_factor=stage_downsample,
                scale_factor=self.config['scale_factor']
            )
            process_time = time.time() - start_time
            
            # Calculate metrics
            qfi_values = detector.calculate_qfi(quantum_states, ds_times)
            detection_metric = detector.calculate_detection_metric(quantum_states)
            
            # Calculate statistics
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
                'times': ds_times.tolist() if save_results else None,
                'detection_metric': detection_metric.tolist() if save_results else None,
                'quantum_states': None  # Don't store full quantum states (too large)
            }
            pipeline_results['stages'].append(stage_results)
            
            # Use this stage's output as input to the next stage
            current_times = ds_times
            current_strain = detection_metric
        
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
        
        # Save results if requested
        if save_results:
            output_dir = os.path.join(self.results_dir, timestamp)
            os.makedirs(output_dir, exist_ok=True)
            
            results_path = os.path.join(output_dir, f"results_{pipeline_str}.json")
            with open(results_path, 'w') as f:
                json.dump(pipeline_results, f, indent=2)
            
            pipeline_results['file_paths'] = {
                'results': results_path
            }
        
        # Generate visualization if requested
        if save_visualization:
            from qgw_detector.visualization.plots import create_pipeline_visualization
            
            output_dir = os.path.join(self.results_dir, timestamp)
            os.makedirs(output_dir, exist_ok=True)
            
            viz_path = os.path.join(output_dir, f"pipeline_{pipeline_str}.png")
            create_pipeline_visualization(pipeline_results, viz_path)
            
            if 'file_paths' not in pipeline_results:
                pipeline_results['file_paths'] = {}
            pipeline_results['file_paths']['visualization'] = viz_path
        
        # Store in cache
        self.results_cache[cache_key] = pipeline_results
        
        return pipeline_results
    
    def compare_pipelines(self, pipeline_configs, names=None):
        """
        Compare multiple pipeline configurations
        
        Args:
            pipeline_configs: List of pipeline configurations
            names: Optional list of names for each configuration
            
        Returns:
            dict: Comparison results
        """
        if names is None:
            names = [f"Pipeline {i+1}" for i in range(len(pipeline_configs))]
        
        results = []
        for i, (config, name) in enumerate(zip(pipeline_configs, names)):
            result = self.run_pipeline(config)
            result['name'] = name
            results.append(result)
        
        # Calculate comparison metrics
        comparison = {
            'event_name': self.config['event_name'],
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'pipelines': []
        }
        
        for result in results:
            pipeline_summary = {
                'name': result['name'],
                'configuration': result['pipeline_config'],
                'final_snr': result['summary']['final_stage_snr'],
                'improvement': result['summary']['improvement_factor'],
                'execution_time': result['summary']['total_execution_time']
            }
            comparison['pipelines'].append(pipeline_summary)
        
        # Sort by final SNR
        comparison['pipelines'].sort(key=lambda x: x['final_snr'], reverse=True)
        
        # Generate visualization
        from qgw_detector.visualization.plots import create_comparison_visualization
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Comparison also gets its own timestamp for viz
        output_dir = os.path.join(self.results_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        viz_path = os.path.join(output_dir, "pipeline_comparison.png")
        create_comparison_visualization(comparison, viz_path)
        
        comparison['file_paths'] = {
            'visualization': viz_path
        }
        
        return comparison

    def list_saved_results(self):
        """List available saved experiment results."""
        saved_results_info = []
        if not os.path.exists(self.results_dir):
            return []
            
        for timestamp_dir in os.listdir(self.results_dir):
            dir_path = os.path.join(self.results_dir, timestamp_dir)
            if os.path.isdir(dir_path):
                for filename in os.listdir(dir_path):
                    if filename.startswith("results_") and filename.endswith(".json"):
                        # Extract info from filename or load minimal metadata
                        # For now, just use filename and timestamp
                        pipeline_name = filename.replace("results_", "").replace(".json", "")
                        saved_results_info.append({
                            "identifier": os.path.join(timestamp_dir, filename), # More robust identifier
                            "pipeline_name": pipeline_name,
                            "timestamp": timestamp_dir,
                            "full_path": os.path.join(dir_path, filename)
                        })
        # Sort by timestamp, newest first
        saved_results_info.sort(key=lambda x: x['timestamp'], reverse=True)
        return saved_results_info

    def get_saved_result(self, result_identifier: str):
        """
        Get a specific saved experiment result.
        
        Args:
            result_identifier: The identifier of the result file (e.g., "YYYYMMDD_HHMMSS/results_pipeline.json")
        
        Returns:
            dict: The loaded result data.
        """
        file_path = os.path.join(self.results_dir, result_identifier)
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            raise FileNotFoundError(f"Result file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            result_data = json.load(f)
        return result_data

    def sweep_qubit_count(self, topology: str, qubit_counts: list, base_config_params: dict = None):
        """Perform a qubit count sweep for a given topology."""
        results = []
        original_config = self.config.copy()
        if base_config_params:
            self.set_config(**base_config_params)

        self._ensure_data_loaded() # Ensure data is loaded based on current config (e.g. event_name)

        for qubits in qubit_counts:
            if self.config.get('use_zx_opt') and qubits == 8 and topology == 'full':
                print(f"Warning: Skipping 8-qubit full topology with ZX optimization due to potential issues.")
                continue
            pipeline_config = [(qubits, topology)]
            run_name = f"{qubits}q-{topology}"
            print(f"Running sweep: {run_name}")
            try:
                result = self.run_pipeline(pipeline_config)
                result['name'] = run_name 
                results.append(result)
            except Exception as e:
                print(f"Error running pipeline {run_name}: {e}")
                results.append({"name": run_name, "error": str(e)})
        
        if base_config_params: # Restore original config
            self.config = original_config
        return results

    def sweep_topology(self, qubit_count: int, topologies: list, base_config_params: dict = None):
        """Perform a topology sweep for a given qubit count."""
        results = []
        original_config = self.config.copy()
        if base_config_params:
            self.set_config(**base_config_params)

        self._ensure_data_loaded()

        for topology in topologies:
            if self.config.get('use_zx_opt') and qubit_count == 8 and topology == 'full':
                print(f"Warning: Skipping 8-qubit full topology with ZX optimization due to potential issues.")
                continue
            pipeline_config = [(qubit_count, topology)]
            run_name = f"{qubit_count}q-{topology}"
            print(f"Running sweep: {run_name}")
            try:
                result = self.run_pipeline(pipeline_config)
                result['name'] = run_name
                results.append(result)
            except Exception as e:
                print(f"Error running pipeline {run_name}: {e}")
                results.append({"name": run_name, "error": str(e)})

        if base_config_params: # Restore original config
            self.config = original_config
        return results

    def sweep_scale_factor(self, pipeline_config: list, scale_factors: list, base_config_params: dict = None):
        """Perform a scale factor sweep for a given pipeline configuration."""
        results = []
        original_config = self.config.copy()
        
        # Apply base_config_params first, but scale_factor will be overridden in the loop
        if base_config_params:
            temp_config = base_config_params.copy()
            if 'scale_factor' in temp_config: # remove scale_factor if present, as it's the sweep variable
                del temp_config['scale_factor']
            self.set_config(**temp_config)

        self._ensure_data_loaded()
        
        original_api_scale_factor = self.config['scale_factor'] # Save the API's current scale_factor

        for factor in scale_factors:
            self.config['scale_factor'] = factor # Temporarily set API's scale_factor for this run
            run_name = f"scale_{factor:.2e}"
            print(f"Running sweep: {run_name} for pipeline {'_'.join([f'{q}-{t}' for q, t in pipeline_config])}")
            try:
                # run_pipeline uses self.config['scale_factor']
                result = self.run_pipeline(pipeline_config) 
                result['name'] = run_name
                results.append(result)
            except Exception as e:
                print(f"Error running pipeline {run_name}: {e}")
                results.append({"name": run_name, "error": str(e)})
        
        self.config['scale_factor'] = original_api_scale_factor # Restore API's scale_factor
        if base_config_params: # Restore original config fully
            self.config = original_config
        return results

    # --- Advanced Tools (Placeholders) ---
    def analyze_quantum_state(self, result_identifier: str, stage_number: int):
        """Placeholder for quantum state analysis."""
        # In a real implementation, this would load a specific state and perform analysis.
        # For now, it depends on how/if quantum states are saved by run_pipeline.
        # Currently, 'quantum_states' are not fully saved in the JSON to avoid large files.
        return {"message": "Quantum state analysis not yet implemented.", 
                "result_identifier": result_identifier, "stage": stage_number}

    def get_circuit_visualization_data(self, pipeline_config: list, stage_number: int):
        """Placeholder for generating circuit visualization data."""
        # This could generate a text representation or data for a visualizer.
        # For example, using PennyLane's qml.draw or qml.draw_mpl.
        if not (0 <= stage_number < len(pipeline_config)):
            raise ValueError("Invalid stage number.")
        stage_config = pipeline_config[stage_number]
        return {"message": "Circuit visualization data generation not yet implemented.",
                "pipeline_stage_config": stage_config}

    def batch_export_results(self, result_identifiers: list, export_format: str = "csv_summary"):
        """Placeholder for batch exporting results."""
        # This would load multiple results and compile them into a specified format.
        return {"message": f"Batch export to {export_format} not yet implemented.",
                "results_to_export": result_identifiers}

    def run_noise_analysis(self, pipeline_config: list, noise_model_params: dict):
        """Placeholder for running pipeline with noise simulation."""
        # This would require significant extension of the QuantumGWDetector or a new class
        # to incorporate noise models into the simulation.
        return {"message": "Noise analysis simulation not yet implemented.",
                "pipeline_config": pipeline_config, "noise_params": noise_model_params}

    def optimize_circuit_with_zx_details(self, circuit_data: any, optimization_level: int):
        """Placeholder for detailed ZX-calculus optimization interaction."""
        # This assumes `circuit_data` is some representation of a quantum circuit.
        # It would interact with PyZX for optimization and return detailed stats/optimized circuit.
        return {"message": "Detailed ZX-calculus optimization not yet implemented.",
                "optimization_level": optimization_level}
    # --- End Advanced Tools ---
    
    def get_available_events(self):
        """Get list of available gravitational wave events"""
        return ["GW150914", "GW151226", "GW170104", "GW170814", "GW170817"]
    
    def _ensure_data_loaded(self):
        """Ensure LIGO data is loaded"""
        event_name = self.config['event_name']
        if event_name not in self.data_cache:
            times, strain, sample_rate = fetch_gw_data(event_name)
            proc_times, proc_strain = preprocess_gw_data(times, strain, sample_rate)
            
            self.data_cache[event_name] = {
                'times': proc_times,
                'strain': proc_strain,
                'sample_rate': sample_rate
            }

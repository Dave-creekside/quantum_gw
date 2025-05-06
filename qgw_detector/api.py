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
        
        # Create directories
        os.makedirs("data/experiments", exist_ok=True)
    
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
        presets = {
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
        return presets
    
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
        current_times = self.data_cache['times']
        current_strain = self.data_cache['strain']
        
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
            output_dir = f"data/experiments/{timestamp}"
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
            
            output_dir = f"data/experiments/{timestamp}"
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
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"data/experiments/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        viz_path = os.path.join(output_dir, "pipeline_comparison.png")
        create_comparison_visualization(comparison, viz_path)
        
        comparison['file_paths'] = {
            'visualization': viz_path
        }
        
        return comparison
    
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
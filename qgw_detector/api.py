# qgw_detector/api.py
import os
import json
import numpy as np
import time
import uuid
from datetime import datetime
import traceback # For more detailed error logging

from qgw_detector.data.ligo import fetch_gw_data, preprocess_gw_data
from qgw_detector.quantum.circuits import QuantumGWDetector
from qgw_detector.utils.gpu_monitor import GPUMonitor # Import the monitor

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

        # Initialize GPU Monitor (log to console, not file by default for API use)
        try:
            self.gpu_monitor = GPUMonitor(log_to_file=False)
            print("GPUMonitor initialized.")
        except Exception as e:
            print(f"Warning: Failed to initialize GPUMonitor: {e}")
            self.gpu_monitor = None
    
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
    
    def run_pipeline(self, pipeline_config, save_results=True, save_visualization=True, active_project_id=None, **kwargs):
        """
        Run a pipeline configuration, optionally associating it with a project.
        
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
            
            # Initialize file_paths if it doesn't exist
            if 'file_paths' not in pipeline_results:
                 pipeline_results['file_paths'] = {}
            pipeline_results['file_paths']['results'] = results_path # Store results path

        # Generate visualization if requested (BEFORE saving results JSON)
        if save_visualization:
            from qgw_detector.visualization.plots import create_pipeline_visualization
            
            output_dir = os.path.join(self.results_dir, timestamp)
            os.makedirs(output_dir, exist_ok=True)
            
            viz_path = os.path.join(output_dir, f"pipeline_{pipeline_str}.png")
            print(f"[run_pipeline] Generating visualization at: {viz_path}")
            try:
                create_pipeline_visualization(pipeline_results, viz_path)
                # Add viz path to the results dict *before* saving
                if 'file_paths' not in pipeline_results:
                    pipeline_results['file_paths'] = {}
                pipeline_results['file_paths']['visualization'] = viz_path
                print(f"[run_pipeline] Visualization path added to results: {viz_path}")
            except Exception as e:
                 print(f"[run_pipeline] Error generating visualization: {e}")
                 # Decide if we should still save results or raise error
                 # For now, just log it and continue without viz path

        # Save results JSON *after* potentially adding visualization path
        if save_results:
            results_path = pipeline_results['file_paths']['results'] # Get path saved earlier
            print(f"[run_pipeline] Saving final results JSON (with potential viz path) to: {results_path}")
            with open(results_path, 'w') as f:
                json.dump(pipeline_results, f, indent=2)
            print(f"[run_pipeline] Results JSON saved successfully.")


        # Associate run with active project if provided (AFTER saving results)
        # active_project_id is now a direct argument from the signature
        print(f"[run_pipeline] Checking association for project ID: {active_project_id}") # Log the ID being checked
        if active_project_id and save_results:
            print(f"[run_pipeline] Attempting to associate run {timestamp} with project {active_project_id}")
            try:
                self.add_run_to_project(active_project_id, timestamp)
            except Exception as e:
                print(f"[run_pipeline] Warning: Could not associate run {timestamp} with project {active_project_id}: {e}")
        elif not active_project_id:
             print("[run_pipeline] No active project ID provided, skipping run association.")
        elif not save_results:
             print("[run_pipeline] save_results is False, skipping run association.")

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

    def sweep_qubit_count(self, topology: str, qubit_counts: list, base_config_params: dict = None, active_project_id=None):
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
            print(f"Running sweep: {run_name} for project {active_project_id}")
            try:
                # Pass active_project_id to run_pipeline
                result = self.run_pipeline(pipeline_config, active_project_id=active_project_id)
                result['name'] = run_name
                results.append(result)
            except Exception as e:
                print(f"Error running pipeline {run_name}: {e}")
                results.append({"name": run_name, "error": str(e)})
        
        if base_config_params: # Restore original config
            self.config = original_config
        return results

    def sweep_topology(self, qubit_count: int, topologies: list, base_config_params: dict = None, active_project_id=None):
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
            print(f"Running sweep: {run_name} for project {active_project_id}")
            try:
                 # Pass active_project_id to run_pipeline
                result = self.run_pipeline(pipeline_config, active_project_id=active_project_id)
                result['name'] = run_name
                results.append(result)
            except Exception as e:
                print(f"Error running pipeline {run_name}: {e}")
                results.append({"name": run_name, "error": str(e)})

        if base_config_params: # Restore original config
            self.config = original_config
        return results

    def sweep_scale_factor(self, pipeline_config: list, scale_factors: list, base_config_params: dict = None, active_project_id=None):
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
            print(f"Running sweep: {run_name} for pipeline {'_'.join([f'{q}-{t}' for q, t in pipeline_config])} for project {active_project_id}")
            try:
                # run_pipeline uses self.config['scale_factor']
                # Pass active_project_id to run_pipeline
                result = self.run_pipeline(pipeline_config, active_project_id=active_project_id)
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

    def create_project(self, name: str, base_configuration: dict):
        """
        Creates a new project workspace file.

        Args:
            name: User-defined name for the project.
            base_configuration: The initial configuration settings for this project.
                                Should include event_name, parameters, pipeline_config.

        Returns:
            Dict with status and the new project's ID.
        """
        try:
            projects_dir = os.path.join(os.path.dirname(self.results_dir), "projects")
            os.makedirs(projects_dir, exist_ok=True)
            print(f"Projects directory: {projects_dir}")

            project_id = str(uuid.uuid4())
            created_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            project_data = {
                "id": project_id,
                "name": name,
                "created_timestamp": created_timestamp,
                "base_configuration": base_configuration,
                "associated_runs": []  # Initialize with empty list of run identifiers
            }

            # Use project ID for the filename for easier lookup
            filename = f"{project_id}.json"
            file_path = os.path.join(projects_dir, filename)

            with open(file_path, "w") as f:
                json.dump(project_data, f, indent=2)

            print(f"Project '{name}' created with ID {project_id} at {file_path}")

            return {
                "success": True,
                "message": f"Project '{name}' created successfully.",
                "project_id": project_id,
                "file_path": file_path
            }
        except Exception as e:
            print(f"Error creating project: {e}")
            return {
                "success": False,
                "message": f"Error creating project: {str(e)}",
                "error": str(e)
            }
    
    def list_projects(self):
        """
        List available saved projects
        
        Returns:
            List of project metadata (id, name, date, etc)
        """
        try:
            # Create projects directory if it doesn't exist
            projects_dir = os.path.join(os.path.dirname(self.results_dir), "projects")
            os.makedirs(projects_dir, exist_ok=True)
            print(f"Projects directory exists: {os.path.exists(projects_dir)}")
            print(f"Looking for projects in: {projects_dir}")
            
            if not os.path.exists(projects_dir):
                print(f"Projects directory does not exist even after attempting to create it: {projects_dir}")
                return []
            
            files = os.listdir(projects_dir)
            print(f"Files in projects directory: {files}")
            
            projects = []
            for filename in files:
                if filename.endswith(".json"):
                    file_path = os.path.join(projects_dir, filename)
                    try:
                        with open(file_path, "r") as f:
                            project_data = json.load(f)
                        
                        # Extract relevant metadata from the project file
                        project_id = project_data.get("id")
                        project_name = project_data.get("name", "Unnamed Project")
                        created_timestamp = project_data.get("created_timestamp", "") # Use created_timestamp for sorting

                        if not project_id:
                             print(f"Warning: Skipping project file {filename} due to missing 'id'.")
                             continue

                        projects.append({
                            "id": project_id,
                            "name": project_name,
                            "timestamp": created_timestamp, # Use created_timestamp for sorting/display
                            "file_path": file_path
                        })
                        print(f"Found project: {project_name} (ID: {project_id})")
                    except json.JSONDecodeError:
                        print(f"Error: Could not decode JSON from project file {filename}")
                    except Exception as e:
                        print(f"Error processing project file {filename}: {e}")

            print(f"Total valid projects found: {len(projects)}")

            # Sort by creation timestamp, newest first
            projects.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return projects
        except Exception as e:
            print(f"Error listing projects: {e}")
            return []
    
    def load_project(self, project_id: str):
        """
        Load a saved project
        
        Args:
            project_id: Project identifier
            
        Returns:
            Project configuration
        """
        try:
            print(f"Loading project with ID: {project_id}")
            projects_dir = os.path.join(os.path.dirname(self.results_dir), "projects")
            os.makedirs(projects_dir, exist_ok=True)
            
            # If project_id is a file path, use it directly
            if os.path.exists(project_id) and project_id.endswith(".json"):
                file_path = project_id
                print(f"Using direct file path: {file_path}")
            else:
                print(f"Searching for project by ID in: {projects_dir}")
                # List files in the directory
                files = os.listdir(projects_dir)
                print(f"Files in directory: {files}")
                
                # Search for the project by ID
                file_path = None
                for filename in files:
                    if filename.endswith(".json"):
                        current_path = os.path.join(projects_dir, filename)
                        try:
                            with open(current_path, "r") as f:
                                current_data = json.load(f)
                            
                            current_id = current_data.get("id")
                            print(f"Checking file {filename} with ID: {current_id}")
                            
                            if current_id == project_id:
                                file_path = current_path
                                print(f"Found matching project: {file_path}")
                                break
                        except Exception as e:
                            print(f"Error reading project file {filename}: {e}")
                            continue
                
                if not file_path:
                    error_msg = f"Project with ID {project_id} not found"
                    print(error_msg)
                    raise FileNotFoundError(error_msg)
            
            # Load the project data
            print(f"Loading project data from: {file_path}")
            with open(file_path, "r") as f:
                project_data = json.load(f)
            
            print(f"Successfully loaded project: {project_data.get('name', 'Unnamed')}")
            return project_data

        except FileNotFoundError:
             # Re-raise FileNotFoundError specifically so the web API can return 404
             raise
        except Exception as e:
            print(f"Error loading project: {e}")
            # Raise a generic exception for other errors
            raise Exception(f"Failed to load project data for ID {project_id}: {e}")

    def add_run_to_project(self, project_id: str, run_timestamp: str):
        """
        Associates a run (identified by its timestamp) with a project.

        Args:
            project_id: The ID of the project to update.
            run_timestamp: The timestamp identifier of the run to add.
        """
        try:
            projects_dir = os.path.join(os.path.dirname(self.results_dir), "projects")
            project_filename = f"{project_id}.json"
            project_filepath = os.path.join(projects_dir, project_filename)
            print(f"[add_run_to_project] Attempting to update project file: {project_filepath}")

            if not os.path.exists(project_filepath):
                print(f"[add_run_to_project] Error: Project file not found for ID {project_id} at {project_filepath}")
                raise FileNotFoundError(f"Project file not found for ID {project_id}")

            # Load existing project data
            print(f"[add_run_to_project] Loading existing data from {project_filepath}")
            with open(project_filepath, "r") as f:
                project_data = json.load(f)
            print(f"[add_run_to_project] Loaded data: {project_data}")

            # Add the run timestamp if not already present
            if "associated_runs" not in project_data:
                project_data["associated_runs"] = []
                print("[add_run_to_project] Initialized 'associated_runs' list.")

            if run_timestamp not in project_data["associated_runs"]:
                print(f"[add_run_to_project] Appending run timestamp: {run_timestamp}")
                project_data["associated_runs"].append(run_timestamp)
                # Optional: Sort runs by timestamp? For now, just append.
                # project_data["associated_runs"].sort(reverse=True)

                # Save the updated project data
                print(f"[add_run_to_project] Saving updated data back to {project_filepath}")
                with open(project_filepath, "w") as f:
                    json.dump(project_data, f, indent=2)
                print(f"[add_run_to_project] Successfully associated run {run_timestamp} with project {project_id}")
            else:
                print(f"[add_run_to_project] Run {run_timestamp} already associated with project {project_id}. No update needed.")

        except Exception as e:
            print(f"Error adding run {run_timestamp} to project {project_id}: {e}")
            # Decide if this should raise an exception or just log the warning
            # For now, just print, as the main run succeeded anyway.

    def get_project_run_details(self, project_id: str):
        """
        Retrieves summary details for all runs associated with a project.

        Args:
            project_id: The ID of the project.

        Returns:
            List of dictionaries, each containing summary details for a run.
        """
        try:
            print(f"[get_project_run_details] Getting runs for project ID: {project_id}")
            project_data = self.load_project(project_id) # Reuse load_project to get the data
            print(f"[get_project_run_details] Loaded project data: {project_data}") # Log loaded data
            run_identifiers = project_data.get("associated_runs", [])
            print(f"[get_project_run_details] Found {len(run_identifiers)} associated runs: {run_identifiers}") # Log the identifiers

            run_details = []
            for run_timestamp in run_identifiers:
                print(f"Processing run timestamp: {run_timestamp}")
                try:
                    # Construct the expected results file path based on timestamp
                    # Note: We don't know the exact pipeline_str here, so we need to find the results file.
                    run_dir = os.path.join(self.results_dir, run_timestamp)
                    if not os.path.isdir(run_dir):
                        print(f"Warning: Run directory not found for timestamp {run_timestamp}")
                        continue

                    results_file = None
                    for filename in os.listdir(run_dir):
                        if filename.startswith("results_") and filename.endswith(".json"):
                            results_file = os.path.join(run_dir, filename)
                            break
                    
                    if not results_file:
                        print(f"Warning: Results JSON file not found in {run_dir}")
                        continue

                    # Load the results data for this run
                    print(f"Loading results file: {results_file}")
                    with open(results_file, "r") as f:
                        run_data = json.load(f)
                    print(f"Successfully loaded results for run {run_timestamp}")

                    # Extract summary information
                    summary = run_data.get("summary", {})
                    run_details.append({
                        "run_timestamp": run_timestamp,
                        "pipeline_config_str": run_data.get("pipeline_config", "Unknown"),
                        "event_name": run_data.get("event_name", "Unknown"),
                        "final_snr": summary.get("final_stage_snr"),
                        "max_qfi": run_data.get("stages", [{}])[-1].get("max_qfi"), # Max QFI from last stage
                        "visualization_path": run_data.get("file_paths", {}).get("visualization")
                    })
                except Exception as e:
                    print(f"Error loading details for run {run_timestamp}: {e}")
                    # Optionally add an error entry to the list
                    run_details.append({
                        "run_timestamp": run_timestamp,
                        "error": f"Failed to load details: {e}"
                    })
            
            # Sort details by timestamp, newest first
            run_details.sort(key=lambda x: x.get("run_timestamp", ""), reverse=True)
            return run_details

        except FileNotFoundError:
            # Project file itself not found
            raise
        except Exception as e:
            print(f"Error getting run details for project {project_id}: {e}")
            raise Exception(f"Failed to get run details for project {project_id}: {e}")

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
            
    def get_system_stats(self):
        """
        Retrieves current system resource usage (CPU, RAM, GPU).
        """
        stats = {}
        try:
            import psutil
            # CPU Stats
            stats['cpu_percent'] = psutil.cpu_percent(interval=0.1) # Short interval for responsiveness

            # RAM Stats
            mem = psutil.virtual_memory()
            stats['ram_total_gb'] = mem.total / (1024**3)
            stats['ram_used_gb'] = mem.used / (1024**3)
            stats['ram_percent'] = mem.percent

        except ImportError:
            print("Warning: psutil not installed. Cannot get CPU/RAM stats.")
            stats['cpu_percent'] = 'N/A'
            stats['ram_percent'] = 'N/A'
            stats['ram_total_gb'] = 'N/A'
            stats['ram_used_gb'] = 'N/A'
        except Exception as e:
            print(f"Error getting CPU/RAM stats: {e}")
            stats['cpu_percent'] = 'Error'
            stats['ram_percent'] = 'Error'
            stats['ram_total_gb'] = 'N/A'
            stats['ram_used_gb'] = 'N/A'

        # Initialize GPU stats with default N/A values
        stats['gpu_name'] = 'N/A'
        stats['gpu_utilization_percent'] = 'N/A'
        stats['vram_percent'] = 'N/A'
        stats['vram_total_mb'] = 'N/A'
        stats['vram_used_mb'] = 'N/A'
        stats['gpu_temperature_c'] = 'N/A'

        # Try to use the existing GPU monitor if it's already initialized
        gpu_monitor = None
        
        # Try up to 2 times to get GPU info
        for attempt in range(2):
            try:
                # Create a new monitor instance if needed
                if gpu_monitor is None:
                    from qgw_detector.utils.gpu_monitor import GPUMonitor
                    gpu_monitor = GPUMonitor(log_to_file=False)
                
                # Get GPU info with potentially enhanced error handling
                gpu_info = gpu_monitor.get_gpu_info()
                
                if gpu_info.get("available"):
                    # GPU Name - this should always be available if GPU is available
                    stats['gpu_name'] = gpu_info.get('device_name', 'N/A')
                    
                    # Utilization - might be None from improved error handling
                    util = gpu_info.get('utilization')
                    stats['gpu_utilization_percent'] = float(util) if isinstance(util, (int, float)) else 'N/A'
                    
                    # Temperature - might be None from improved error handling
                    temp = gpu_info.get('temperature')
                    stats['gpu_temperature_c'] = float(temp) if isinstance(temp, (int, float)) else 'N/A'
                    
                    # Memory metrics
                    if 'nvidia_smi_memory_total' in gpu_info and 'nvidia_smi_memory_used' in gpu_info:
                        total = gpu_info['nvidia_smi_memory_total']
                        used = gpu_info['nvidia_smi_memory_used']
                        
                        # Validate numbers and calculate percentage
                        if isinstance(total, (int, float)) and total > 0:
                            stats['vram_total_mb'] = float(total)
                            
                            if isinstance(used, (int, float)):
                                stats['vram_used_mb'] = float(used)
                                stats['vram_percent'] = (used / total * 100)
                            else:
                                # Fallback to PyTorch values
                                reserved = gpu_info.get('memory_reserved')
                                if isinstance(reserved, (int, float)):
                                    stats['vram_used_mb'] = float(reserved)
                                    stats['vram_percent'] = (reserved / total * 100)
                        else:
                            # Use PyTorch's reserved memory if nvidia-smi values aren't valid
                            reserved = gpu_info.get('memory_reserved')
                            if isinstance(reserved, (int, float)):
                                stats['vram_used_mb'] = float(reserved)
                    else:
                        # No nvidia-smi memory info, use PyTorch's reserved memory
                        reserved = gpu_info.get('memory_reserved')
                        if isinstance(reserved, (int, float)):
                            stats['vram_used_mb'] = float(reserved)
                
                # If we got here without errors, break the retry loop
                break
                
            except ImportError as e:
                print(f"Warning: GPUMonitor utility not found: {e}")
                # No need to retry on import error
                break
            except Exception as e:
                print(f"Error getting GPU stats (attempt {attempt+1}/2): {e}")
                # Only set error message on last attempt
                if attempt == 1:
                    stats['gpu_name'] = f'Error: {str(e)[:50]}...' if len(str(e)) > 50 else f'Error: {str(e)}'
                # Short delay before retry
                if attempt == 0:
                    time.sleep(0.5)

        # Ensure all values are either numbers or string indicators
        for key, value in stats.items():
            if not isinstance(value, (int, float, str)):
                stats[key] = 'Error'
                
        print(f"System stats collected: {stats}")
        return stats

    def update_project_configuration(self, project_id: str, new_config_data: dict):
        """
        Updates the base_configuration of a specific project.

        Args:
            project_id: The ID of the project to update.
            new_config_data: A dictionary containing the new configuration
                             (e.g., {"parameters": {...}, "pipeline_config": [...]}).
        
        Returns:
            Dict with status and updated project data.
        """
        try:
            projects_dir = os.path.join(os.path.dirname(self.results_dir), "projects")
            project_filename = f"{project_id}.json"
            project_filepath = os.path.join(projects_dir, project_filename)

            if not os.path.exists(project_filepath):
                raise FileNotFoundError(f"Project file not found for ID {project_id}")

            with open(project_filepath, "r") as f:
                project_data = json.load(f)
            
            # Update the base_configuration
            # Ensure new_config_data has the expected structure (parameters, pipeline_config)
            if "parameters" in new_config_data and "pipeline_config" in new_config_data:
                project_data["base_configuration"] = {
                    "event_name": new_config_data["parameters"].get("event_name", project_data["base_configuration"].get("event_name")), # Keep old if not provided
                    "parameters": new_config_data["parameters"],
                    "pipeline_config": new_config_data["pipeline_config"]
                }
                # Update timestamp of modification (optional, but good practice)
                project_data["modified_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
            else:
                raise ValueError("Invalid new_config_data structure. Must contain 'parameters' and 'pipeline_config'.")

            with open(project_filepath, "w") as f:
                json.dump(project_data, f, indent=2)
            
            return {
                "success": True,
                "message": f"Project '{project_data.get('name', project_id)}' configuration updated successfully.",
                "project_data": project_data # Return the updated project data
            }
        except FileNotFoundError:
            raise # Re-raise to be caught by web_api for 404
        except ValueError as ve:
            raise # Re-raise to be caught by web_api for 400
        except Exception as e:
            print(f"Error updating project configuration for {project_id}: {e}")
            traceback.print_exc()
            raise Exception(f"Failed to update project configuration: {str(e)}")

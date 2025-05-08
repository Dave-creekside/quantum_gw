 # qgw_detector/web_api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict, Any, Union
import uvicorn
import os

from qgw_detector.api import QuantumGWAPI

# Create FastAPI app
app = FastAPI(
    title="Quantum Gravitational Wave Detector API",
    description="API for Quantum Gravitational Wave Detection experiments",
    version="1.0.0"
)

# Add CORS middleware to allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create API instance
api = QuantumGWAPI()

# Define models
class ConfigUpdate(BaseModel):
    event_name: Optional[str] = None
    downsample_factor: Optional[int] = None
    scale_factor: Optional[float] = None
    use_gpu: Optional[bool] = None
    use_zx_opt: Optional[bool] = None
    zx_opt_level: Optional[int] = None
    
class ProjectData(BaseModel):
    name: str
    configuration: Dict[str, Any]

class PipelineRunRequest(BaseModel):
    stages: List[Tuple[int, str]]
    save_results: bool = True
    save_visualization: bool = True
    active_project_id: Optional[str] = None # Add active project ID

class ComparisonRequest(BaseModel):
    pipeline_configs: List[List[Tuple[int, str]]]
    names: Optional[List[str]] = None

class AddPresetRequest(BaseModel):
    name: str
    config: List[Tuple[int, str]]

class SweepQubitRequest(BaseModel):
    topology: str
    qubit_counts: List[int]
    base_config_params: Optional[Dict[str, Any]] = None
    active_project_id: Optional[str] = None # Add project ID

class SweepTopologyRequest(BaseModel):
    qubit_count: int
    topologies: List[str]
    base_config_params: Optional[Dict[str, Any]] = None
    active_project_id: Optional[str] = None # Add project ID

class SweepScaleFactorRequest(BaseModel):
    pipeline_config: List[Tuple[int, str]]
    scale_factors: List[float]
    base_config_params: Optional[Dict[str, Any]] = None
    active_project_id: Optional[str] = None # Add project ID

# --- Advanced Tools Models (Placeholders) ---
class AnalyzeStateRequest(BaseModel):
    result_identifier: str
    stage_number: int

class CircuitVisualizationRequest(BaseModel):
    pipeline_config: List[Tuple[int, str]]
    stage_number: int

class BatchExportRequest(BaseModel):
    result_identifiers: List[str]
    export_format: Optional[str] = "csv_summary"

class NoiseAnalysisRequest(BaseModel):
    pipeline_config: List[Tuple[int, str]]
    noise_model_params: Dict[str, Any]

class ZXOptimizeDetailsRequest(BaseModel):
    circuit_data: Any # This would be more specific in a real implementation
    optimization_level: int
# --- End Advanced Tools Models ---


# Mount static directories
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Base directory: {base_dir}")

# Mount frontend static files
frontend_dir = os.path.join(base_dir, "frontend")
if os.path.exists(frontend_dir):
    print(f"Mounting frontend directory: {frontend_dir}")
    app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")

# Mount data directory for visualizations and results
data_dir = os.path.join(base_dir, "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    print(f"Created data directory: {data_dir}")

# Create subdirectories in data if they don't exist
projects_dir = os.path.join(data_dir, "projects")
experiments_dir = os.path.join(data_dir, "experiments")
os.makedirs(projects_dir, exist_ok=True)
os.makedirs(experiments_dir, exist_ok=True)

# Mount data directory
print(f"Mounting data directory: {data_dir}")
app.mount("/data", StaticFiles(directory=data_dir), name="data")

# Define routes
@app.get("/")
def read_root():
    """Redirect to the frontend if it exists, otherwise show API info"""
    if os.path.exists(os.path.join(frontend_dir, "index.html")):
        return RedirectResponse(url="/frontend/index.html")
    return {"message": "Quantum Gravitational Wave Detector API"}

@app.get("/api")
def api_info():
    """API information endpoint"""
    return {
        "title": "Quantum Gravitational Wave Detector API",
        "version": "1.0.0", 
        "description": "API for Quantum Gravitational Wave Detection experiments"
    }

@app.get("/config")
def get_config():
    return api.get_config()

@app.post("/config")
def update_config(update: ConfigUpdate):
    config_updates = {k: v for k, v in update.dict().items() if v is not None}
    return api.set_config(**config_updates)

@app.get("/presets")
def get_presets():
    return api.list_presets()

@app.post("/presets")
def add_preset(request: AddPresetRequest):
    try:
        return api.add_preset(name=request.name, config=request.config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
def list_saved_results():
    try:
        return api.list_saved_results()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{result_identifier:path}")
def get_saved_result(result_identifier: str):
    try:
        return api.get_saved_result(result_identifier)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects")
def list_projects():
    """List all saved projects"""
    try:
        return api.list_projects()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CreateProjectRequest(BaseModel):
    name: str
    base_configuration: Dict[str, Any]

@app.post("/projects")
def create_project(request: CreateProjectRequest):
    """Create a new project workspace"""
    try:
        # The base_configuration should contain event_name, parameters, pipeline_config
        return api.create_project(
            name=request.name,
            base_configuration=request.base_configuration
        )
    except Exception as e:
        print(f"Error in /projects POST endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/{project_id}")
def get_project(project_id: str):
    """Get a specific project by ID"""
    try:
        return api.load_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/{project_id}/runs")
def get_project_runs(project_id: str):
    """Get details of runs associated with a specific project"""
    try:
        # We need a new method in the API class for this
        # Let's assume it's called get_project_run_details
        return api.get_project_run_details(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    except Exception as e:
        print(f"Error getting runs for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class UpdateProjectConfigRequest(BaseModel):
    parameters: Dict[str, Any]
    pipeline_config: List[Tuple[int, str]]

@app.put("/api/projects/{project_id}/configuration")
def update_project_configuration(project_id: str, request: UpdateProjectConfigRequest):
    """Update the base configuration of a specific project"""
    try:
        # Prepare the data structure expected by the API method
        new_config_data = {
            "parameters": request.parameters,
            "pipeline_config": request.pipeline_config
        }
        return api.update_project_configuration(project_id, new_config_data)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    except ValueError as ve: # Catch validation errors from API method
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error updating configuration for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system_stats")
def get_system_stats():
    """Get current system resource statistics"""
    try:
        return api.get_system_stats()
    except Exception as e:
        print(f"Error getting system stats: {e}")
        # Return a specific error structure or just raise 500
        raise HTTPException(status_code=500, detail=f"Failed to get system stats: {e}")

@app.get("/events")
def get_events():
    return api.get_available_events()

@app.post("/run")
def run_pipeline(request: PipelineRunRequest):
    """Run a pipeline, optionally associating it with a project."""
    try:
        result = api.run_pipeline(
            pipeline_config=request.stages,
            save_results=request.save_results,
            save_visualization=request.save_visualization,
            active_project_id=request.active_project_id # Pass the project ID
        )
        return result
    except Exception as e:
        print(f"Error in /run endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare")
def compare_pipelines(request: ComparisonRequest):
    try:
        result = api.compare_pipelines(
            pipeline_configs=request.pipeline_configs,
            names=request.names
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sweep/qubit_count")
def sweep_qubit_count(request: SweepQubitRequest):
    """Perform a qubit count sweep, optionally associating runs with a project."""
    try:
        results = api.sweep_qubit_count(
            topology=request.topology,
            qubit_counts=request.qubit_counts,
            base_config_params=request.base_config_params,
            active_project_id=request.active_project_id # Pass project ID
        )
        return results
    except Exception as e:
        print(f"Error in /sweep/qubit_count endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sweep/topology")
def sweep_topology(request: SweepTopologyRequest):
    """Perform a topology sweep, optionally associating runs with a project."""
    try:
        results = api.sweep_topology(
            qubit_count=request.qubit_count,
            topologies=request.topologies,
            base_config_params=request.base_config_params,
            active_project_id=request.active_project_id # Pass project ID
        )
        return results
    except Exception as e:
        print(f"Error in /sweep/topology endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sweep/scale_factor")
def sweep_scale_factor(request: SweepScaleFactorRequest):
    """Perform a scale factor sweep, optionally associating runs with a project."""
    try:
        results = api.sweep_scale_factor(
            pipeline_config=request.pipeline_config,
            scale_factors=request.scale_factors,
            base_config_params=request.base_config_params,
            active_project_id=request.active_project_id # Pass project ID
        )
        return results
    except Exception as e:
        print(f"Error in /sweep/scale_factor endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Advanced Tools Endpoints (Placeholders) ---
@app.post("/tools/analyze_state")
def analyze_state(request: AnalyzeStateRequest):
    try:
        return api.analyze_quantum_state(request.result_identifier, request.stage_number)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/circuit_visualization_data")
def get_circuit_visualization_data(request: CircuitVisualizationRequest):
    try:
        return api.get_circuit_visualization_data(request.pipeline_config, request.stage_number)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/batch_export")
def batch_export(request: BatchExportRequest):
    try:
        return api.batch_export_results(request.result_identifiers, request.export_format)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/run_noise_analysis")
def run_noise_analysis(request: NoiseAnalysisRequest):
    try:
        return api.run_noise_analysis(request.pipeline_config, request.noise_model_params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/optimize_zx_details")
def optimize_zx_details(request: ZXOptimizeDetailsRequest):
    try:
        return api.optimize_circuit_with_zx_details(request.circuit_data, request.optimization_level)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# --- End Advanced Tools Endpoints ---


def start_server():
    """Start the FastAPI server"""
    uvicorn.run("qgw_detector.web_api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start_server()

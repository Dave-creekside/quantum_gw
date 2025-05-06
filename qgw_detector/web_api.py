# qgw_detector/web_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict, Any, Union
import uvicorn

from qgw_detector.api import QuantumGWAPI

# Create FastAPI app
app = FastAPI(
    title="Quantum Gravitational Wave Detector API",
    description="API for Quantum Gravitational Wave Detection experiments",
    version="1.0.0"
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

class PipelineConfig(BaseModel):
    stages: List[Tuple[int, str]]
    save_results: bool = True
    save_visualization: bool = True

class ComparisonRequest(BaseModel):
    pipeline_configs: List[List[Tuple[int, str]]]
    names: Optional[List[str]] = None

# Define routes
@app.get("/")
def read_root():
    return {"message": "Quantum Gravitational Wave Detector API"}

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

@app.get("/events")
def get_events():
    return api.get_available_events()

@app.post("/run")
def run_pipeline(config: PipelineConfig):
    try:
        result = api.run_pipeline(
            pipeline_config=config.stages,
            save_results=config.save_results,
            save_visualization=config.save_visualization
        )
        return result
    except Exception as e:
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

def start_server():
    """Start the FastAPI server"""
    uvicorn.run("qgw_detector.web_api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start_server()
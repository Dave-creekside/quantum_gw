/**
 * API Module for quantum gravitational wave detector
 * Handles all interactions with the backend API
 */

const API = (() => {
    // API configuration - Use relative path so it works from any host
    const API_BASE_URL = ''; // Was 'http://localhost:8000'
    
    /**
     * Generic function to handle API errors
     */
    const handleApiError = (error) => {
        console.error('API Error:', error);
        return {
            success: false,
            error: error.message || 'Unknown error occurred'
        };
    };

    /**
     * Make a GET request to the API
     */
    const get = async (endpoint) => {
        try {
            const response = await fetch(`${API_BASE_URL}${endpoint}`);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Error: ${response.status} ${response.statusText}`);
            }
            const data = await response.json();
            return { success: true, data };
        } catch (error) {
            return handleApiError(error);
        }
    };

    /**
     * Make a POST request to the API
     */
    const post = async (endpoint, data) => {
        try {
            const response = await fetch(`${API_BASE_URL}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Error: ${response.status} ${response.statusText}`);
            }
            
            const responseData = await response.json();
            return { success: true, data: responseData };
        } catch (error) {
            return handleApiError(error);
        }
    };

    // Configuration Endpoints
    const getConfig = () => get('/config');
    
    const updateConfig = (configData) => post('/config', configData);

    // Preset Pipeline Endpoints
    const getPresets = () => get('/presets');
    
    const addPreset = (name, config) => post('/presets', { name, config });

    // Results Endpoints
    const getSavedResults = () => get('/results');
    
    const getResultById = (resultId) => get(`/results/${resultId}`);

    // Pipeline Execution Endpoints
    const runPipeline = (stages, options = {}) => {
        const payload = {
            stages,
            save_results: options.saveResults !== false,
            save_visualization: options.saveVisualization !== false,
            active_project_id: options.activeProjectId || null // Include active project ID if provided
        };
        console.log("Running pipeline with payload:", payload);
        return post('/run', payload);
    };

    const comparePipelines = (pipelineConfigs, names) => {
        return post('/compare', {
            pipeline_configs: pipelineConfigs,
            names
        });
    };

    // Gravitational Wave Events
    const getEvents = () => get('/events');

    // Parameter Sweep Endpoints
    const sweepQubitCount = (topology, qubitCounts, options = {}) => {
        const payload = {
            topology,
            qubit_counts: qubitCounts,
            base_config_params: options.baseConfigParams || null,
            active_project_id: options.activeProjectId || null // Add project ID
        };
        console.log("Running qubit sweep with payload:", payload);
        return post('/sweep/qubit_count', payload);
    };

    const sweepTopology = (qubitCount, topologies, options = {}) => {
         const payload = {
            qubit_count: qubitCount,
            topologies,
            base_config_params: options.baseConfigParams || null,
            active_project_id: options.activeProjectId || null // Add project ID
        };
        console.log("Running topology sweep with payload:", payload);
        return post('/sweep/topology', payload);
    };

    const sweepScaleFactor = (pipelineConfig, scaleFactors, options = {}) => {
         const payload = {
            pipeline_config: pipelineConfig,
            scale_factors: scaleFactors,
            base_config_params: options.baseConfigParams || null,
            active_project_id: options.activeProjectId || null // Add project ID
        };
        console.log("Running scale factor sweep with payload:", payload);
        return post('/sweep/scale_factor', payload);
    };

    // Advanced Tool Endpoints
    const analyzeQuantumState = (resultIdentifier, stageNumber) => {
        return post('/tools/analyze_state', {
            result_identifier: resultIdentifier,
            stage_number: stageNumber
        });
    };

    const getCircuitVisualization = (pipelineConfig, stageNumber) => {
        return post('/tools/circuit_visualization_data', {
            pipeline_config: pipelineConfig,
            stage_number: stageNumber
        });
    };

    const batchExportResults = (resultIdentifiers, exportFormat = 'csv_summary') => {
        return post('/tools/batch_export', {
            result_identifiers: resultIdentifiers,
            export_format: exportFormat
        });
    };

    const runNoiseAnalysis = (pipelineConfig, noiseModelParams) => {
        return post('/tools/run_noise_analysis', {
            pipeline_config: pipelineConfig,
            noise_model_params: noiseModelParams
        });
    };

    // Project Management Endpoints
    const createProject = (name, baseConfiguration) => {
        console.log('Creating project:', name, baseConfiguration);
        return post('/projects', {
            name: name,
            base_configuration: baseConfiguration
        });
    };

    const listProjects = async () => {
        console.log('Fetching projects list...');
        const response = await get('/projects');
        console.log('Projects API response:', response);
        return response;
    };

    const loadProject = async (projectId) => {
        console.log('Loading project:', projectId);
        const response = await get(`/projects/${projectId}`);
        console.log('Project load response:', response);
        return response;
    };

    const getProjectRuns = async (projectId) => {
        console.log('Fetching runs for project:', projectId);
        const response = await get(`/projects/${projectId}/runs`);
        console.log('Project runs response:', response);
        return response;
    };
    
    // Return the public API
    return {
        // Basic config management
        getConfig,
        updateConfig,
        
        // Preset management
        getPresets,
        addPreset,
        
        // Results management
        getSavedResults,
        getResultById,
        
        // Pipeline execution
        runPipeline,
        comparePipelines,
        
        // Project management
        createProject,
        listProjects,
        loadProject,
        getProjectRuns,
        
        // Events
        getEvents,
        
        // Parameter sweeps
        sweepQubitCount,
        sweepTopology,
        sweepScaleFactor,
        
        // Advanced tools
        analyzeQuantumState,
        getCircuitVisualization,
        batchExportResults,
        runNoiseAnalysis,

        // System Stats
        getSystemStats: () => get('/api/system_stats'),

        // Update Project Configuration
        updateProjectConfiguration: (projectId, configData) => {
            // Need a PUT method helper
            const put = async (endpoint, data) => {
                try {
                    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
                        method: 'PUT',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(data)
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json().catch(() => ({}));
                        throw new Error(errorData.detail || `Error: ${response.status} ${response.statusText}`);
                    }
                    
                    const responseData = await response.json();
                    return { success: true, data: responseData };
                } catch (error) {
                    return handleApiError(error);
                }
            };
            
            console.log(`Updating configuration for project ${projectId}:`, configData);
            return put(`/api/projects/${projectId}/configuration`, configData);
        }
    };
})();

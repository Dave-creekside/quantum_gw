/**
 * API Module for quantum gravitational wave detector
 * Handles all interactions with the backend API
 */

const API = (() => {
    // API configuration
    const API_BASE_URL = 'http://localhost:8000';
    
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
        const config = {
            stages,
            save_results: options.saveResults !== false,
            save_visualization: options.saveVisualization !== false
        };
        return post('/run', config);
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
    const sweepQubitCount = (topology, qubitCounts, baseConfigParams = null) => {
        return post('/sweep/qubit_count', {
            topology,
            qubit_counts: qubitCounts,
            base_config_params: baseConfigParams
        });
    };

    const sweepTopology = (qubitCount, topologies, baseConfigParams = null) => {
        return post('/sweep/topology', {
            qubit_count: qubitCount,
            topologies,
            base_config_params: baseConfigParams
        });
    };

    const sweepScaleFactor = (pipelineConfig, scaleFactors, baseConfigParams = null) => {
        return post('/sweep/scale_factor', {
            pipeline_config: pipelineConfig,
            scale_factors: scaleFactors,
            base_config_params: baseConfigParams
        });
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
        runNoiseAnalysis
    };
})();

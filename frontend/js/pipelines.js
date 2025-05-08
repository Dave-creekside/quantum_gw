/**
 * Pipelines Module for quantum gravitational wave detector
 * Handles pipeline execution UI and interactions
 */

const PipelinesModule = (() => {
    // DOM elements
    const presetSelect = document.getElementById('preset-select');
    const presetDescription = document.getElementById('preset-description');
    const runPresetButton = document.getElementById('run-preset');
    const pipelineStagesContainer = document.getElementById('pipeline-stages');
    const addStageButton = document.getElementById('add-stage');
    const runCustomButton = document.getElementById('run-custom');
    const pipelineResultContainer = document.getElementById('pipeline-result');
    
    // State
    let presets = {};
    let customStageCount = 0;
    let currentCustomPipeline = []; // Add state for the custom pipeline
    
    /**
     * Load available presets
     */
    const loadPresets = async () => {
        const response = await API.getPresets();
        
        if (response.success) {
            presets = response.data;
            updatePresetsList();
        } else {
            console.error('Error loading presets:', response.error);
        }
    };
    
    /**
     * Update the presets dropdown
     */
    const updatePresetsList = () => {
        if (!presetSelect) return;
        
        presetSelect.innerHTML = '';
        
        // Add option for each preset
        for (const [name, config] of Object.entries(presets)) {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            presetSelect.appendChild(option);
        }
        
        // Update description for selected preset
        updatePresetDescription();
    };
    
    /**
     * Update the description of the selected preset
     */
    const updatePresetDescription = () => {
        if (!presetSelect || !presetDescription) return;
        
        const selectedPreset = presetSelect.value;
        const config = presets[selectedPreset];
        
        if (config) {
            const stages = config.map(([qubits, topology]) => `${qubits} qubits (${topology})`);
            
            presetDescription.innerHTML = `
                <p><strong>Pipeline Configuration:</strong></p>
                <ol>
                    ${stages.map(stage => `<li>${stage}</li>`).join('')}
                </ol>
            `;
        } else {
            presetDescription.innerHTML = '<p>Select a preset to view its configuration.</p>';
        }
    };
    
    /**
     * Run a preset pipeline
     */
    const runPresetPipeline = async () => {
        if (!presetSelect) return;
        
        const selectedPreset = presetSelect.value;
        const config = presets[selectedPreset];
        
        if (!config) {
            alert('Please select a valid preset pipeline.');
            return;
        }
        
        UI.showLoading('Running preset pipeline...');
        
        // Get active project ID
        const activeProjectId = (typeof ProjectsModule !== 'undefined') ? ProjectsModule.getActiveProjectId() : null;
        console.log("Running preset pipeline for active project:", activeProjectId);

        const response = await API.runPipeline(config, { activeProjectId: activeProjectId });
        
        UI.hideLoading();
        
        if (response.success) {
            displayPipelineResults(response.data, selectedPreset);
            // Update dashboard if a project is active
            if (activeProjectId && typeof ProjectsModule !== 'undefined' && ProjectsModule.updateDashboardProjectInfo) {
                 console.log("Pipeline run successful, updating dashboard project info...");
                 ProjectsModule.updateDashboardProjectInfo();
            }
        } else {
            alert(`Error running pipeline: ${response.error}`);
        }
    };
    
    /**
     * Add a new stage to the custom pipeline
     */
    const addCustomStage = () => {
        if (!pipelineStagesContainer) return;
        
        customStageCount++;
        
        const stageEl = document.createElement('div');
        stageEl.className = 'pipeline-stage';
        stageEl.dataset.stageIndex = customStageCount;
        
        stageEl.innerHTML = `
            <span>Stage ${customStageCount}:</span>
            <select class="qubit-select">
                <option value="4">4 qubits</option>
                <option value="6">6 qubits</option>
                <option value="8">8 qubits</option>
            </select>
            <select class="topology-select">
                <option value="star">star</option>
                <option value="linear">linear</option>
                <option value="full">full</option>
            </select>
            <button class="remove-stage" data-stage="${customStageCount}">
                <i class="fas fa-trash"></i>
            </button>
        `;
        
        pipelineStagesContainer.appendChild(stageEl);
        
        // Add event listener to remove button
        const removeButton = stageEl.querySelector('.remove-stage');
        removeButton.addEventListener('click', (e) => {
            const stageToRemove = e.currentTarget.dataset.stage;
            removeCustomStage(stageToRemove);
        });
        
        // Update internal state
        updateInternalCustomPipeline();

        // Update button state
        updateCustomPipelineButtonState();
    };
    
     /**
      * Updates the internal `currentCustomPipeline` state based on the DOM.
      */
     const updateInternalCustomPipeline = () => {
         if (!pipelineStagesContainer) {
             currentCustomPipeline = [];
             return;
         }
         const stages = pipelineStagesContainer.querySelectorAll('.pipeline-stage');
         currentCustomPipeline = Array.from(stages).map(stage => {
             const qubits = parseInt(stage.querySelector('.qubit-select').value);
             const topology = stage.querySelector('.topology-select').value;
             return [qubits, topology];
         });
         console.log("Internal custom pipeline state updated:", currentCustomPipeline);
     };

    /**
     * Remove a stage from the custom pipeline
     */
    const removeCustomStage = (stageIndex) => {
        if (!pipelineStagesContainer) return;
        
        const stageEl = pipelineStagesContainer.querySelector(`.pipeline-stage[data-stage-index="${stageIndex}"]`);
        
        if (stageEl) {
            stageEl.remove();
            
            // Renumber remaining stages
            const remainingStages = pipelineStagesContainer.querySelectorAll('.pipeline-stage');
            remainingStages.forEach((stage, index) => {
                stage.querySelector('span').textContent = `Stage ${index + 1}:`;
                stage.dataset.stageIndex = index + 1;
                stage.querySelector('.remove-stage').dataset.stage = index + 1;
            });
            
            customStageCount = remainingStages.length;
            
            // Update internal state
            updateInternalCustomPipeline();

            // Update button state
            updateCustomPipelineButtonState();
        }
    };
    
    /**
     * Update the state of the run custom pipeline button
     */
    const updateCustomPipelineButtonState = () => {
        if (!runCustomButton) return;
        
        const stages = pipelineStagesContainer ? pipelineStagesContainer.querySelectorAll('.pipeline-stage') : [];
        runCustomButton.disabled = stages.length === 0;
    };
    
    /**
     * Run a custom pipeline
     */
    const runCustomPipeline = async () => {
        if (!pipelineStagesContainer) return;
        
        const stages = pipelineStagesContainer.querySelectorAll('.pipeline-stage');
        
        if (stages.length === 0) {
            alert('Please add at least one stage to your pipeline.');
            return;
        }
        
        // Build pipeline configuration
        const config = Array.from(stages).map(stage => {
            const qubits = parseInt(stage.querySelector('.qubit-select').value);
            const topology = stage.querySelector('.topology-select').value;
            return [qubits, topology];
        });
        
        UI.showLoading('Running custom pipeline...');

        // Get active project ID
        const activeProjectId = (typeof ProjectsModule !== 'undefined') ? ProjectsModule.getActiveProjectId() : null;
        console.log("Running custom pipeline for active project:", activeProjectId);
        
        const response = await API.runPipeline(config, { activeProjectId: activeProjectId });
        
        UI.hideLoading();
        
        if (response.success) {
            displayPipelineResults(response.data, 'Custom Pipeline');
             // Update dashboard if a project is active
             if (activeProjectId && typeof ProjectsModule !== 'undefined' && ProjectsModule.updateDashboardProjectInfo) {
                 console.log("Pipeline run successful, updating dashboard project info...");
                 ProjectsModule.updateDashboardProjectInfo();
             }
        } else {
            alert(`Error running pipeline: ${response.error}`);
        }
    };
    
    /**
     * Display pipeline results with collapsible sections
     */
    const displayPipelineResults = (results, pipelineName) => {
        if (!pipelineResultContainer) return;
        
        // Format stage results
        const stagesHtml = results.stages.map((stage, index) => {
            return `
                <tr>
                    <td>${index + 1}</td>
                    <td>${stage.qubits} qubits (${stage.topology})</td>
                    <td>${UI.formatValue(stage.qfi_snr, 'decimal')}</td>
                    <td>${UI.formatValue(stage.max_qfi, 'decimal')}</td>
                    <td>${UI.formatValue(stage.execution_time, 'decimal')} s</td>
                </tr>
            `;
        }).join('');
        
        // Create HTML for result with collapsible sections
        const html = `
            <h3>${pipelineName} Results</h3>
            
            <div class="collapsible-section">
                <div class="collapsible-header">
                    <h4>Summary</h4>
                    <button class="collapse-toggle" aria-expanded="true" aria-controls="summary-content">
                        <i class="fas fa-chevron-up"></i>
                    </button>
                </div>
                <div id="summary-content" class="collapsible-content expanded">
                    <table>
                        <tr>
                            <th>Event</th>
                            <td>${results.event_name}</td>
                        </tr>
                        <tr>
                            <th>Pipeline Configuration</th>
                            <td>${results.pipeline_config}</td>
                        </tr>
                        <tr>
                            <th>First Stage SNR</th>
                            <td>${UI.formatValue(results.summary.first_stage_snr, 'decimal')}</td>
                        </tr>
                        <tr>
                            <th>Final Stage SNR</th>
                            <td>${UI.formatValue(results.summary.final_stage_snr, 'decimal')}</td>
                        </tr>
                        <tr>
                            <th>Improvement Factor</th>
                            <td>${UI.formatValue(results.summary.improvement_factor, 'decimal')}×</td>
                        </tr>
                        <tr>
                            <th>Total Execution Time</th>
                            <td>${UI.formatValue(results.summary.total_execution_time, 'decimal')} s</td>
                        </tr>
                    </table>
                </div>
            </div>
            
            <div class="collapsible-section">
                <div class="collapsible-header">
                    <h4>Stage Results</h4>
                    <button class="collapse-toggle" aria-expanded="false" aria-controls="stages-content">
                        <i class="fas fa-chevron-up"></i>
                    </button>
                </div>
                <div id="stages-content" class="collapsible-content collapsed">
                    <table>
                        <thead>
                            <tr>
                                <th>Stage</th>
                                <th>Configuration</th>
                                <th>SNR</th>
                                <th>Max QFI</th>
                                <th>Execution Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${stagesHtml}
                        </tbody>
                    </table>
                </div>
            </div>
            
            ${results.file_paths && results.file_paths.visualization ? `
                <div class="collapsible-section">
                    <div class="collapsible-header">
                        <h4>Visualization</h4>
                        <button class="collapse-toggle" aria-expanded="true" aria-controls="visualization-content">
                            <i class="fas fa-chevron-up"></i>
                        </button>
                    </div>
                    <div id="visualization-content" class="collapsible-content expanded">
                        <div class="result-visualization-container">
                            <img src="${VisualizationsModule.formatVisualizationPath(results.file_paths.visualization)}" alt="Pipeline Visualization"
                                 onerror="this.onerror=null; this.src=''; this.alt='Failed to load image'; this.style.display='none'; this.parentNode.innerHTML += '<p class=\\'error\\'>Failed to load visualization image.</p>';">
                        </div>
                    </div>
                </div>
            ` : ''}
        `;
        
        pipelineResultContainer.innerHTML = html;
        
        // Add event listeners for collapsible sections
        addCollapseListeners();
        
        pipelineResultContainer.scrollIntoView({ behavior: 'smooth' });
    };
    
    /**
     * Add event listeners for collapsible content sections
     */
    const addCollapseListeners = () => {
        if (!pipelineResultContainer) return;
        
        const toggleButtons = pipelineResultContainer.querySelectorAll('.collapse-toggle');
        toggleButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                
                // Get the content container this button controls
                const contentId = button.getAttribute('aria-controls');
                const contentEl = document.getElementById(contentId);
                
                // Get current state
                const isExpanded = button.getAttribute('aria-expanded') === 'true';
                
                // Toggle state
                button.setAttribute('aria-expanded', !isExpanded);
                if (isExpanded) {
                    contentEl.classList.remove('expanded');
                    contentEl.classList.add('collapsed');
                } else {
                    contentEl.classList.remove('collapsed');
                    contentEl.classList.add('expanded');
                }
            });
        });
        
        // Also make entire headers clickable (for better UX)
        const headers = pipelineResultContainer.querySelectorAll('.collapsible-header');
        headers.forEach(header => {
            header.addEventListener('click', (e) => {
                // Don't trigger if they clicked directly on the button (it has its own handler)
                if (e.target.closest('.collapse-toggle')) return;
                
                // Simulate click on the header's button
                const button = header.querySelector('.collapse-toggle');
                button.click();
            });
        });
    };
    
    /**
     * Initialize pipeline module
     */
    const init = () => {
        // Load presets
        loadPresets();
        
        // Set up event listeners
        if (presetSelect) {
            presetSelect.addEventListener('change', updatePresetDescription);
        }
        
        if (runPresetButton) {
            runPresetButton.addEventListener('click', runPresetPipeline);
        }
        
        if (addStageButton) {
            addStageButton.addEventListener('click', addCustomStage);
        }
        
        if (runCustomButton) {
            runCustomButton.addEventListener('click', runCustomPipeline);
            updateCustomPipelineButtonState();
        }
        
        // Restore or initialize custom pipeline state
        if (currentCustomPipeline.length > 0) {
             // If state exists (e.g., from previous interaction), restore it
             setCustomPipeline(currentCustomPipeline);
        } else if (pipelineStagesContainer && pipelineStagesContainer.children.length === 0) {
             // Otherwise, add one stage by default only if the container is empty
             addCustomStage(); // This will also call updateInternalCustomPipeline
        } else {
             // If container has stages but state is empty, sync state from DOM
             updateInternalCustomPipeline();
        }
        updateCustomPipelineButtonState(); // Ensure button state is correct on init
    };
    
    /**
     * Set custom pipeline configuration
     * This is used when loading a project
     */
    const setCustomPipeline = (pipelineConfig) => {
        if (!pipelineStagesContainer) return;
        
        // Clear existing stages
        pipelineStagesContainer.innerHTML = '';
        customStageCount = 0;
        
        // Add each stage from the configuration
        for (const [qubits, topology] of pipelineConfig) {
            customStageCount++;
            
            const stageEl = document.createElement('div');
            stageEl.className = 'pipeline-stage';
            stageEl.dataset.stageIndex = customStageCount;
            
            stageEl.innerHTML = `
                <span>Stage ${customStageCount}:</span>
                <select class="qubit-select">
                    <option value="4" ${qubits === 4 ? 'selected' : ''}>4 qubits</option>
                    <option value="6" ${qubits === 6 ? 'selected' : ''}>6 qubits</option>
                    <option value="8" ${qubits === 8 ? 'selected' : ''}>8 qubits</option>
                </select>
                <select class="topology-select">
                    <option value="star" ${topology === 'star' ? 'selected' : ''}>star</option>
                    <option value="linear" ${topology === 'linear' ? 'selected' : ''}>linear</option>
                    <option value="full" ${topology === 'full' ? 'selected' : ''}>full</option>
                </select>
                <button class="remove-stage" data-stage="${customStageCount}">
                    <i class="fas fa-trash"></i>
                </button>
            `;
            
            pipelineStagesContainer.appendChild(stageEl);
            
            // Add event listener to remove button
            const removeButton = stageEl.querySelector('.remove-stage');
            removeButton.addEventListener('click', (e) => {
                const stageToRemove = e.currentTarget.dataset.stage;
                removeCustomStage(stageToRemove);
            });
        }
        
        // Update button state
        updateCustomPipelineButtonState();
        
        // Update internal state as well
        currentCustomPipeline = [...pipelineConfig];
        console.log("Internal custom pipeline state set by project load:", currentCustomPipeline);

        // Switch to custom pipeline tab if it exists
        const customPipelineTab = document.querySelector('.tab-btn[data-tab="custom"]');
        if (customPipelineTab) {
            UI.switchTab(customPipelineTab);
        }
    };
    
    // Return public interface
    return {
        init,
        loadPresets,
        runPresetPipeline,
        runCustomPipeline,
        setCustomPipeline
    };
})();

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
    // PipelinesModule will be initialized by the main app.js
});

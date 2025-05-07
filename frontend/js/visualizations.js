/**
 * Visualizations Module for quantum gravitational wave detector
 * Handles visualization rendering and formatting
 */

const VisualizationsModule = (() => {
    /**
     * Format the file path for viewing
     * This ensures the path is relative to the server root and points to the /data mount.
     */
    const formatVisualizationPath = (path) => {
        if (!path) {
            console.error("formatVisualizationPath received null or empty path");
            return null;
        }
        console.log("Original viz path:", path);

        // Check if it's already a valid absolute URL
        if (path.startsWith('http://') || path.startsWith('https://')) {
            console.log("Path is already a URL:", path);
            return path;
        }

        // Remove any leading './' or '/'
        let relativePath = path.startsWith('./') ? path.substring(2) : path;
        relativePath = relativePath.startsWith('/') ? relativePath.substring(1) : relativePath;

        // Remove potential incorrect prefixes like 'frontend/'
        if (relativePath.startsWith('frontend/')) {
            relativePath = relativePath.substring('frontend/'.length);
        }

        // Ensure it starts with 'data/' if it contains 'experiments' or 'plots'
        if (relativePath.includes('experiments/') || relativePath.includes('plots/')) {
            if (!relativePath.startsWith('data/')) {
                 // If it's missing 'data/', prepend it. This handles cases like 'experiments/...'
                 relativePath = 'data/' + relativePath;
            }
        } else {
             // If it doesn't seem to be an experiment/plot path, maybe it's already relative to /data?
             // Or it's an unexpected path. For safety, prepend /data/ if not already there.
             if (!relativePath.startsWith('data/')) {
                  console.warn("Unexpected viz path format, prepending /data/ :", path);
                  relativePath = 'data/' + relativePath;
             }
        }


        // Ensure the final path starts with a single '/'
        const finalPath = '/' + relativePath;
        console.log("Formatted viz path:", finalPath);
        return finalPath;
    };
    
    /**
     * Load and display visualization for a specific run within a project.
     */
    const loadRunVisualization = async (runTimestamp, runDetails) => {
        const container = document.getElementById('visualization-container');
        if (!container) return;

        console.log("Loading visualization for run:", runTimestamp, runDetails);

        if (runDetails.error) {
             createDefaultVisualization(container, `Error loading run details: ${runDetails.error}`);
             return;
        }

        const vizPath = runDetails.visualization_path;

        if (!vizPath) {
            createDefaultVisualization(container, 'No visualization available for this run.');
            return;
        }

        console.log('Using visualization path:', vizPath);
        const formattedPath = formatVisualizationPath(vizPath);
        console.log('Final formatted path for img src:', formattedPath); // Log the final path

        // Display the visualization
        container.innerHTML = `
            <h4>Run: ${runTimestamp} (${runDetails.pipeline_config_str || 'Unknown Pipeline'})</h4>
            <div class="visualization-image">
                <img src="${formattedPath}" alt="Run Visualization"
                     onerror="console.error('Image load error for path:', '${formattedPath}'); this.onerror=null; this.src=''; this.alt='Failed to load image'; this.style.display='none'; this.parentNode.innerHTML += '<p class=\\'error\\'>Failed to load visualization image. Path: ${formattedPath}</p>';">
            </div>
        `;

        // Add controls (download, fullscreen)
        const imgContainer = container.querySelector('.visualization-image');
        if (imgContainer) {
            addVisualizationControls(imgContainer, formattedPath, `visualization_${runTimestamp}.png`);
        }
    };

    /**
     * Populate the runs dropdown/list for the selected project.
     */
    const populateRunsList = async (projectId) => {
        const runsSelect = document.getElementById('visualization-run-select'); // Need to add this element
        const container = document.getElementById('visualization-container');
        if (!runsSelect || !container) return;

        runsSelect.innerHTML = '<option>Loading runs...</option>';
        container.innerHTML = ''; // Clear previous visualization
        createDefaultVisualization(container, 'Select a run to view visualization.'); // Show default message

        console.log(`[populateRunsList] Fetching runs for project ${projectId}`);
        const response = await API.getProjectRuns(projectId);
        console.log(`[populateRunsList] Runs response:`, response);


        if (!response.success) {
            console.error(`[populateRunsList] Error loading runs: ${response.error}`);
            runsSelect.innerHTML = `<option disabled selected>Error loading runs: ${response.error}</option>`;
            return;
        }

        const runs = response.data;
        if (!runs || runs.length === 0) {
            runsSelect.innerHTML = '<option disabled selected>No runs found for this project</option>';
            return;
        }

        runsSelect.innerHTML = '<option value="" disabled selected>Select a run...</option>'; // Default prompt
        runs.forEach(run => {
            const option = document.createElement('option');
            option.value = run.run_timestamp;
            // Display timestamp and maybe pipeline config
            const dateStr = run.run_timestamp.substring(0, 8);
            const timeStr = run.run_timestamp.substring(9, 15);
            option.textContent = `${dateStr.replace(/(\d{4})(\d{2})(\d{2})/, '$1-$2-$3')} ${timeStr.replace(/(\d{2})(\d{2})(\d{2})/, '$1:$2:$3')} (${run.pipeline_config_str || 'Unknown'})`;
            option.dataset.details = JSON.stringify(run); // Store details on the option
            runsSelect.appendChild(option);
        });

        // Add event listener to load visualization when a run is selected
        runsSelect.onchange = (e) => {
            const selectedOption = e.target.selectedOptions[0];
            if (selectedOption && selectedOption.value) {
                const runTimestamp = selectedOption.value;
                const runDetails = JSON.parse(selectedOption.dataset.details);
                loadRunVisualization(runTimestamp, runDetails);
            }
        };
    };
    
    /**
     * Create a download link for visualization
     */
    const createDownloadLink = (path, filename) => {
        if (!path) return null;
        
        const downloadLink = document.createElement('a');
        downloadLink.href = formatVisualizationPath(path);
        downloadLink.download = filename || 'visualization.png';
        downloadLink.className = 'download-link';
        downloadLink.innerHTML = '<i class="fas fa-download"></i> Download visualization';
        
        return downloadLink;
    };
    
    /**
     * Add visualization controls to a container
     */
    const addVisualizationControls = (container, imagePath, filename) => {
        if (!container || !imagePath) return;
        
        // Create controls container
        const controlsContainer = document.createElement('div');
        controlsContainer.className = 'visualization-controls';
        
        // Add download link
        const downloadLink = createDownloadLink(imagePath, filename);
        if (downloadLink) {
            controlsContainer.appendChild(downloadLink);
        }
        
        // Add fullscreen button
        const fullscreenButton = document.createElement('button');
        fullscreenButton.className = 'visualization-fullscreen';
        fullscreenButton.innerHTML = '<i class="fas fa-expand"></i> View fullscreen';
        fullscreenButton.addEventListener('click', () => {
            openFullscreenViewer(imagePath);
        });
        
        controlsContainer.appendChild(fullscreenButton);
        
        // Add to container
        container.appendChild(controlsContainer);
    };
    
    /**
     * Open a fullscreen viewer for the visualization
     */
    const openFullscreenViewer = (imagePath) => {
        // Create fullscreen overlay
        const overlay = document.createElement('div');
        overlay.className = 'fullscreen-overlay';
        
        overlay.innerHTML = `
            <div class="fullscreen-content">
                <button class="close-fullscreen">×</button>
                <img src="${formatVisualizationPath(imagePath)}" alt="Visualization">
            </div>
        `;
        
        // Add to body
        document.body.appendChild(overlay);
        
        // Prevent scrolling of the body
        document.body.style.overflow = 'hidden';
        
        // Add close button event
        const closeButton = overlay.querySelector('.close-fullscreen');
        closeButton.addEventListener('click', () => {
            document.body.removeChild(overlay);
            document.body.style.overflow = '';
        });
        
        // Close on overlay click (but not on image click)
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                document.body.removeChild(overlay);
                document.body.style.overflow = '';
            }
        });
        
        // Add fullscreen styles if not already added
        if (!document.getElementById('fullscreen-styles')) {
            const styleEl = document.createElement('style');
            styleEl.id = 'fullscreen-styles';
            styleEl.textContent = `
                .fullscreen-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background-color: rgba(0, 0, 0, 0.9);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    z-index: 9999;
                }
                .fullscreen-content {
                    position: relative;
                    max-width: 90vw;
                    max-height: 90vh;
                }
                .fullscreen-content img {
                    max-width: 100%;
                    max-height: 90vh;
                    object-fit: contain;
                    display: block;
                }
                .close-fullscreen {
                    position: absolute;
                    top: -40px;
                    right: -40px;
                    width: 40px;
                    height: 40px;
                    background: none;
                    border: none;
                    color: white;
                    font-size: 24px;
                    cursor: pointer;
                }
                .visualization-controls {
                    display: flex;
                    gap: 1rem;
                    margin-top: 0.5rem;
                }
                .download-link,
                .visualization-fullscreen {
                    display: inline-flex;
                    align-items: center;
                    gap: 0.5rem;
                    font-size: 0.9rem;
                    color: var(--primary-color);
                    text-decoration: none;
                    background: none;
                    border: none;
                    cursor: pointer;
                    padding: 0;
                }
                .download-link:hover,
                .visualization-fullscreen:hover {
                    text-decoration: underline;
                }
            `;
            document.head.appendChild(styleEl);
        }
    };
    
    /**
     * Process visualization containers in the document
     * This adds controls to all visualization images
     */
    const processVisualizationContainers = () => {
        const containers = document.querySelectorAll('.result-visualization, .sweep-visualization');
        
        containers.forEach(container => {
            const img = container.querySelector('img');
            if (img && img.src) {
                // Remove existing controls if any
                const existingControls = container.querySelector('.visualization-controls');
                if (existingControls) {
                    container.removeChild(existingControls);
                }
                
                // Add new controls
                addVisualizationControls(container, img.src);
            }
        });
    };
    
    /**
     * Populate project select dropdown for visualization panel.
     */
    const populateProjectSelect = async () => {
        const projectSelect = document.getElementById('visualization-project-select');
        if (!projectSelect) return;

        projectSelect.innerHTML = '<option>Loading projects...</option>'; // Loading state

        const response = await API.listProjects();

        if (!response.success || !response.data || response.data.length === 0) {
            projectSelect.innerHTML = '<option disabled selected>No saved projects found</option>';
            // Clear runs list and visualization if no projects
            const runsSelect = document.getElementById('visualization-run-select');
            if (runsSelect) runsSelect.innerHTML = '<option disabled selected>Select a project first</option>';
            const container = document.getElementById('visualization-container');
            if (container) createDefaultVisualization(container, 'Create or select a project.');
            return;
        }

        projectSelect.innerHTML = '<option value="" disabled selected>Select a project...</option>'; // Default prompt
        const projects = response.data;
        projects.forEach(project => {
            const option = document.createElement('option');
            option.value = project.id;
            option.textContent = project.name || 'Unnamed Project';
            projectSelect.appendChild(option);
        });

        // Add event listener to load runs when a project is selected
        projectSelect.onchange = (e) => {
            const projectId = e.target.value;
            if (projectId) {
                populateRunsList(projectId);
            }
        };
    };
    
    /**
     * Initialize visualizations module.
     */
    const init = () => {
        console.log("Initializing VisualizationsModule...");
        // No loadVisualizationButton needed anymore, selection triggers load

        // Load projects when visualization panel is shown
        const navVisualizations = document.getElementById('nav-visualizations');
        if (navVisualizations) {
            console.log("Visualizations nav link found, adding listener.");
            navVisualizations.addEventListener('click', () => {
                console.log("Visualizations tab clicked, populating project select.");
                populateProjectSelect(); // Populate projects when tab is clicked
                // Also clear/reset the runs dropdown and visualization area
                const runsSelect = document.getElementById('visualization-run-select');
                 if (runsSelect) runsSelect.innerHTML = '<option disabled selected>Select a project first</option>';
                 const container = document.getElementById('visualization-container');
                 if (container) createDefaultVisualization(container, 'Select a project and run.');
            });
        } else {
             console.warn("Visualizations nav link (#nav-visualizations) not found.");
        }
        
        // Add mutation observer to detect when visualizations are added to the DOM
        const observer = new MutationObserver((mutations) => {
            let shouldProcess = false;
            
            mutations.forEach(mutation => {
                // Check if any nodes were added
                if (mutation.addedNodes.length > 0) {
                    // Check if any of the added nodes contain visualizations
                    for (let i = 0; i < mutation.addedNodes.length; i++) {
                        const node = mutation.addedNodes[i];
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            if (node.classList.contains('result-visualization') ||
                                node.classList.contains('sweep-visualization') ||
                                node.querySelector('.result-visualization, .sweep-visualization')) {
                                shouldProcess = true;
                                break;
                            }
                        }
                    }
                }
            });
            
            if (shouldProcess) {
                processVisualizationContainers();
            }
        });
        
        // Observe changes to the entire document
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
        
        // Process any existing visualizations
        processVisualizationContainers();
    };
    
    // Return public interface
    /**
     * Create a default visualization with a message
     */
    const createDefaultVisualization = (container, message = 'No visualization available') => {
        if (!container) return;
        
        container.innerHTML = `
            <div class="no-visualization">
                <i class="fas fa-image"></i>
                <p>${message}</p>
            </div>
        `;
    };

    /**
     * Find visualization path in project data
     */
    const findVisualizationPath = (projectData) => {
        // Output the project structure to console for debugging
        console.log('Project data structure:', JSON.stringify(projectData, null, 2));
        
        // Check multiple possible locations for visualization path
        if (projectData.file_paths && projectData.file_paths.visualization) {
            console.log('Found viz path in file_paths.visualization:', projectData.file_paths.visualization);
            return projectData.file_paths.visualization;
        }
        
        if (projectData.configuration && projectData.configuration.file_paths && 
            projectData.configuration.file_paths.visualization) {
            console.log('Found viz path in configuration.file_paths.visualization:', 
                      projectData.configuration.file_paths.visualization);
            return projectData.configuration.file_paths.visualization;
        }
        
        // Try to find in results
        if (projectData.configuration && projectData.configuration.results && 
            projectData.configuration.results.file_paths && 
            projectData.configuration.results.file_paths.visualization) {
            console.log('Found viz path in configuration.results.file_paths.visualization:', 
                      projectData.configuration.results.file_paths.visualization);
            return projectData.configuration.results.file_paths.visualization;
        }
        
        // Check in stages for older projects
        if (projectData.stages && projectData.stages.length > 0) {
            for (const stage of projectData.stages) {
                if (stage.file_paths && stage.file_paths.visualization) {
                    console.log('Found viz path in stage.file_paths.visualization:', 
                              stage.file_paths.visualization);
                    return stage.file_paths.visualization;
                }
            }
        }
        
        console.log('No visualization path found in project data');
        return null;
    };
    
    return {
        init,
        formatVisualizationPath,
        openFullscreenViewer,
        processVisualizationContainers,
        // loadProjectVisualization, // Removed - no longer used directly
        populateProjectSelect,
        findVisualizationPath, // Keep utility function if needed elsewhere
        createDefaultVisualization,
        loadRunVisualization // Expose if needed externally (maybe not)
    };
})();

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
    // VisualizationsModule will be initialized by the main app.js
});

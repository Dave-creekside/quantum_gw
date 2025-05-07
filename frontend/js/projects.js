/**
 * Projects Module for quantum gravitational wave detector
 * Handles project creation, activation, and management.
 */

const ProjectsModule = (() => {
    // DOM elements
    const createProjectButton = document.getElementById('create-project-button'); // Renamed from save-project-button
    const projectNameInput = document.getElementById('project-name');
    const projectsListContainer = document.getElementById('projects-list-container');
    const activeProjectDisplay = document.getElementById('active-project-display');

    // State
    let activeProjectId = null;
    let activeProjectName = null;

    /**
     * Create a new project workspace.
     */
    const createNewProject = async () => {
        console.log("Attempting to create a new project...");
        const name = projectNameInput ? projectNameInput.value.trim() : '';

        if (!name) {
            alert('Please enter a project name.');
            console.error("Create failed: Project name is empty.");
            return false;
        }
        console.log("New project name:", name);

        // Get current configuration to use as base
        console.log("Fetching current config for base configuration...");
        const configResponse = await API.getConfig();
        if (!configResponse.success) {
            alert('Failed to get current configuration for the new project.');
            console.error("Create failed: Could not get current config.", configResponse.error);
            return false;
        }
        console.log("Base config fetched:", configResponse.data);

        const currentPipeline = getCurrentPipelineConfig();
        console.log("Base pipeline config:", currentPipeline);

        const baseConfiguration = {
            event_name: configResponse.data.event_name,
            parameters: { ...configResponse.data },
            pipeline_config: currentPipeline,
        };
        console.log("Base configuration object for new project:", baseConfiguration);

        // Create project via API
        UI.showLoading('Creating project...');
        console.log("Calling API.createProject...");
        const createResponse = await API.createProject(name, baseConfiguration);
        UI.hideLoading();
        console.log("Create project API response:", createResponse);

        if (createResponse.success) {
            alert(`Project "${name}" created successfully!`);
            if (projectNameInput) {
                projectNameInput.value = ''; // Clear input
            }
            loadProjectsList(); // Refresh the list
            // Automatically activate the newly created project
            if (createResponse.data && createResponse.data.project_id) {
                activateProject(createResponse.data.project_id, name);
            }
            return true;
        } else {
            alert(`Error creating project: ${createResponse.error || 'Unknown error'}`);
            return false;
        }
    };
    
    /**
     * Get the current pipeline configuration from UI
     */
    const getCurrentPipelineConfig = () => {
        // Try to get from pipeline stages UI
        const stages = document.querySelectorAll('.pipeline-stage');
        if (stages && stages.length > 0) {
            return Array.from(stages).map(stage => {
                const qubits = parseInt(stage.querySelector('.qubit-select').value);
                const topology = stage.querySelector('.topology-select').value;
                return [qubits, topology];
            });
        }
        
        // Otherwise, use a default configuration
        return [[4, "star"]];
    };
    
    /**
     * Load the list of saved projects.
     */
    const loadProjectsList = async () => {
        if (!projectsListContainer) {
            console.log("Projects list container element not found, skipping load.");
            return;
        }

        console.log("Loading projects list...");
        projectsListContainer.innerHTML = '<p>Loading projects...</p>'; // Show loading state
        // No need for global loading indicator here, just local feedback

        const response = await API.listProjects();
        console.log("Load projects list API response:", response);

        if (response.success) {
            console.log("Projects data received:", response.data);
            displayProjectsList(response.data);
        } else {
            console.error("Error loading projects:", response.error);
            projectsListContainer.innerHTML = `<p class="error">Error loading projects: ${response.error}</p>`;
        }
    };
    
    /**
     * Display the list of projects in the UI.
     */
    const displayProjectsList = (projects) => {
        if (!projectsListContainer) return;

        if (!projects || projects.length === 0) {
            projectsListContainer.innerHTML = '<p>No saved projects found.</p>';
            return;
        }

        // Format date for display (using created_timestamp)
        const formatDate = (timestamp) => {
             if (!timestamp || timestamp.length < 8) return 'Unknown date';
             // Assumes format YYYYMMDD_HHMMSS
             const year = timestamp.substring(0, 4);
             const month = timestamp.substring(4, 6);
             const day = timestamp.substring(6, 8);
             let timeStr = '';
             if (timestamp.includes('_') && timestamp.length >= 15) {
                 const time = timestamp.substring(9, 15);
                 const hours = time.substring(0, 2);
                 const minutes = time.substring(2, 4);
                 timeStr = ` ${hours}:${minutes}`;
             }
             return `${year}-${month}-${day}${timeStr}`;
        };

        // Create HTML for projects list
        let html = '<ul class="projects-list">';
        for (const project of projects) {
            // Add 'active' class if this project is the currently active one
            const isActive = project.id === activeProjectId;
            html += `
                <li class="project-item ${isActive ? 'active' : ''}" data-id="${project.id}" data-name="${project.name}">
                    <div class="project-details">
                        <h4>${project.name}</h4>
                        <div class="project-meta">Created: ${formatDate(project.timestamp)}</div>
                    </div>
                    <div class="project-actions">
                        <button class="activate-project-button" data-id="${project.id}" data-name="${project.name}">Activate</button>
                        <!-- Add delete button later if needed -->
                    </div>
                </li>
            `;
        }
        html += '</ul>';
        projectsListContainer.innerHTML = html;

        // Add event listeners to activate buttons
        const activateButtons = projectsListContainer.querySelectorAll('.activate-project-button');
        activateButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const projectId = e.currentTarget.dataset.id;
                const projectName = e.currentTarget.dataset.name;
                activateProject(projectId, projectName);
            });
        });
    };
    
    /**
     * Activate a project: Set it as the current workspace context.
     */
    const activateProject = async (projectId, projectName) => {
        console.log(`Activating project: ${projectName} (ID: ${projectId})`);
        UI.showLoading('Activating project...');

        const response = await API.loadProject(projectId);
        UI.hideLoading();

        if (response.success) {
            activeProjectId = projectId;
            activeProjectName = projectName;
            console.log("Active project set:", { id: activeProjectId, name: activeProjectName });

            // Update UI to reflect active project
            updateActiveProjectDisplay();
            highlightActiveProjectInList(projectId);

            // Load base configuration into UI
            const projectData = response.data;
            const baseConfig = projectData.base_configuration?.parameters || {};
            const basePipeline = projectData.base_configuration?.pipeline_config || [];

            console.log("Loading base config:", baseConfig);
            const configUpdateResponse = await API.updateConfig(baseConfig);
            if (!configUpdateResponse.success) {
                alert(`Error loading project configuration: ${configUpdateResponse.error}`);
                // Continue activation but log error
            } else {
                 // Refresh config display if module exists
                if (typeof ConfigModule !== 'undefined' && ConfigModule.loadConfig) {
                    ConfigModule.loadConfig();
                }
            }

            console.log("Loading base pipeline:", basePipeline);
            if (typeof PipelinesModule !== 'undefined' && PipelinesModule.setCustomPipeline) {
                PipelinesModule.setCustomPipeline(basePipeline);
            }

            // Update dashboard
            updateDashboardProjectInfo(); // Call the function to update dashboard

            alert(`Project "${projectName}" activated.`);
        } else {
            alert(`Error activating project: ${response.error}`);
            activeProjectId = null; // Clear active project on error
            activeProjectName = null;
            updateActiveProjectDisplay();
            highlightActiveProjectInList(null);
        }
    };

    // Removed startNewProject function as it's redundant with create + activate

    /**
     * Update the display showing the currently active project.
     */
    const updateActiveProjectDisplay = () => {
        if (activeProjectDisplay) {
            if (activeProjectId && activeProjectName) {
                activeProjectDisplay.textContent = `Active Project: ${activeProjectName}`;
                activeProjectDisplay.style.display = 'block';
            } else {
                activeProjectDisplay.textContent = 'Active Project: None (Unsaved)';
                 activeProjectDisplay.style.display = 'block'; // Keep it visible
            }
        }
    };

     /**
      * Highlight the active project in the list.
      */
     const highlightActiveProjectInList = (projectId) => {
         if (!projectsListContainer) return;
         const items = projectsListContainer.querySelectorAll('.project-item');
         items.forEach(item => {
             if (item.dataset.id === projectId) {
                 item.classList.add('active');
             } else {
                 item.classList.remove('active');
             }
         });
     };

    /**
     * Update the dashboard project info card.
     */
    const updateDashboardProjectInfo = async () => {
        console.log("Updating dashboard project info...", { activeProjectId, activeProjectName });
        const container = document.getElementById('dashboard-active-project');
        if (!container) {
            console.warn("Dashboard active project container not found.");
            return;
        }

        if (!activeProjectId) {
            container.innerHTML = `
                <h4>Unsaved Project</h4>
                <p>Create or activate a project from the Projects panel.</p>
                <div class="dashboard-metric awaiting-run">
                    <span>SNR:</span> Awaiting Run
                </div>
                <div class="dashboard-metric awaiting-run">
                    <span>Max QFI:</span> Awaiting Run
                </div>
                <div class="dashboard-metric">
                     <span>LIGO Event:</span> <span id="dashboard-project-event">N/A</span>
                </div>
                <div id="dashboard-event-plot-container"></div>
            `;
            // Update event name based on current config
            const config = await API.getConfig();
            const eventSpan = document.getElementById('dashboard-project-event');
            if (eventSpan && config.success) {
                 eventSpan.textContent = config.data.event_name || 'N/A';
                 // TODO: Add call to fetch and display event plot
            }
            return;
        }

        // Fetch project details and associated runs
        container.innerHTML = `<p>Loading project details...</p>`; // Loading state
        console.log(`[updateDashboardProjectInfo] Fetching runs for project ${activeProjectId}`);
        const runsResponse = await API.getProjectRuns(activeProjectId);
        console.log(`[updateDashboardProjectInfo] Runs response:`, runsResponse);

        if (!runsResponse.success) {
            console.error(`[updateDashboardProjectInfo] Error loading runs: ${runsResponse.error}`);
            container.innerHTML = `<p class="error">Error loading project runs: ${runsResponse.error}</p>`;
            return;
        }

        const runs = runsResponse.data;
        const latestRun = (runs && runs.length > 0) ? runs[0] : null; // Assumes sorted newest first

        // Fetch CURRENT config for event name
        const configResponse = await API.getConfig();
        const eventName = configResponse.success ? (configResponse.data.event_name || 'N/A') : 'N/A';
        console.log(`[updateDashboardProjectInfo] Current event name from config: ${eventName}`);

        let snrHtml = '<div class="dashboard-metric awaiting-run"><span>SNR:</span> Awaiting Run</div>';
        let qfiHtml = '<div class="dashboard-metric awaiting-run"><span>Max QFI:</span> Awaiting Run</div>';

        if (latestRun && !latestRun.error) {
            snrHtml = `<div class="dashboard-metric"><span>SNR:</span> ${UI.formatValue(latestRun.final_snr, 'decimal')}</div>`;
            qfiHtml = `<div class="dashboard-metric"><span>Max QFI:</span> ${UI.formatValue(latestRun.max_qfi, 'decimal')}</div>`;
        } else if (latestRun && latestRun.error) {
             snrHtml = `<div class="dashboard-metric error"><span>SNR:</span> Error</div>`;
             qfiHtml = `<div class="dashboard-metric error"><span>Max QFI:</span> Error</div>`;
        }

        container.innerHTML = `
            <h4>${activeProjectName || 'Unnamed Project'}</h4>
            ${snrHtml}
            ${qfiHtml}
            <div class="dashboard-metric">
                 <span>LIGO Event:</span> ${eventName}
            </div>
            <div id="dashboard-event-plot-container">
                 <!-- TODO: Add event plot image here -->
                 <p><small>Event data plot coming soon...</small></p>
            </div>
        `;
         // TODO: Add call to fetch and display event plot for 'eventName'
    };
    
    // Removed visualizeProject function as visualization is now handled differently

    /**
     * Initialize the projects module.
     */
    const init = () => {
        console.log("Initializing ProjectsModule...");

        // Rename "Save Project" to "Create Project"
        if (createProjectButton) {
            console.log("Create project button found, adding listener.");
            createProjectButton.addEventListener('click', () => {
                console.log("Create Project button clicked!"); // Add log here
                createNewProject();
            });
        } else {
            console.warn("Create project button (#create-project-button) not found.");
        }

        if (!projectNameInput) {
             console.warn("Project name input (#project-name) not found.");
        }

        // Removed listener for startNewProjectButton

        // Load projects list if the container element exists
        if (projectsListContainer) {
            console.log("Projects list container (#projects-list-container) found, calling loadProjectsList.");
            loadProjectsList(); // Initial load
        } else {
            console.error("Projects list container (#projects-list-container) not found in the DOM.");
        }

        // Initial display for active project
        updateActiveProjectDisplay();
    };

    // Public interface
    return {
        init,
        createNewProject,
        loadProjectsList,
        activateProject,
        // Removed startNewProject from export
        getActiveProjectId: () => activeProjectId, // Function to get the current active project ID
        updateDashboardProjectInfo // Expose the dashboard update function
    };
})();

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
    // ProjectsModule will be initialized by the main app.js
});

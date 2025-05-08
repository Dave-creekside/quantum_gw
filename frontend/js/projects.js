/**
 * Projects Module for quantum gravitational wave detector
 * Handles project creation, activation, and management.
 */

const ProjectsModule = (() => {
    // DOM elements
    const createProjectButton = document.getElementById('create-project-button');
    const projectNameInput = document.getElementById('project-name');
    const projectsListContainer = document.getElementById('projects-list-container');
    const activeProjectDisplay = document.getElementById('active-project-display');

    // State
    let activeProjectId = null;
    let activeProjectData = null; // Store the full data of the active project

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
        const currentParameters = ConfigModule.getCurrentConfigValues(); // Use ConfigModule
        const currentPipeline = PipelinesModule.getCurrentPipeline(); // Use PipelinesModule
        console.log("Base parameters fetched:", currentParameters);
        console.log("Base pipeline config:", currentPipeline);

        const baseConfiguration = {
            // event_name is part of parameters now
            parameters: currentParameters,
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
                // Need the full project data to activate, API returns it now
                activateProject(createResponse.data.project_id, createResponse.data.project_data);
            }
            return true;
        } else {
            alert(`Error creating project: ${createResponse.error || 'Unknown error'}`);
            return false;
        }
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
                <li class="project-item ${isActive ? 'active' : ''}" data-id="${project.id}">
                    <div class="project-details">
                        <h4>${project.name}</h4>
                        <div class="project-meta">Created: ${formatDate(project.timestamp)}</div>
                    </div>
                    <div class="project-actions">
                        <button class="activate-project-button" data-id="${project.id}">Activate</button>
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
            button.addEventListener('click', async (e) => {
                const projectId = e.currentTarget.dataset.id;
                UI.showLoading('Activating project...');
                const response = await API.loadProject(projectId); // Load full data on click
                UI.hideLoading();
                if (response.success) {
                    activateProject(projectId, response.data); // Pass full data
                } else {
                    alert(`Error loading project: ${response.error}`);
                }
            });
        });
    };

    /**
     * Activate a project: Load its config and update UI
     * Now accepts full project data object.
     */
    const activateProject = (projectId, projectData) => {
        if (!projectData || !projectId) {
            console.error("Activation failed: Invalid project ID or data.");
            return;
        }
        if (activeProjectId === projectId) {
            console.log(`Project ${projectId} is already active.`);
            return; // Already active
        }

        console.log(`Activating project: ${projectData.name} (ID: ${projectId})`);
        activeProjectId = projectId;
        activeProjectData = projectData; // Store the full project data

        console.log('Activating project data:', activeProjectData);

        // Update the main application configuration
        if (activeProjectData.base_configuration && activeProjectData.base_configuration.parameters) {
            ConfigModule.setConfigValues(activeProjectData.base_configuration.parameters);
            console.log("Loaded project parameters into ConfigModule.");
        } else {
             console.warn("Project loaded, but base configuration parameters are missing.");
             // Optionally reset config to defaults or leave as is
        }

        // Update the pipeline builder
        if (activeProjectData.base_configuration && activeProjectData.base_configuration.pipeline_config) {
            PipelinesModule.loadPipeline(activeProjectData.base_configuration.pipeline_config);
            console.log("Loaded project pipeline into PipelinesModule.");
        } else {
             console.warn("Project loaded, but base pipeline configuration is missing.");
             // Optionally clear the pipeline builder
             PipelinesModule.clearPipeline();
        }

        // Update the UI
        updateActiveProjectDisplay();
        highlightActiveProjectInList(projectId); // Highlight in the main list
        UI.updateDashboardActiveProject(); // Update dashboard card

        // Optionally, trigger updates in other modules like Visualizations
        if (typeof VisualizationsModule !== 'undefined' && VisualizationsModule.loadProjectRuns) {
            VisualizationsModule.loadProjectRuns(projectId);
        }

        console.log(`Project '${activeProjectData.name}' activated.`);
        // No alert needed here as it's called internally now mostly
    };

    /**
     * Update the display showing the currently active project.
     */
    const updateActiveProjectDisplay = () => {
        if (activeProjectDisplay) {
            if (activeProjectData) {
                activeProjectDisplay.textContent = `Active Project: ${activeProjectData.name}`;
                activeProjectDisplay.style.display = 'block';
            } else {
                activeProjectDisplay.textContent = 'Active Project: None (Unsaved)';
                 activeProjectDisplay.style.display = 'block'; // Keep it visible
            }
        }
        // Also update the dashboard card
        UI.updateDashboardActiveProject();
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
     * Update the locally stored active project data (e.g., after saving state).
     */
    const updateActiveProjectData = (updatedData) => {
        if (activeProjectId && updatedData && activeProjectId === updatedData.id) {
            activeProjectData = updatedData;
            console.log("Locally stored active project data updated.");
        }
    };

    /**
     * Initialize the projects module.
     */
    const init = () => {
        console.log("Initializing ProjectsModule...");

        if (createProjectButton) {
            console.log("Create project button found, adding listener.");
            createProjectButton.addEventListener('click', createNewProject);
        } else {
            console.warn("Create project button (#create-project-button) not found.");
        }

        if (!projectNameInput) {
             console.warn("Project name input (#project-name) not found.");
        }

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
        getActiveProjectId: () => activeProjectId,
        getActiveProject: () => activeProjectData, // Expose full active project data
        updateActiveProjectData // Expose function to update local data
    };
})();

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
    // ProjectsModule will be initialized by the main app.js
});

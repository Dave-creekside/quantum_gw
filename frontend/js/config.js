/**
 * Configuration Module for quantum gravitational wave detector
 * Handles configuration management UI and interactions
 */

const ConfigModule = (() => {
    // DOM elements
    const configPanel = document.getElementById('config-panel');
    const eventSelect = document.getElementById('event-name');
    const downsampleFactorInput = document.getElementById('downsample-factor');
    const scaleFactorInput = document.getElementById('scale-factor');
    const useGpuCheckbox = document.getElementById('use-gpu');
    const useZxOptCheckbox = document.getElementById('use-zx-opt');
    const zxOptLevelSelect = document.getElementById('zx-opt-level');
    const zxOptLevelGroup = document.getElementById('zx-opt-level-group');
    const saveConfigButton = document.getElementById('save-config');
    const resetConfigButton = document.getElementById('reset-config');
    const configStatus = document.getElementById('config-status');
    
    // Dashboard elements
    const dashboardConfigSummary = document.getElementById('dashboard-config-summary');
    const dashboardEditConfigButton = document.getElementById('dashboard-edit-config');
    
    // Default configuration
    const defaultConfig = {
        event_name: 'GW150914',
        downsample_factor: 200,
        scale_factor: 1e21,
        use_gpu: true,
        use_zx_opt: false,
        zx_opt_level: 1
    };
    
    /**
     * Populate the events dropdown
     */
    const populateEvents = async () => {
        const response = await API.getEvents();
        if (response.success) {
            const events = response.data;
            eventSelect.innerHTML = '';
            
            events.forEach(event => {
                const option = document.createElement('option');
                option.value = event;
                option.textContent = event;
                eventSelect.appendChild(option);
            });
        } else {
            showStatus('Error loading events: ' + response.error, 'error');
        }
    };
    
    /**
     * Load and display the current configuration
     */
    const loadConfig = async () => {
        UI.showLoading('Loading configuration...');
        
        const response = await API.getConfig();
        
        UI.hideLoading();
        
        if (response.success) {
            displayConfig(response.data);
            updateDashboardConfig(response.data);
        } else {
            showStatus('Error loading configuration: ' + response.error, 'error');
        }
    };
    
    /**
     * Display the configuration in the form
     */
    const displayConfig = (config) => {
        // Populate form fields
        eventSelect.value = config.event_name;
        downsampleFactorInput.value = config.downsample_factor;
        scaleFactorInput.value = config.scale_factor;
        useGpuCheckbox.checked = config.use_gpu;
        useZxOptCheckbox.checked = config.use_zx_opt;
        zxOptLevelSelect.value = config.zx_opt_level;
        
        // Toggle ZX optimization level visibility
        toggleZxOptLevel();
    };
    
    /**
     * Update the dashboard configuration summary
     */
    const updateDashboardConfig = (config) => {
        if (!dashboardConfigSummary) return;
        
        dashboardConfigSummary.innerHTML = `
            <table>
                <tr><th>Event:</th><td>${config.event_name}</td></tr>
                <tr><th>Downsample Factor:</th><td>${config.downsample_factor}</td></tr>
                <tr><th>Scale Factor:</th><td>${config.scale_factor.toExponential()}</td></tr>
                <tr><th>GPU Acceleration:</th><td>${config.use_gpu ? 'Enabled' : 'Disabled'}</td></tr>
                <tr><th>ZX Optimization:</th><td>${config.use_zx_opt ? 'Enabled (Level ' + config.zx_opt_level + ')' : 'Disabled'}</td></tr>
            </table>
        `;
    };
    
    /**
     * Save the configuration
     */
    const saveConfig = async () => {
        // Get values from form
        const config = {
            event_name: eventSelect.value,
            downsample_factor: parseInt(downsampleFactorInput.value),
            scale_factor: parseFloat(scaleFactorInput.value),
            use_gpu: useGpuCheckbox.checked,
            use_zx_opt: useZxOptCheckbox.checked,
            zx_opt_level: parseInt(zxOptLevelSelect.value)
        };
        
        // Validate
        if (!validateConfig(config)) {
            return;
        }
        
        UI.showLoading('Saving configuration...');
        
        const response = await API.updateConfig(config);
        
        UI.hideLoading();
        
        if (response.success) {
            showStatus('Configuration saved successfully!', 'success');
            updateDashboardConfig(response.data);
        } else {
            showStatus('Error saving configuration: ' + response.error, 'error');
        }
    };
    
    /**
     * Reset configuration to defaults
     */
    const resetConfig = async () => {
        if (confirm('Reset configuration to default values?')) {
            UI.showLoading('Resetting configuration...');
            
            const response = await API.updateConfig(defaultConfig);
            
            UI.hideLoading();
            
            if (response.success) {
                displayConfig(response.data);
                updateDashboardConfig(response.data);
                showStatus('Configuration reset to defaults!', 'success');
            } else {
                showStatus('Error resetting configuration: ' + response.error, 'error');
            }
        }
    };
    
    /**
     * Toggle ZX optimization level visibility
     */
    const toggleZxOptLevel = () => {
        if (useZxOptCheckbox.checked) {
            zxOptLevelGroup.style.display = 'block';
        } else {
            zxOptLevelGroup.style.display = 'none';
        }
    };
    
    /**
     * Show a status message
     */
    const showStatus = (message, type = 'success') => {
        configStatus.textContent = message;
        configStatus.className = `status-message ${type}`;
        
        // Auto-hide after delay
        setTimeout(() => {
            configStatus.className = 'status-message';
        }, 5000);
    };
    
    /**
     * Validate configuration
     */
    const validateConfig = (config) => {
        // Check downsample factor (must be positive integer)
        if (isNaN(config.downsample_factor) || config.downsample_factor <= 0) {
            showStatus('Downsample factor must be a positive number', 'error');
            return false;
        }
        
        // Check scale factor (must be positive number)
        if (isNaN(config.scale_factor) || config.scale_factor <= 0) {
            showStatus('Scale factor must be a positive number', 'error');
            return false;
        }
        
        return true;
    };
    
    /**
     * Initialize configuration module
     */
    const init = () => {
        // Set up event listeners
        if (saveConfigButton) {
            saveConfigButton.addEventListener('click', saveConfig);
        }
        
        if (resetConfigButton) {
            resetConfigButton.addEventListener('click', resetConfig);
        }
        
        if (useZxOptCheckbox) {
            useZxOptCheckbox.addEventListener('change', toggleZxOptLevel);
            toggleZxOptLevel(); // Initial visibility
        }
        
        if (dashboardEditConfigButton) {
            dashboardEditConfigButton.addEventListener('click', () => {
                UI.showPanel('config-panel');
            });
        }
        
        // Load events and configuration
        populateEvents();
        loadConfig();
    };
    
    // Return public interface
    return {
        init,
        loadConfig,
        saveConfig,
        resetConfig
    };
})();

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
    // ConfigModule will be initialized by the main app.js
});

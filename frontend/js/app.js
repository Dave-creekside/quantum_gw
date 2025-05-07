/**
 * Main application module for quantum gravitational wave detector
 * Handles UI navigation, panel management, and initializes other modules
 */

const UI = (() => {
    // DOM elements
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingMessage = document.getElementById('loading-message');
    const panels = document.querySelectorAll('.panel');
    const navLinks = document.querySelectorAll('.main-nav a');
    const tabButtons = document.querySelectorAll('.tab-btn');
    
    /**
     * Show a specific panel by ID
     */
    const showPanel = (panelId) => {
        // Hide all panels
        panels.forEach(panel => {
            panel.classList.remove('active');
        });
        
        // Show the selected panel
        const selectedPanel = document.getElementById(panelId);
        if (selectedPanel) {
            selectedPanel.classList.add('active');
            
            // Update navigation highlight
            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.id === `nav-${panelId.replace('-panel', '')}`) {
                    link.classList.add('active');
                }
            });
            
            // Scroll to top
            window.scrollTo(0, 0);
        }
    };
    
    /**
     * Switch tabs within a tab set
     */
    const switchTab = (tabBtn) => {
        // Get the tab set and tab content parent
        const tabSet = tabBtn.closest('.tabs');
        const tabContentParent = tabSet.parentElement;
        
        // Hide all tab contents in this set
        const tabContents = tabContentParent.querySelectorAll('.tab-content');
        tabContents.forEach(content => {
            content.classList.remove('active');
        });
        
        // Remove active class from all buttons in this set
        const tabBtns = tabSet.querySelectorAll('.tab-btn');
        tabBtns.forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Activate the selected tab and its content
        tabBtn.classList.add('active');
        const tabId = tabBtn.getAttribute('data-tab');
        const selectedContent = tabContentParent.querySelector(`#${tabId}-tab`);
        if (selectedContent) {
            selectedContent.classList.add('active');
        }
    };
    
    /**
     * Show loading overlay with message
     */
    const showLoading = (message = 'Processing...') => {
        if (loadingMessage) {
            loadingMessage.textContent = message;
        }
        if (loadingOverlay) {
            loadingOverlay.classList.remove('hidden');
        }
    };
    
    /**
     * Hide loading overlay
     */
    const hideLoading = () => {
        if (loadingOverlay) {
            loadingOverlay.classList.add('hidden');
        }
    };
    
    /**
     * Format a value for display
     */
    const formatValue = (value, type = 'default') => {
        if (value === null || value === undefined) {
            return 'N/A';
        }
        
        switch (type) {
            case 'scientific':
                return typeof value === 'number' ? value.toExponential(4) : value;
            case 'decimal':
                return typeof value === 'number' ? value.toFixed(4) : value;
            case 'percentage':
                return typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : value;
            case 'boolean':
                return value ? 'Yes' : 'No';
            default:
                return value;
        }
    };
    
    /**
     * Initialize UI module
     */
    const init = () => {
        // Set up navigation event listeners
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const panelId = link.id.replace('nav-', '') + '-panel';
                showPanel(panelId);
            });
        });
        
        // Set up tab event listeners
        tabButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                switchTab(btn);
            });
        });
        
        // Add quick dashboard navigation
        const quickRunPreset = document.getElementById('quick-run-preset');
        if (quickRunPreset) {
            quickRunPreset.addEventListener('click', () => {
                showPanel('pipelines-panel');
            });
        }
        
        const dashboardViewResults = document.getElementById('dashboard-view-results');
        if (dashboardViewResults) {
            dashboardViewResults.addEventListener('click', () => {
                showPanel('results-panel');
            });
        }
    };
    
    // Return public interface
    return {
        init,
        showPanel,
        switchTab,
        showLoading,
        hideLoading,
        formatValue
    };
})();

// Main App Initialization
document.addEventListener('DOMContentLoaded', () => {
    // Initialize UI
    UI.init();
    
    // Initialize modules
    if (typeof ConfigModule !== 'undefined') {
        ConfigModule.init();
    }
    
    if (typeof PipelinesModule !== 'undefined') {
        PipelinesModule.init();
    }
    
    if (typeof ResultsModule !== 'undefined') {
        ResultsModule.init();
    }
    
    if (typeof SweepsModule !== 'undefined') {
        SweepsModule.init();
    }
    
    if (typeof VisualizationsModule !== 'undefined') {
        VisualizationsModule.init();
    }
    
    // Load dashboard data if we're starting on the dashboard
    const dashboardRecentResults = document.getElementById('dashboard-recent-results');
    if (dashboardRecentResults && typeof ResultsModule !== 'undefined') {
        ResultsModule.loadRecentResultsForDashboard();
    }
    
    console.log('Quantum Gravitational Wave Detector UI initialized');
});

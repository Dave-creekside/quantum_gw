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

            // Fetch system stats if dashboard is shown
            if (panelId === 'dashboard-panel') {
                fetchAndDisplaySystemStats();
            }
        }
    };

    /**
     * Fetch and display system stats on the dashboard.
     */
    const fetchAndDisplaySystemStats = async () => {
        const container = document.getElementById('dashboard-system-stats');
        if (!container) return;

        // Find elements within the container
        const cpuBar = container.querySelector('.cpu-bar');
        const cpuValue = container.querySelector('.cpu-bar + .stat-value');
        const ramBar = container.querySelector('.ram-bar');
        const ramValue = container.querySelector('.ram-bar + .stat-value');
        const gpuBar = container.querySelector('.gpu-bar');
        const gpuValue = container.querySelector('.gpu-bar + .stat-value');
        const vramBar = container.querySelector('.vram-bar');
        const vramValue = container.querySelector('.vram-bar + .stat-value');
        const gpuDetails = document.getElementById('gpu-details');

        // Helper to update a bar and its value
        const updateStat = (barEl, valueEl, percent, textValue = null) => {
            if (barEl && valueEl) {
                // Handle the 'Error' string case explicitly
                const isError = percent === 'Error' || (textValue && textValue.includes('Error'));
                const isNA = percent === 'N/A' || (textValue && textValue === 'N/A');
                
                // Determine what percentage to display (0 for error/N/A, the actual percent for valid numbers)
                const displayPercent = (typeof percent === 'number' && !isNaN(percent)) ? percent : 0;
                
                // Determine what text to display
                let displayText;
                if (textValue !== null) {
                    displayText = textValue; // Use provided text value if available
                } else if (isError) {
                    displayText = 'Error'; // Show Error text
                } else if (isNA) {
                    displayText = 'N/A'; // Show N/A text
                } else if (typeof percent === 'number' && !isNaN(percent)) {
                    displayText = `${percent.toFixed(1)}%`; // Show formatted percentage
                } else {
                    displayText = 'N/A'; // Fallback for any other unexpected value
                }

                // Update UI elements
                barEl.style.width = `${displayPercent}%`;
                valueEl.textContent = displayText;
                
                // Add appropriate classes for styling
                const parentEl = barEl.parentElement;
                parentEl.classList.remove('error', 'na');
                if (isError) {
                    parentEl.classList.add('error');
                } else if (isNA) {
                    parentEl.classList.add('na');
                }
            }
        };

        // Show loading state
        updateStat(cpuBar, cpuValue, 0, 'Loading...');
        updateStat(ramBar, ramValue, 0, 'Loading...');
        updateStat(gpuBar, gpuValue, 0, 'Loading...');
        updateStat(vramBar, vramValue, 0, 'Loading...');
        if(gpuDetails) gpuDetails.textContent = 'Loading GPU details...';

        const response = await API.getSystemStats();

        if (response.success) {
            const stats = response.data;
            console.log("System Stats:", stats);

            // Safe number conversion helper
            const safeNumber = (val) => {
                return (typeof val === 'number' && !isNaN(val)) ? val :
                       (typeof val === 'string' && !isNaN(parseFloat(val))) ? parseFloat(val) : null;
            };
            
            // Format a value safely
            const formatNumber = (val, decimals = 1) => {
                const num = safeNumber(val);
                return num !== null ? num.toFixed(decimals) : 'N/A';
            };
            
            // CPU stats
            updateStat(cpuBar, cpuValue, safeNumber(stats.cpu_percent));
            
            // RAM stats - Create human-readable display
            const ramPercentVal = safeNumber(stats.ram_percent);
            let ramDisplayText;
            
            if (stats.ram_used_gb === 'N/A' || stats.ram_total_gb === 'N/A') {
                ramDisplayText = 'N/A';
            } else if (stats.ram_used_gb === 'Error' || stats.ram_total_gb === 'Error') {
                ramDisplayText = 'Error';
            } else {
                const usedGB = formatNumber(stats.ram_used_gb);
                const totalGB = formatNumber(stats.ram_total_gb);
                const percentText = ramPercentVal !== null ? `(${formatNumber(ramPercentVal)}%)` : '';
                ramDisplayText = `${usedGB}/${totalGB} GB ${percentText}`;
            }
            updateStat(ramBar, ramValue, ramPercentVal, ramDisplayText);
            
            // GPU Utilization
            updateStat(gpuBar, gpuValue, safeNumber(stats.gpu_utilization_percent));
            
            // VRAM stats - Create human-readable display
            const vramPercentVal = safeNumber(stats.vram_percent);
            let vramDisplayText;
            
            if (stats.vram_used_mb === 'N/A' || stats.vram_total_mb === 'N/A') {
                vramDisplayText = 'N/A';
            } else if (stats.vram_used_mb === 'Error' || stats.vram_total_mb === 'Error') {
                vramDisplayText = 'Error';
            } else {
                const usedMB = formatNumber(stats.vram_used_mb, 0);
                const totalMB = formatNumber(stats.vram_total_mb, 0);
                const percentText = vramPercentVal !== null ? `(${formatNumber(vramPercentVal)}%)` : '';
                vramDisplayText = `${usedMB}/${totalMB} MB ${percentText}`;
            }
            updateStat(vramBar, vramValue, vramPercentVal, vramDisplayText);

            // GPU Details
            if(gpuDetails) {
                let detailsText = '';
                if (stats.gpu_name && stats.gpu_name !== 'N/A' && !stats.gpu_name.includes('Error')) {
                    detailsText += `${stats.gpu_name}`;
                    
                    // Add temperature if available
                    const tempVal = safeNumber(stats.gpu_temperature_c);
                    if (tempVal !== null) {
                        detailsText += ` (${tempVal.toFixed(0)}°C)`;
                    }
                } else if (stats.gpu_name && stats.gpu_name.includes('Error')) {
                    detailsText = `Error detecting GPU: ${stats.gpu_name.replace('Error: ', '')}`;
                } else {
                    detailsText = 'GPU not available';
                }
                gpuDetails.textContent = detailsText;
            }

        } else {
            console.error("Error fetching system stats:", response.error);
            updateStat(cpuBar, cpuValue, 'Error', 'Error');
            updateStat(ramBar, ramValue, 'Error', 'Error');
            updateStat(gpuBar, gpuValue, 'Error', 'Error');
            updateStat(vramBar, vramValue, 'Error', 'Error');
            if(gpuDetails) gpuDetails.textContent = `Error fetching stats: ${response.error || 'Unknown error'}`;
        }
    };

    /**
     * Apply the saved or preferred theme on load.
     */
    const applyInitialTheme = () => {
        const savedTheme = localStorage.getItem('theme');
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const themeSwitch = document.getElementById('theme-switch');

        if (savedTheme) {
            document.body.classList.toggle('dark-mode', savedTheme === 'dark');
            if (themeSwitch) themeSwitch.checked = (savedTheme === 'dark');
        } else if (prefersDark) {
            document.body.classList.add('dark-mode');
             if (themeSwitch) themeSwitch.checked = true;
        }
        console.log(`Initial theme applied: ${document.body.classList.contains('dark-mode') ? 'dark' : 'light'}`);
    };

    /**
     * Handle theme switching.
     */
    const handleThemeSwitch = (event) => {
        const isDarkMode = event.target.checked;
        document.body.classList.toggle('dark-mode', isDarkMode);
        localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
        console.log(`Theme switched to: ${isDarkMode ? 'dark' : 'light'}`);
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
     * Show header spinner (message parameter is ignored now).
     */
    const showLoading = (message = 'Processing...') => {
        const spinner = document.getElementById('header-spinner');
        if (spinner) {
            spinner.classList.remove('hidden');
        }
    };
    
    /**
     * Hide header spinner.
     */
    const hideLoading = () => {
         const spinner = document.getElementById('header-spinner');
        if (spinner) {
            spinner.classList.add('hidden');
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

        // Add theme switch listener
        const themeSwitch = document.getElementById('theme-switch');
        if (themeSwitch) {
            themeSwitch.addEventListener('change', handleThemeSwitch);
        }

        // Apply initial theme on load
        applyInitialTheme();

        // Add listener for stats refresh button
        const refreshStatsBtn = document.getElementById('refresh-stats-btn');
        if (refreshStatsBtn) {
            refreshStatsBtn.addEventListener('click', fetchAndDisplaySystemStats);
        }

        // Initial fetch for dashboard if it's the active panel on load
        if (document.getElementById('dashboard-panel')?.classList.contains('active')) {
             fetchAndDisplaySystemStats();
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
    
    if (typeof ProjectsModule !== 'undefined') {
        ProjectsModule.init();
    }
    
    // Load dashboard data if we're starting on the dashboard
    const dashboardRecentResults = document.getElementById('dashboard-recent-results');
    if (dashboardRecentResults && typeof ResultsModule !== 'undefined') {
        ResultsModule.loadRecentResultsForDashboard();
    }
    
    console.log('Quantum Gravitational Wave Detector UI initialized');
});

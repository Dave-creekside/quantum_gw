/**
 * Sweeps Module for quantum gravitational wave detector
 * Handles parameter sweeps UI and interactions
 */

const SweepsModule = (() => {
    // DOM elements - Qubit Count Sweep
    const qubitSweepTopologySelect = document.getElementById('qubit-sweep-topology');
    const qubitSweepCountsContainer = document.getElementById('qubit-sweep-counts');
    const runQubitSweepButton = document.getElementById('run-qubit-sweep');
    
    // DOM elements - Topology Sweep
    const topologySweepQubitsSelect = document.getElementById('topology-sweep-qubits');
    const topologySweepTypesContainer = document.getElementById('topology-sweep-types');
    const runTopologySweepButton = document.getElementById('run-topology-sweep');
    
    // DOM elements - Scale Factor Sweep
    const scaleSweepQubitsSelect = document.getElementById('scale-sweep-qubits');
    const scaleSweepTopologySelect = document.getElementById('scale-sweep-topology');
    const scaleFactorsInput = document.getElementById('scale-factors');
    const runScaleSweepButton = document.getElementById('run-scale-sweep');
    
    // DOM elements - Results
    const sweepResultContainer = document.getElementById('sweep-result');
    
    /**
     * Run a qubit count sweep
     */
    const runQubitSweep = async () => {
        if (!qubitSweepTopologySelect || !qubitSweepCountsContainer) return;
        
        // Get selected topology
        const topology = qubitSweepTopologySelect.value;
        
        // Get selected qubit counts
        const checkboxes = qubitSweepCountsContainer.querySelectorAll('input[type="checkbox"]:checked');
        const qubitCounts = Array.from(checkboxes).map(checkbox => parseInt(checkbox.value));
        
        if (qubitCounts.length === 0) {
            alert('Please select at least one qubit count.');
            return;
        }
        
        UI.showLoading('Running qubit count sweep...');
        
        const response = await API.sweepQubitCount(topology, qubitCounts);
        
        UI.hideLoading();
        
        if (response.success) {
            displaySweepResults(response.data, 'Qubit Count Sweep');
        } else {
            alert(`Error running sweep: ${response.error}`);
        }
    };
    
    /**
     * Run a topology sweep
     */
    const runTopologySweep = async () => {
        if (!topologySweepQubitsSelect || !topologySweepTypesContainer) return;
        
        // Get selected qubit count
        const qubitCount = parseInt(topologySweepQubitsSelect.value);
        
        // Get selected topologies
        const checkboxes = topologySweepTypesContainer.querySelectorAll('input[type="checkbox"]:checked');
        const topologies = Array.from(checkboxes).map(checkbox => checkbox.value);
        
        if (topologies.length === 0) {
            alert('Please select at least one topology.');
            return;
        }
        
        UI.showLoading('Running topology sweep...');
        
        const response = await API.sweepTopology(qubitCount, topologies);
        
        UI.hideLoading();
        
        if (response.success) {
            displaySweepResults(response.data, 'Topology Sweep');
        } else {
            alert(`Error running sweep: ${response.error}`);
        }
    };
    
    /**
     * Run a scale factor sweep
     */
    const runScaleSweep = async () => {
        if (!scaleSweepQubitsSelect || !scaleSweepTopologySelect || !scaleFactorsInput) return;
        
        // Get pipeline config
        const qubits = parseInt(scaleSweepQubitsSelect.value);
        const topology = scaleSweepTopologySelect.value;
        const pipelineConfig = [[qubits, topology]];
        
        // Parse scale factors
        const scaleFactorsText = scaleFactorsInput.value.trim();
        if (!scaleFactorsText) {
            alert('Please enter at least one scale factor.');
            return;
        }
        
        let scaleFactors;
        try {
            // Handle comma-separated values like "1e19, 1e20, 1e21"
            scaleFactors = scaleFactorsText.split(',')
                .map(s => s.trim())
                .filter(s => s)
                .map(s => {
                    // Handle scientific notation (e.g., 1e21)
                    if (s.toLowerCase().includes('e')) {
                        const [base, exp] = s.toLowerCase().split('e');
                        return parseFloat(base) * Math.pow(10, parseInt(exp));
                    }
                    return parseFloat(s);
                });
        } catch (e) {
            alert('Invalid scale factors. Please enter numbers in scientific notation (e.g., 1e21) separated by commas.');
            return;
        }
        
        if (scaleFactors.some(isNaN)) {
            alert('Invalid scale factors. Please enter numbers in scientific notation (e.g., 1e21) separated by commas.');
            return;
        }
        
        UI.showLoading('Running scale factor sweep...');
        
        const response = await API.sweepScaleFactor(pipelineConfig, scaleFactors);
        
        UI.hideLoading();
        
        if (response.success) {
            displaySweepResults(response.data, 'Scale Factor Sweep');
        } else {
            alert(`Error running sweep: ${response.error}`);
        }
    };
    
    /**
     * Display sweep results
     */
    const displaySweepResults = (results, sweepName) => {
        if (!sweepResultContainer) return;
        
        // Format results into a table
        let sweepDataTable = '';
        let sweepChartData = [];
        
        if (Array.isArray(results)) {
            sweepDataTable = `
                <table>
                    <thead>
                        <tr>
                            <th>Configuration</th>
                            <th>SNR</th>
                            <th>Max QFI</th>
                            <th>Execution Time</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            results.forEach(result => {
                if (result.error) {
                    sweepDataTable += `
                        <tr>
                            <td>${result.name || 'Unknown'}</td>
                            <td colspan="3" class="error">Error: ${result.error}</td>
                        </tr>
                    `;
                    return;
                }
                
                // Extract name and SNR for chart
                if (result.summary && result.name) {
                    sweepChartData.push({
                        name: result.name,
                        snr: result.summary.final_stage_snr
                    });
                }
                
                sweepDataTable += `
                    <tr>
                        <td>${result.name || 'Unknown'}</td>
                        <td>${result.summary ? UI.formatValue(result.summary.final_stage_snr, 'decimal') : 'N/A'}</td>
                        <td>${result.stages && result.stages[0] ? UI.formatValue(result.stages[0].max_qfi, 'decimal') : 'N/A'}</td>
                        <td>${result.summary ? UI.formatValue(result.summary.total_execution_time, 'decimal') + ' s' : 'N/A'}</td>
                    </tr>
                `;
            });
            
            sweepDataTable += `
                    </tbody>
                </table>
            `;
        } else {
            sweepDataTable = '<p>No sweep results available.</p>';
        }
        
        // Create HTML with sweep results
        const html = `
            <h3>${sweepName} Results</h3>
            
            <div class="sweep-data">
                ${sweepDataTable}
            </div>
            
            ${results.file_paths && results.file_paths.visualization ? `
                <div class="sweep-visualization">
                    <h4>Visualization</h4>
                    <img src="${results.file_paths.visualization}" alt="Sweep Visualization">
                </div>
            ` : ''}
            
            <div id="sweep-chart" class="sweep-chart"></div>
        `;
        
        sweepResultContainer.innerHTML = html;
        
        // Generate chart if we have data
        if (sweepChartData.length > 0) {
            createSweepChart(sweepChartData, 'sweep-chart');
        }
        
        // Scroll to results
        sweepResultContainer.scrollIntoView({ behavior: 'smooth' });
    };
    
    /**
     * Create a simple chart for sweep results
     */
    const createSweepChart = (data, containerId) => {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        // Sort data by SNR descending
        data.sort((a, b) => b.snr - a.snr);
        
        // Get max SNR for scaling
        const maxSNR = Math.max(...data.map(d => d.snr));
        
        // Create bars
        let barsHtml = '';
        data.forEach(item => {
            const percentage = (item.snr / maxSNR) * 100;
            barsHtml += `
                <div class="chart-bar-container">
                    <div class="chart-label">${item.name}</div>
                    <div class="chart-bar" style="width: ${percentage}%;">
                        <span class="chart-value">${UI.formatValue(item.snr, 'decimal')}</span>
                    </div>
                </div>
            `;
        });
        
        const chartHtml = `
            <div class="chart-title">Signal-to-Noise Ratio Comparison</div>
            <div class="chart-content">
                ${barsHtml}
            </div>
        `;
        
        container.innerHTML = chartHtml;
        
        // Add CSS for chart
        if (!document.getElementById('chart-styles')) {
            const styleEl = document.createElement('style');
            styleEl.id = 'chart-styles';
            styleEl.textContent = `
                .sweep-chart {
                    margin-top: 2rem;
                    padding: 1rem;
                    background: #f8f9fa;
                    border-radius: 5px;
                    border: 1px solid var(--border-color);
                }
                .chart-title {
                    font-weight: 500;
                    margin-bottom: 1rem;
                    text-align: center;
                }
                .chart-content {
                    display: flex;
                    flex-direction: column;
                    gap: 0.75rem;
                }
                .chart-bar-container {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }
                .chart-label {
                    width: 150px;
                    text-align: right;
                    font-size: 0.9rem;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }
                .chart-bar {
                    height: 25px;
                    background: var(--primary-color);
                    border-radius: 3px;
                    position: relative;
                    min-width: 40px;
                    display: flex;
                    align-items: center;
                    padding: 0 0.5rem;
                    color: white;
                    transition: width 0.5s ease;
                }
                .chart-value {
                    font-size: 0.8rem;
                    font-weight: 500;
                }
            `;
            document.head.appendChild(styleEl);
        }
    };
    
    /**
     * Initialize sweeps module
     */
    const init = () => {
        // Set up event listeners
        if (runQubitSweepButton) {
            runQubitSweepButton.addEventListener('click', runQubitSweep);
        }
        
        if (runTopologySweepButton) {
            runTopologySweepButton.addEventListener('click', runTopologySweep);
        }
        
        if (runScaleSweepButton) {
            runScaleSweepButton.addEventListener('click', runScaleSweep);
        }
    };
    
    // Return public interface
    return {
        init,
        runQubitSweep,
        runTopologySweep,
        runScaleSweep
    };
})();

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
    // SweepsModule will be initialized by the main app.js
});

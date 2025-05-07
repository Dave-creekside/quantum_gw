/**
 * Results Module for quantum gravitational wave detector
 * Handles results viewing and management
 */

const ResultsModule = (() => {
    // DOM elements
    const resultsList = document.getElementById('results-list');
    const resultDetail = document.getElementById('result-detail');
    const dashboardRecentResults = document.getElementById('dashboard-recent-results');
    
    // State
    let savedResults = [];
    
    /**
     * Load all saved results
     */
    const loadResults = async () => {
        UI.showLoading('Loading results...');
        
        const response = await API.getSavedResults();
        
        UI.hideLoading();
        
        if (response.success) {
            savedResults = response.data;
            displayResultsList();
        } else {
            console.error('Error loading results:', response.error);
            if (resultsList) {
                resultsList.innerHTML = `<p class="error">Error loading results: ${response.error}</p>`;
            }
        }
    };
    
    /**
     * Display the list of saved results
     */
    const displayResultsList = () => {
        if (!resultsList) return;
        
        if (savedResults.length === 0) {
            resultsList.innerHTML = '<p>No saved results found. Run a pipeline to generate results.</p>';
            return;
        }
        
        // Group results by date
        const resultsByDate = savedResults.reduce((groups, result) => {
            // Extract date from timestamp
            const timestamp = result.timestamp;
            const dateOnly = timestamp.split('_')[0]; // Extract YYYYMMDD part
            
            // Format the date nicely
            const year = dateOnly.substring(0, 4);
            const month = dateOnly.substring(4, 6);
            const day = dateOnly.substring(6, 8);
            const formattedDate = `${year}-${month}-${day}`;
            
            if (!groups[formattedDate]) {
                groups[formattedDate] = [];
            }
            groups[formattedDate].push(result);
            
            return groups;
        }, {});
        
        // Sort dates in descending order
        const sortedDates = Object.keys(resultsByDate).sort().reverse();
        
        // Create HTML
        let html = '';
        
        sortedDates.forEach(date => {
            const dateResults = resultsByDate[date];
            
            html += `<div class="result-date-group"><h4>${date}</h4>`;
            
            dateResults.forEach(result => {
                // Extract time from timestamp
                const timestamp = result.timestamp;
                const timeOnly = timestamp.split('_')[1]; // Extract HHMMSS part
                const hours = timeOnly ? timeOnly.substring(0, 2) : '00';
                const minutes = timeOnly ? timeOnly.substring(2, 4) : '00';
                const formattedTime = `${hours}:${minutes}`;
                
                html += `
                    <div class="result-item" data-id="${result.identifier}">
                        <div class="result-time">${formattedTime}</div>
                        <div class="result-name">${result.pipeline_name || 'Pipeline'}</div>
                    </div>
                `;
            });
            
            html += '</div>';
        });
        
        resultsList.innerHTML = html;
        
        // Add click event listeners
        const resultItems = resultsList.querySelectorAll('.result-item');
        resultItems.forEach(item => {
            item.addEventListener('click', () => {
                const resultId = item.dataset.id;
                loadResultDetail(resultId);
                
                // Update active state
                resultItems.forEach(i => i.classList.remove('active'));
                item.classList.add('active');
            });
        });
        
        // Load the first result by default
        if (resultItems.length > 0) {
            resultItems[0].click();
        }
    };
    
    /**
     * Load recent results for dashboard
     */
    const loadRecentResultsForDashboard = async () => {
        if (!dashboardRecentResults) return;
        
        const response = await API.getSavedResults();
        
        if (response.success) {
            const results = response.data;
            
            if (results.length === 0) {
                dashboardRecentResults.innerHTML = '<p>No saved results found. Run a pipeline to generate results.</p>';
                return;
            }
            
            // Get up to 5 most recent results
            const recentResults = results.slice(0, 5);
            
            let html = '<table><thead><tr><th>Date</th><th>Pipeline</th><th>SNR</th></tr></thead><tbody>';
            
            for (const result of recentResults) {
                // Try to fetch the detailed result to get SNR
                const detailResponse = await API.getResultById(result.identifier);
                let snr = 'N/A';
                
                if (detailResponse.success && detailResponse.data.summary) {
                    snr = UI.formatValue(detailResponse.data.summary.final_stage_snr, 'decimal');
                }
                
                // Format timestamp
                const timestamp = result.timestamp;
                const dateOnly = timestamp.split('_')[0]; // YYYYMMDD
                const timeOnly = timestamp.split('_')[1]; // HHMMSS
                
                const year = dateOnly.substring(0, 4);
                const month = dateOnly.substring(4, 6);
                const day = dateOnly.substring(6, 8);
                
                const hours = timeOnly ? timeOnly.substring(0, 2) : '00';
                const minutes = timeOnly ? timeOnly.substring(2, 4) : '00';
                
                const formattedDate = `${year}-${month}-${day} ${hours}:${minutes}`;
                
                html += `
                    <tr>
                        <td>${formattedDate}</td>
                        <td>${result.pipeline_name || 'Pipeline'}</td>
                        <td>${snr}</td>
                    </tr>
                `;
            }
            
            html += '</tbody></table>';
            dashboardRecentResults.innerHTML = html;
        } else {
            dashboardRecentResults.innerHTML = `<p class="error">Error loading results: ${response.error}</p>`;
        }
    };
    
    /**
     * Load and display a specific result
     */
    const loadResultDetail = async (resultId) => {
        if (!resultDetail) return;
        
        UI.showLoading('Loading result details...');
        
        const response = await API.getResultById(resultId);
        
        UI.hideLoading();
        
        if (response.success) {
            displayResultDetail(response.data);
        } else {
            resultDetail.innerHTML = `<p class="error">Error loading result: ${response.error}</p>`;
        }
    };
    
    /**
     * Display a result's details
     */
    const displayResultDetail = (result) => {
        if (!resultDetail) return;
        
        // Format timestamp
        const timestamp = result.timestamp;
        const dateOnly = timestamp.split('_')[0]; // YYYYMMDD
        const timeOnly = timestamp.split('_')[1]; // HHMMSS
        
        const year = dateOnly.substring(0, 4);
        const month = dateOnly.substring(4, 6);
        const day = dateOnly.substring(6, 8);
        
        const hours = timeOnly ? timeOnly.substring(0, 2) : '00';
        const minutes = timeOnly ? timeOnly.substring(2, 4) : '00';
        const seconds = timeOnly ? timeOnly.substring(4, 6) : '00';
        
        const formattedDate = `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
        
        // Format stage results
        const stagesHtml = result.stages.map((stage, index) => {
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
        
        // Create HTML for result
        const html = `
            <h3>Pipeline Results</h3>
            <p class="timestamp">Executed on: ${formattedDate}</p>
            
            <div class="result-summary">
                <h4>Summary</h4>
                <table>
                    <tr>
                        <th>Event</th>
                        <td>${result.event_name}</td>
                    </tr>
                    <tr>
                        <th>Pipeline Configuration</th>
                        <td>${result.pipeline_config}</td>
                    </tr>
                    <tr>
                        <th>First Stage SNR</th>
                        <td>${UI.formatValue(result.summary.first_stage_snr, 'decimal')}</td>
                    </tr>
                    <tr>
                        <th>Final Stage SNR</th>
                        <td>${UI.formatValue(result.summary.final_stage_snr, 'decimal')}</td>
                    </tr>
                    <tr>
                        <th>Improvement Factor</th>
                        <td>${UI.formatValue(result.summary.improvement_factor, 'decimal')}×</td>
                    </tr>
                    <tr>
                        <th>Total Execution Time</th>
                        <td>${UI.formatValue(result.summary.total_execution_time, 'decimal')} s</td>
                    </tr>
                </table>
            </div>
            
            <div class="result-stages">
                <h4>Stage Results</h4>
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
            
            ${result.file_paths && result.file_paths.visualization ? `
                <div class="result-visualization">
                    <h4>Visualization</h4>
                    <img src="${result.file_paths.visualization}" alt="Pipeline Visualization">
                </div>
            ` : ''}
        `;
        
        resultDetail.innerHTML = html;
    };
    
    /**
     * Initialize results module
     */
    const init = () => {
        // Load all results
        loadResults();
    };
    
    // Return public interface
    return {
        init,
        loadResults,
        loadResultDetail,
        loadRecentResultsForDashboard
    };
})();

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
    // ResultsModule will be initialized by the main app.js
});

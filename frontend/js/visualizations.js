/**
 * Visualizations Module for quantum gravitational wave detector
 * Handles visualization rendering and formatting
 */

const VisualizationsModule = (() => {
    /**
     * Format the file path for viewing
     * This can handle different file paths and make them accessible for web viewing
     */
    const formatVisualizationPath = (path) => {
        if (!path) return null;
        
        // Remove 'data/' prefix if present for relative paths
        if (path.startsWith('data/')) {
            return path.replace('data/', '/data/');
        }
        
        // Check if it's already a URL
        if (path.startsWith('http://') || path.startsWith('https://')) {
            return path;
        }
        
        // Otherwise, just return the path as is
        return path;
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
     * Initialize visualizations module
     */
    const init = () => {
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
    return {
        init,
        formatVisualizationPath,
        openFullscreenViewer,
        processVisualizationContainers
    };
})();

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
    // VisualizationsModule will be initialized by the main app.js
});

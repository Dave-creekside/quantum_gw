# qgw_detector/visualization/plots.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime

class GWDetectionVisualizer:
    """
    Visualization tools for quantum gravitational wave detection
    """
    def __init__(self, save_dir="data/visualization"):
        """
        Initialize the visualizer
        
        Args:
            save_dir: Directory to save plot files
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_strain_data(self, times, strain, event_name="GW event", processed=None, 
                        save_filename=None, display=False):
        """
        Plot gravitational wave strain data
        
        Args:
            times: Time array
            strain: Strain values
            event_name: Name of the event for the title
            processed: Optional tuple of (processed_times, processed_strain) for comparison
            save_filename: Optional filename to save the plot
            display: Whether to display the plot (default: False)
        """
        plt.figure(figsize=(12, 6))
        
        if processed:
            # Create two subplots
            plt.subplot(2, 1, 1)
            plt.plot(times, strain, 'b')
            plt.title(f"Raw Data: {event_name}")
            plt.ylabel("Strain")
            
            plt.subplot(2, 1, 2)
            proc_times, proc_strain = processed
            plt.plot(proc_times, proc_strain, 'r')
            plt.title("Processed Data")
            plt.xlabel("Time (s)")
            plt.ylabel("Normalized Strain")
        else:
            # Single plot
            plt.plot(times, strain * 1e21, 'b')
            plt.title(f"Gravitational Wave Data: {event_name}")
            plt.xlabel("Time (s)")
            plt.ylabel("Strain (×10²¹)")
            plt.grid(True)
        
        plt.tight_layout()
        
        # Always save if filename provided
        if save_filename:
            save_path = os.path.join(self.save_dir, save_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        # Only show if display=True
        if display:
            plt.show()
        else:
            plt.close()
    
    def plot_quantum_detection(self, times, strain, quantum_states, qfi=None, 
                              event_name="GW event", n_qubits=None, 
                              detection_metric=None, save_filename=None,
                              display=False):
        """
        Comprehensive visualization of quantum gravitational wave detection
        
        Args:
            times: Time points
            strain: GW strain values
            quantum_states: Quantum states at each time point
            qfi: Quantum Fisher Information (optional)
            event_name: Name of the event
            n_qubits: Number of qubits (derived from states if not provided)
            detection_metric: Optional detection metric values
            save_filename: Optional filename to save the plot
            display: Whether to display the plot (default: False)
        """
        # Calculate probabilities
        probabilities = np.abs(quantum_states)**2
        
        # Determine number of qubits if not provided
        if n_qubits is None:
            n_qubits = int(np.log2(quantum_states.shape[1]))
        
        # Create figure
        fig = plt.figure(figsize=(12, 16))
        
        # Determine number of subplots
        n_plots = 0
        if strain is not None:
            n_plots += 1
        if qfi is not None:
            n_plots += 1
        if probabilities is not None:
            n_plots += 1
        if detection_metric is not None:
            n_plots += 1
        
        # Create GridSpec for flexible subplot layout
        gs = GridSpec(n_plots, 1, height_ratios=[1] * n_plots)
        
        # Track current subplot index
        plot_idx = 0
        
        # Plot strain
        if strain is not None:
            ax1 = fig.add_subplot(gs[plot_idx])
            ax1.plot(times, strain * 1e21, 'b-', linewidth=2)
            ax1.set_ylabel('Strain (×10²¹)')
            ax1.set_title(f'Gravitational Wave Strain: {event_name}')
            ax1.grid(True)
            plot_idx += 1
        
        # Plot QFI if provided
        if qfi is not None:
            ax2 = fig.add_subplot(gs[plot_idx])
            
            # Ensure compatible array lengths
            if len(qfi) == len(times) - 1:
                qfi_times = times[:-1]
            else:
                qfi_times = times
            
            ax2.plot(qfi_times, qfi, 'r-', linewidth=2)
            ax2.set_ylabel('Quantum Fisher\nInformation')
            ax2.set_title('Quantum Detector Sensitivity (QFI)')
            
            # Add horizontal line at mean QFI
            mean_qfi = np.mean(qfi)
            ax2.axhline(y=mean_qfi, color='k', linestyle='--', alpha=0.5)
            
            # Add text with QFI statistics
            max_qfi = np.max(qfi)
            std_qfi = np.std(qfi)
            snr = max_qfi / std_qfi if std_qfi > 0 else 0
            
            stats_text = f"Max QFI: {max_qfi:.2f}\nMean QFI: {mean_qfi:.2f}\nQFI SNR: {snr:.2f}"
            ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            ax2.grid(True)
            plot_idx += 1
        
        # Plot probabilities for key states
        if probabilities is not None:
            ax3 = fig.add_subplot(gs[plot_idx])
            
            # For larger systems, only plot a subset of states
            if n_qubits <= 4:
                # For small systems, plot all states
                plot_states = range(min(8, 2**n_qubits))
            else:
                # For larger systems, plot only key states
                plot_states = [0, 1, 2, 3, 2**n_qubits-4, 2**n_qubits-3, 2**n_qubits-2, 2**n_qubits-1]
            
            # Set up colormap
            cmap = plt.cm.viridis
            colors = cmap(np.linspace(0, 1, len(plot_states)))
            
            for i, state_idx in enumerate(plot_states):
                if state_idx < 2**n_qubits:
                    state_label = format(state_idx, f'0{n_qubits}b')
                    ax3.plot(times, probabilities[:, state_idx], 
                           label=f"|{state_label}⟩", color=colors[i], linewidth=2)
            
            ax3.set_ylabel('Probability')
            ax3.set_title('Quantum State Probabilities')
            ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax3.grid(True)
            plot_idx += 1
        
        # Plot detection metric if provided
        if detection_metric is not None:
            ax4 = fig.add_subplot(gs[plot_idx])
            ax4.plot(times, detection_metric, 'g-', linewidth=2)
            ax4.set_ylabel('Detection Metric')
            ax4.set_title('Quantum Detection Metric (|11...1⟩ - |00...0⟩)')
            ax4.grid(True)
            
            # Add horizontal line at mean
            mean_metric = np.mean(detection_metric)
            ax4.axhline(y=mean_metric, color='k', linestyle='--', alpha=0.5)
            
            # Add text with metric statistics
            max_metric = np.max(detection_metric)
            min_metric = np.min(detection_metric)
            range_metric = max_metric - min_metric
            
            stats_text = f"Max: {max_metric:.4f}\nMin: {min_metric:.4f}\nRange: {range_metric:.4f}"
            ax4.text(0.02, 0.95, stats_text, transform=ax4.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add common x-axis label to bottom subplot
        plt.xlabel('Time (s)')
        
        plt.tight_layout()
        
        # Always save if filename provided
        if save_filename:
            save_path = os.path.join(self.save_dir, save_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        # Only show if display=True
        if display:
            plt.show()
        else:
            plt.close()
        
        # Return detection statistics if QFI was provided
        if qfi is not None:
            return {
                "max_qfi": max_qfi,
                "mean_qfi": mean_qfi,
                "std_qfi": std_qfi,
                "qfi_snr": snr
            }
    
    def plot_entanglement_comparison(self, results_dict, event_name="GW event", 
                                   save_filename=None, display=False):
        """
        Compare QFI for different entanglement topologies
        
        Args:
            results_dict: Dictionary with entanglement types as keys and 
                         results dictionaries as values
            event_name: Name of the event
            save_filename: Optional filename to save the plot
            display: Whether to display the plot (default: False)
        """
        plt.figure(figsize=(12, 10))
        
        # Plot strain at the top
        plt.subplot(2, 1, 1)
        
        # Use the strain from the first result (should be the same for all)
        first_key = list(results_dict.keys())[0]
        times = results_dict[first_key]["times"]
        strain = results_dict[first_key]["strain"]
        
        plt.plot(times, strain * 1e21, 'k-')
        plt.title(f"Gravitational Wave Strain: {event_name}")
        plt.ylabel("Strain (×10²¹)")
        plt.grid(True)
        
        # Plot QFI comparison
        plt.subplot(2, 1, 2)
        
        colors = {"linear": "blue", "star": "red", "full": "green"}
        
        for entanglement_type, results in results_dict.items():
            if "qfi" in results and "times" in results:
                qfi = results["qfi"]
                qfi_times = results["times"][:-1] if len(qfi) == len(results["times"]) - 1 else results["times"]
                
                # Calculate SNR if available
                snr = results.get("qfi_snr", np.max(qfi) / np.std(qfi) if np.std(qfi) > 0 else 0)
                
                color = colors.get(entanglement_type, 'gray')
                plt.plot(qfi_times, qfi, color=color, 
                       label=f"{entanglement_type.capitalize()} (SNR: {snr:.2f})")
        
        plt.title("QFI Comparison Across Entanglement Topologies")
        plt.xlabel("Time (s)")
        plt.ylabel("Quantum Fisher Information")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Always save if filename provided
        if save_filename:
            save_path = os.path.join(self.save_dir, save_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        # Only show if display=True
        if display:
            plt.show()
        else:
            plt.close()
    
    def plot_qubit_scaling(self, results_dict, metric="qfi_snr", 
                         event_name="GW event", save_filename=None,
                         display=False):
        """
        Plot scaling of detection performance with qubit count
        
        Args:
            results_dict: Dictionary with qubit counts as keys and 
                         results dictionaries as values
            metric: Metric to plot ('qfi_snr', 'max_qfi', 'time')
            event_name: Name of the event
            save_filename: Optional filename to save the plot
            display: Whether to display the plot (default: False)
        """
        plt.figure(figsize=(10, 6))
        
        # Convert qubit counts to integers and sort
        qubit_counts = sorted([int(k) for k in results_dict.keys()])
        
        # Define metrics to extract
        if metric == "qfi_snr":
            values = [results_dict[str(n)].get("qfi_snr", 0) for n in qubit_counts]
            title = "Scaling of QFI SNR with Qubit Count"
            ylabel = "QFI Signal-to-Noise Ratio"
        elif metric == "max_qfi":
            values = [results_dict[str(n)].get("max_qfi", 0) for n in qubit_counts]
            title = "Scaling of Peak QFI with Qubit Count"
            ylabel = "Peak QFI Value"
        elif metric == "time":
            values = [results_dict[str(n)].get("execution_time", 0) for n in qubit_counts]
            title = "Scaling of Execution Time with Qubit Count"
            ylabel = "Execution Time (s)"
        else:
            values = [0] * len(qubit_counts)
            title = f"Scaling of {metric} with Qubit Count"
            ylabel = metric
        
        # Plot the scaling relationship
        plt.plot(qubit_counts, values, 'bo-', linewidth=2, markersize=8)
        
        # Add a trend line if we have enough points
        if len(qubit_counts) >= 3:
            try:
                # Fit exponential for time, polynomial for others
                if metric == "time":
                    # Log-linear fit for execution time (exponential scaling)
                    non_zero_indices = [i for i, v in enumerate(values) if v > 0]
                    if len(non_zero_indices) >= 3:
                        log_values = np.log([values[i] for i in non_zero_indices])
                        x_fit = [qubit_counts[i] for i in non_zero_indices]
                        coeffs = np.polyfit(x_fit, log_values, 1)
                        x_range = np.linspace(min(qubit_counts), max(qubit_counts), 100)
                        y_fit = np.exp(coeffs[1]) * np.exp(coeffs[0] * x_range)
                        
                        plt.plot(x_range, y_fit, 'r--', linewidth=1.5, 
                               label=f"Exponential fit: {np.exp(coeffs[1]):.2e} × e^({coeffs[0]:.2f}n)")
                else:
                    # Polynomial fit for other metrics
                    coeffs = np.polyfit(qubit_counts, values, 2)
                    x_range = np.linspace(min(qubit_counts), max(qubit_counts), 100)
                    y_fit = coeffs[0] * x_range**2 + coeffs[1] * x_range + coeffs[2]
                    
                    plt.plot(x_range, y_fit, 'r--', linewidth=1.5, 
                           label=f"Quadratic fit: {coeffs[0]:.2e}n² + {coeffs[1]:.2e}n + {coeffs[2]:.2e}")
                
                plt.legend()
            except:
                # Curve fitting may fail with certain data patterns
                pass
        
        plt.title(title)
        plt.xlabel("Number of Qubits")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.xticks(qubit_counts)
        
        # Always save if filename provided
        if save_filename:
            save_path = os.path.join(self.save_dir, save_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        # Only show if display=True
        if display:
            plt.show()
        else:
            plt.close()

# Simple test function
def test_visualizer():
    """Test the visualization tools with synthetic data"""
    visualizer = GWDetectionVisualizer()
    
    print("Testing visualization with synthetic data...")
    
    # Create synthetic gravitational wave data
    sample_rate = 4096  # Hz
    duration = 1  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a synthetic "chirp" signal
    freq = 100 * np.exp(t/2)  # Exponentially increasing frequency
    amplitude = 1e-21 * np.exp(-((t-0.5)**2)/0.1)  # Gaussian envelope
    strain = amplitude * np.sin(2 * np.pi * freq * t)
    
    # Downsample for easier visualization
    ds_factor = 100
    ds_t = t[::ds_factor]
    ds_strain = strain[::ds_factor]
    
    # Plot strain data without display
    visualizer.plot_strain_data(ds_t, ds_strain, event_name="Synthetic GW", 
                              save_filename="test_strain.png")
    
    # Create synthetic quantum states (random for testing)
    n_qubits = 4
    n_states = len(ds_t)
    np.random.seed(42)  # For reproducibility
    
    # Generate random quantum states with amplitude variation reflecting the GW
    quantum_states = []
    for i, s in enumerate(ds_strain):
        # Create a state vector with normalized random complex values
        state = np.random.normal(0, 1, (2**n_qubits, 2)) + s * 1e21  # Adding strain influence
        state = state[:, 0] + 1j * state[:, 1]  # Convert to complex
        state = state / np.sqrt(np.sum(np.abs(state)**2))  # Normalize
        quantum_states.append(state)
    
    quantum_states = np.array(quantum_states)
    
    # Calculate QFI
    qfi = np.zeros(n_states - 1)
    for i in range(n_states - 1):
        fidelity = np.abs(np.vdot(quantum_states[i], quantum_states[i+1]))**2
        dt = ds_t[i+1] - ds_t[i]
        qfi[i] = 8 * (1 - np.sqrt(fidelity)) / (dt**2)
    
    # Calculate detection metric
    detection_metric = np.abs(quantum_states[:, -1])**2 - np.abs(quantum_states[:, 0])**2
    
    # Test full visualization without display
    visualizer.plot_quantum_detection(
        ds_t, ds_strain, quantum_states, 
        qfi=qfi, detection_metric=detection_metric,
        event_name="Synthetic GW", n_qubits=n_qubits,
        save_filename="test_detection.png"
    )
    
    # Test entanglement comparison without display
    results_dict = {
        "linear": {"times": ds_t, "strain": ds_strain, "qfi": qfi * 0.8, "qfi_snr": 2.5},
        "star": {"times": ds_t, "strain": ds_strain, "qfi": qfi, "qfi_snr": 3.2},
        "full": {"times": ds_t, "strain": ds_strain, "qfi": qfi * 1.2, "qfi_snr": 3.8}
    }
    
    visualizer.plot_entanglement_comparison(
        results_dict, event_name="Synthetic GW",
        save_filename="test_entanglement_comparison.png"
    )
    
    # Test qubit scaling without display
    qubit_results = {
        "4": {"max_qfi": 1000, "qfi_snr": 2.0, "execution_time": 0.5},
        "6": {"max_qfi": 2000, "qfi_snr": 3.0, "execution_time": 2.0},
        "8": {"max_qfi": 4000, "qfi_snr": 3.5, "execution_time": 8.0},
        "10": {"max_qfi": 8000, "qfi_snr": 4.0, "execution_time": 32.0}
    }
    
    visualizer.plot_qubit_scaling(
        qubit_results, metric="qfi_snr",
        event_name="Synthetic GW",
        save_filename="test_qubit_scaling.png"
    )
    
    print("Visualization tests completed successfully!")
    print("All plots were saved to the 'data/visualization' directory.")
    return visualizer

if __name__ == "__main__":
    test_visualizer()
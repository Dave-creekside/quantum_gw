# qgw_detector/quantum/error_corrected_detector.py
import pennylane as qml
import numpy as np
import time
from tqdm import tqdm
import os
import sys

# Add the parent directory to the path to import our GPU monitor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qgw_detector.utils.gpu_monitor import GPUMonitor
from qgw_detector.quantum.error_correction.repetition_code import RepetitionCode

class ErrorCorrectedQuantumGWDetector:
    """
    Error-corrected quantum gravitational wave detector using simple repetition codes
    """
    def __init__(self, n_logical_qubits=4, entanglement_type="star", use_gpu=True):
        """
        Initialize the error-corrected quantum detector
        
        Args:
            n_logical_qubits: Number of logical qubits in the detector
            entanglement_type: Type of entanglement ('linear', 'star', or 'full')
            use_gpu: Whether to use GPU acceleration
        """
        self.n_logical_qubits = n_logical_qubits
        self.entanglement_type = entanglement_type
        self.use_gpu = use_gpu
        
        # Initialize repetition code
        self.error_code = RepetitionCode(n_logical_qubits=n_logical_qubits)
        
        # Total number of physical qubits needed
        self.n_physical_qubits = self.error_code.n_physical_qubits
        
        # Set up GPU monitoring
        self.gpu_monitor = GPUMonitor(log_dir="logs/gpu")
        
        # Calculate number of phases based on entanglement topology
        if self.entanglement_type == "linear":
            self.n_phases = self.n_logical_qubits - 1
        elif self.entanglement_type == "star":
            self.n_phases = self.n_logical_qubits - 1
        elif self.entanglement_type == "full":
            self.n_phases = self.n_logical_qubits * (self.n_logical_qubits - 1) // 2
        else:
            raise ValueError(f"Unknown entanglement type: {self.entanglement_type}")
        
        # Create the quantum circuit
        self._create_circuit()
    
    def _create_circuit(self):
        """Create the error-corrected quantum circuit for gravitational wave detection"""
        # Log GPU state before creating device
        if self.use_gpu:
            self.gpu_monitor.log("Before creating quantum device (EC)")
        
        # Select appropriate device (GPU or CPU)
        try:
            if self.use_gpu:
                self.device = qml.device('lightning.gpu', wires=self.n_physical_qubits)
                print(f"✅ Using GPU-accelerated simulator with {self.n_physical_qubits} physical qubits")
                print(f"   ({self.n_logical_qubits} logical qubits with error correction)")
            else:
                self.device = qml.device('lightning.qubit', wires=self.n_physical_qubits)
                print(f"Using CPU simulator with {self.n_physical_qubits} physical qubits")
                print(f"({self.n_logical_qubits} logical qubits with error correction)")
                
        except Exception as e:
            print(f"❌ Error initializing device: {e}")
            print("Falling back to CPU simulator")
            self.device = qml.device('lightning.qubit', wires=self.n_physical_qubits)
            self.use_gpu = False
        
        # Log GPU state after creating device
        if self.use_gpu:
            self.gpu_monitor.log("After creating quantum device (EC)")
        
        # Define the quantum circuit with error correction
        @qml.qnode(self.device)
        def circuit(phases):
            # Apply the logical circuit using repetition code
            self.error_code.apply_logical_circuit(
                phases=phases,
                entanglement_type=self.entanglement_type
            )
            
            # Return the full quantum state
            return qml.state()
        
        self.circuit = circuit
    
    def process_gw_data(self, times, strain, downsample_factor=100, scale_factor=1e21):
        """
        Process gravitational wave data through the error-corrected quantum circuit
        
        Args:
            times: Time array
            strain: Strain values
            downsample_factor: Factor to downsample data
            scale_factor: Scale factor to amplify strain values
            
        Returns:
            tuple: (downsampled_times, downsampled_strain, quantum_states)
        """
        # Downsample to reduce computation time
        ds_indices = np.arange(0, len(times), downsample_factor)
        ds_times = times[ds_indices]
        ds_strain = strain[ds_indices]
        
        print(f"\nProcessing {len(ds_strain)} time points with {self.n_logical_qubits} logical qubits")
        print(f"Encoded into {self.n_physical_qubits} physical qubits using repetition code")
        print(f"Entanglement type: {self.entanglement_type}")
        print(f"Number of phases: {self.n_phases}")
        print(f"Strain amplification factor: {scale_factor:.2e}")
        
        # Log GPU state before main processing loop
        if self.use_gpu:
            self.gpu_monitor.log("Before processing GW data (EC)")
        
        # Process data through circuit
        quantum_states = []
        
        # Scale factor to convert strain to phase (π multiplier for phase in radians)
        scale_factor *= np.pi
        
        # Record processing start time
        start_time = time.time()
        
        # Process each time point
        for i, s in enumerate(tqdm(ds_strain)):
            # Create phase array from strain value
            phases = np.ones(self.n_phases) * s * scale_factor
            
            # Execute quantum circuit
            state = self.circuit(phases)
            
            # Store resulting quantum state
            quantum_states.append(state)
            
            # Periodically log GPU state during processing
            if self.use_gpu and i > 0 and i % max(len(ds_strain)//5, 1) == 0:
                self.gpu_monitor.log(f"During processing (step {i+1}/{len(ds_strain)}) (EC)", print_info=False)
        
        # Record processing end time
        end_time = time.time()
        process_time = end_time - start_time
        
        # Log final GPU state
        if self.use_gpu:
            self.gpu_monitor.log("After processing GW data (EC)")
            self.gpu_monitor.summary()
        
        print(f"\nProcessing completed in {process_time:.2f} seconds")
        print(f"Average time per state: {process_time/len(ds_strain):.4f} seconds")
        
        return ds_times, ds_strain, np.array(quantum_states)
    
    def calculate_qfi(self, states, times):
        """
        Calculate quantum Fisher information from quantum states
        
        Args:
            states: Array of quantum states
            times: Corresponding time points
            
        Returns:
            array: Quantum Fisher Information values
        """
        n_steps = len(states)
        qfi = np.zeros(n_steps - 1)
        
        # Log GPU state if using GPU
        if self.use_gpu:
            self.gpu_monitor.log("Before QFI calculation (EC)")
        
        print("Calculating Quantum Fisher Information...")
        
        # Calculate QFI between each pair of adjacent states
        for i in tqdm(range(n_steps - 1)):
            # Calculate fidelity between adjacent states
            fidelity = np.abs(np.vdot(states[i], states[i+1]))**2
            
            # QFI calculation (based on fidelity rate of change)
            dt = times[i+1] - times[i]
            if dt > 0:
                qfi[i] = 8 * (1 - np.sqrt(fidelity)) / (dt**2)
            else:
                qfi[i] = 0
        
        # Log GPU state after QFI calculation
        if self.use_gpu:
            self.gpu_monitor.log("After QFI calculation (EC)")
        
        # Calculate QFI statistics
        max_qfi = np.max(qfi)
        mean_qfi = np.mean(qfi)
        std_qfi = np.std(qfi)
        snr = max_qfi / std_qfi if std_qfi > 0 else 0
        
        print(f"QFI Calculation complete:")
        print(f"  Max QFI: {max_qfi:.4f}")
        print(f"  Mean QFI: {mean_qfi:.4f}")
        print(f"  QFI SNR: {snr:.4f}")
        
        return qfi
    
    def calculate_detection_metric(self, states):
        """
        Calculate a detection metric based on state probabilities
        
        Args:
            states: Array of quantum states
            
        Returns:
            array: Detection metric values
        """
        # Calculate probabilities
        probabilities = np.abs(states)**2
        
        # For error-corrected states, we'll use a simplified approach
        # that focuses on the all-zeros and all-ones states
        all_zeros_idx = 0  # All qubits in |0⟩
        all_ones_idx = -1  # All qubits in |1⟩
        
        # Detection metric: difference between |11...1⟩ and |00...0⟩ probability
        detection_metric = probabilities[:, all_ones_idx] - probabilities[:, all_zeros_idx]
        
        return detection_metric
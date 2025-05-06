# qgw_detector/quantum/circuits.py
import pennylane as qml
import numpy as np
import time
from tqdm import tqdm
import os
import sys

# Add the parent directory to the path to import our GPU monitor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qgw_detector.utils.gpu_monitor import GPUMonitor

class QuantumGWDetector:
    """
    Quantum Gravitational Wave Detector using relativistic quantum circuits
    """
    def __init__(self, n_qubits=8, entanglement_type="star", use_gpu=True, 
             use_zx_opt=False, zx_opt_level=1):
        """
        Initialize the quantum detector
    
        Args:
        n_qubits: Number of qubits in the detector
        entanglement_type: Type of entanglement ('linear', 'star', or 'full')
        use_gpu: Whether to use GPU acceleration
        use_zx_opt: Whether to use ZX-calculus optimization
        zx_opt_level: ZX optimization level (1-3)
        """
        self.n_qubits = n_qubits
        self.entanglement_type = entanglement_type
        self.use_gpu = use_gpu
        self.use_zx_opt = use_zx_opt  # New parameter
        self.zx_opt_level = zx_opt_level  # New parameter
    
    # Rest of your existing initialization code...
        
        
        # Set up GPU monitoring
        self.gpu_monitor = GPUMonitor(log_dir="logs/gpu")
        
        # Calculate number of phases based on entanglement topology
        if self.entanglement_type == "linear":
            self.n_phases = self.n_qubits - 1
        elif self.entanglement_type == "star":
            self.n_phases = self.n_qubits - 1
        elif self.entanglement_type == "full":
            self.n_phases = self.n_qubits * (self.n_qubits - 1) // 2
        else:
            raise ValueError(f"Unknown entanglement type: {self.entanglement_type}")
        
        # Create the quantum circuit
        self._create_circuit()
    
    def _create_circuit(self):
        """Create the quantum circuit for gravitational wave detection"""
        # Log GPU state before creating device
        if self.use_gpu:
            self.gpu_monitor.log("Before creating quantum device")
        
        # Select appropriate device (GPU or CPU)
        try:
            if self.use_gpu:
                self.device = qml.device('lightning.gpu', wires=self.n_qubits)
                print(f"✅ Using GPU-accelerated simulator with {self.n_qubits} qubits")
            else:
                self.device = qml.device('lightning.qubit', wires=self.n_qubits)
                print(f"Using CPU simulator with {self.n_qubits} qubits")
                
        except Exception as e:
            print(f"❌ Error initializing device: {e}")
            print("Falling back to CPU simulator")
            self.device = qml.device('lightning.qubit', wires=self.n_qubits)
            self.use_gpu = False
        
        # Log GPU state after creating device
        if self.use_gpu:
            self.gpu_monitor.log("After creating quantum device")
        
        # Define the quantum circuit
        @qml.qnode(self.device)
        def circuit(phases):
            # Initialize qubits in superposition
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # Apply entanglement and phase shifts based on topology
            if self.entanglement_type == "linear":
                # Linear chain of entanglement
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
                    # Apply relativistic phase shift
                    qml.ControlledPhaseShift(phases[i], wires=[i, i+1])
                
            elif self.entanglement_type == "star":
                # Star topology with central qubit
                for i in range(1, self.n_qubits):
                    qml.CNOT(wires=[0, i])
                    # Apply relativistic phase shift
                    qml.ControlledPhaseShift(phases[i-1], wires=[0, i])
                
            elif self.entanglement_type == "full":
                # Fully connected topology
                phase_idx = 0
                for i in range(self.n_qubits):
                    for j in range(i+1, self.n_qubits):
                        qml.CNOT(wires=[i, j])
                        # Apply relativistic phase shift
                        qml.ControlledPhaseShift(phases[phase_idx], wires=[i, j])
                        phase_idx += 1
            
            # Return the full quantum state
            return qml.state()
        
        # Apply PyZX optimization if requested
        if self.use_zx_opt:
            try:
                # Import PyZX optimizer
                from qgw_detector.quantum.optimizers import PyZXOptimizer
                
                # Create optimizer
                optimizer = PyZXOptimizer(optimization_level=self.zx_opt_level)
                
                # Optimize circuit
                optimized_circuit, self.zx_stats = optimizer.optimize_circuit(
                    circuit, self.n_qubits, self.entanglement_type
                )
                
                # Use optimized circuit
                self.circuit = optimized_circuit
                print(f"Circuit optimized with PyZX (level {self.zx_opt_level})")
                
            except ImportError as e:
                print(f"PyZX optimizer not available: {e}")
                print("Using original circuit")
                self.circuit = circuit
                self.zx_stats = {"available": False, "error": str(e)}
        else:
            # Use original circuit
            self.circuit = circuit
            self.zx_stats = {"used": False}


        self.circuit = circuit
    
    def process_gw_data(self, times, strain, downsample_factor=100, scale_factor=1e21):
        """
        Process gravitational wave data through the quantum circuit
        
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
        
        print(f"\nProcessing {len(ds_strain)} time points with {self.n_qubits} qubits...")
        print(f"Entanglement type: {self.entanglement_type}")
        print(f"Number of phases: {self.n_phases}")
        print(f"Strain amplification factor: {scale_factor:.2e}")
        
        # Log GPU state before main processing loop
        if self.use_gpu:
            self.gpu_monitor.log("Before processing GW data")
        
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
                self.gpu_monitor.log(f"During processing (step {i+1}/{len(ds_strain)})", print_info=False)
        
        # Record processing end time
        end_time = time.time()
        process_time = end_time - start_time
        
        # Log final GPU state
        if self.use_gpu:
            self.gpu_monitor.log("After processing GW data")
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
            self.gpu_monitor.log("Before QFI calculation")
        
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
            self.gpu_monitor.log("After QFI calculation")
        
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
        Calculate a simple detection metric based on state probabilities
        
        Args:
            states: Array of quantum states
            
        Returns:
            array: Detection metric values
        """
        # Calculate probabilities
        probabilities = np.abs(states)**2
        
        # Use difference between |000...0⟩ and |111...1⟩ as detection metric
        p0 = probabilities[:, 0]  # |000...0⟩ state
        p1 = probabilities[:, -1]  # |111...1⟩ state
        
        detection_metric = p1 - p0
        
        return detection_metric

# Function to test the quantum circuit with random phases
def test_circuit(n_qubits=4, entanglement_type="star", use_gpu=True):
    """Test the quantum circuit with random phases"""
    print(f"Testing quantum circuit with {n_qubits} qubits, {entanglement_type} entanglement...")
    
    # Create detector
    detector = QuantumGWDetector(n_qubits, entanglement_type, use_gpu)
    
    # Create random phases
    phases = np.random.random(detector.n_phases) * np.pi
    
    # Run circuit
    state = detector.circuit(phases)
    
    print(f"Circuit executed successfully with {n_qubits} qubits")
    print(f"State vector shape: {state.shape}")
    
    # Print a few amplitudes
    print("First few state amplitudes:")
    for i in range(min(4, len(state))):
        binary = format(i, f'0{n_qubits}b')
        print(f"|{binary}⟩: {state[i]}")
    
    return detector

# Main function to test the module
if __name__ == "__main__":
    # Test with small number of qubits
    detector = test_circuit(n_qubits=6, entanglement_type="star", use_gpu=True)
    
    # Create some synthetic test data 
    print("\nTesting with synthetic gravitational wave data...")
    sample_rate = 4096  # Hz
    duration = 1  # seconds (short for testing)
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a synthetic "chirp" signal resembling a gravitational wave
    freq = 100 * np.exp(t/2)  # Exponentially increasing frequency
    amplitude = 1e-21 * np.exp(-((t-0.5)**2)/0.1)  # Gaussian envelope
    strain = amplitude * np.sin(2 * np.pi * freq * t)
    
    # Process through quantum detector
    ds_times, ds_strain, quantum_states = detector.process_gw_data(
        t, strain, downsample_factor=200
    )
    
    # Calculate QFI
    qfi = detector.calculate_qfi(quantum_states, ds_times)
    
    print("Test completed successfully!")
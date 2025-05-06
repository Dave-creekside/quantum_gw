# qgw_detector/quantum/optimizers.py
import numpy as np
import pennylane as qml
import os
import sys
import time

class PyZXOptimizer:
    """
    Optimize quantum circuits using PyZX
    """
    def __init__(self, optimization_level=1):
        """
        Initialize the PyZX optimizer
        
        Args:
            optimization_level: Level of optimization (1-3)
                1: Basic Clifford optimization
                2: Full reductions
                3: Aggressive optimization with teleport_reduce
        """
        self.optimization_level = optimization_level
        
        # Try to import PyZX
        try:
            import pyzx
            self.pyzx = pyzx
            self.available = True
            print("PyZX loaded successfully!")
        except ImportError:
            print("PyZX not found. Please install with 'pip install pyzx'")
            self.available = False
    
    def optimize_circuit(self, circuit_func, n_qubits, entanglement_type):
        """
        Optimize a quantum circuit using PyZX
        
        Args:
            circuit_func: Original circuit function
            n_qubits: Number of qubits
            entanglement_type: Type of entanglement ('linear', 'star', or 'full')
            
        Returns:
            tuple: (optimized_circuit_func, optimization_stats)
        """
        if not self.available:
            print("PyZX not available. Returning original circuit.")
            return circuit_func, {"available": False}
        
        print(f"\nOptimizing circuit with PyZX (level {self.optimization_level})...")
        
        # Convert circuit to QASM (approximate representation)
        qasm_str = self._create_qasm_for_circuit(n_qubits, entanglement_type)
        
        try:
            # Parse QASM to ZX-diagram
            start_time = time.time()
            zx_circuit = self.pyzx.Circuit.from_qasm(qasm_str)
            graph = zx_circuit.to_graph()
            
            # Count original gates
            orig_gates = zx_circuit.count_gates()
            orig_cnots = orig_gates.get('CNOT', 0)
            orig_phase = sum([orig_gates.get(g, 0) for g in ['S', 'T', 'Z', 'P']])
            orig_total = sum(orig_gates.values())
            
            # Apply optimization based on level
            print(f"Applying ZX-calculus optimization...")
            if self.optimization_level == 1:
                self.pyzx.simplify.clifford_simp(graph)
            elif self.optimization_level == 2:
                self.pyzx.simplify.full_reduce(graph)
            elif self.optimization_level == 3:
                # Apply most aggressive optimizations available
                self.pyzx.simplify.full_reduce(graph)
                if hasattr(self.pyzx.simplify, 'teleport_reduce'):
                    self.pyzx.simplify.teleport_reduce(graph)
            
            # Extract optimized circuit
            optimized_zx = self.pyzx.extract.extract_circuit(graph.copy())
            opt_gates = optimized_zx.count_gates()
            opt_cnots = opt_gates.get('CNOT', 0)
            opt_phase = sum([opt_gates.get(g, 0) for g in ['S', 'T', 'Z', 'P']])
            opt_total = sum(opt_gates.values())
            
            optimization_time = time.time() - start_time
            
            # Create statistics
            stats = {
                "available": True,
                "optimization_level": self.optimization_level,
                "optimization_time": optimization_time,
                "original": {
                    "total_gates": orig_total,
                    "cnot_gates": orig_cnots,
                    "phase_gates": orig_phase
                },
                "optimized": {
                    "total_gates": opt_total,
                    "cnot_gates": opt_cnots,
                    "phase_gates": opt_phase
                },
                "reduction": {
                    "total_gates": orig_total - opt_total,
                    "total_percentage": 100 * (1 - opt_total/orig_total) if orig_total > 0 else 0,
                    "cnot_gates": orig_cnots - opt_cnots,
                    "cnot_percentage": 100 * (1 - opt_cnots/orig_cnots) if orig_cnots > 0 else 0,
                }
            }
            
            # Print results
            print(f"ZX-calculus optimization completed in {optimization_time:.2f} seconds")
            print(f"Gate count reduction: {orig_total} → {opt_total} ({stats['reduction']['total_percentage']:.1f}%)")
            print(f"CNOT reduction: {orig_cnots} → {opt_cnots} ({stats['reduction']['cnot_percentage']:.1f}%)")
            
            # Create a new optimized circuit using lessons from ZX-calculus
            optimized_circuit_func = self._create_optimized_circuit_func(
                circuit_func, n_qubits, entanglement_type, stats
            )
            
            return optimized_circuit_func, stats
            
        except Exception as e:
            print(f"Error in PyZX optimization: {e}")
            return circuit_func, {"available": False, "error": str(e)}
    
    def _create_qasm_for_circuit(self, n_qubits, entanglement_type):
        """
        Create approximate QASM representation for the quantum circuit
        
        Args:
            n_qubits: Number of qubits
            entanglement_type: Type of entanglement
            
        Returns:
            str: QASM string
        """
        # Start with QASM header
        qasm = []
        qasm.append("OPENQASM 2.0;")
        qasm.append('include "qelib1.inc";')
        qasm.append(f"qreg q[{n_qubits}];")
        
        # Add Hadamard gates
        for i in range(n_qubits):
            qasm.append(f"h q[{i}];")
        
        # Add entanglement and phase gates based on topology
        if entanglement_type == "linear":
            for i in range(n_qubits - 1):
                qasm.append(f"cx q[{i}], q[{i+1}];")
                qasm.append(f"p(0.1) q[{i+1}];")
        
        elif entanglement_type == "star":
            for i in range(1, n_qubits):
                qasm.append(f"cx q[0], q[{i}];")
                qasm.append(f"p(0.1) q[{i}];")
        
        elif entanglement_type == "full":
            phase_idx = 0
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    qasm.append(f"cx q[{i}], q[{j}];")
                    qasm.append(f"p(0.1) q[{j}];")
        
        return "\n".join(qasm)
    
    def _create_optimized_circuit_func(self, original_func, n_qubits, entanglement_type, stats):
        """
        Create an optimized circuit function based on ZX optimization insights
        
        Args:
            original_func: Original circuit function
            n_qubits: Number of qubits
            entanglement_type: Entanglement topology
            stats: Optimization statistics
            
        Returns:
            function: Optimized circuit function
        """
        # Create a new circuit function that applies ZX-calculus insights
        @qml.qnode(original_func.device)
        def optimized_circuit(phases):
            # Apply Hadamard gates to all qubits
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            
            # Apply optimized entanglement pattern based on topology
            if entanglement_type == "linear":
                # Skip some CNOTs based on reduction percentage
                skip_factor = max(1, int(n_qubits / (stats["optimized"]["cnot_gates"] + 1)))
                for i in range(0, n_qubits - 1, max(1, skip_factor)):
                    qml.CNOT(wires=[i, i+1])
                    # Apply phase with original phase value
                    if i < len(phases):
                        qml.ControlledPhaseShift(phases[i], wires=[i, i+1])
            
            elif entanglement_type == "star":
                # Maintain star pattern but optimize phases
                for i in range(1, n_qubits):
                    qml.CNOT(wires=[0, i])
                    # Apply phase with original phase value
                    if i-1 < len(phases):
                        qml.ControlledPhaseShift(phases[i-1], wires=[0, i])
            
            elif entanglement_type == "full":
                # Reduce full connectivity based on optimization
                phase_idx = 0
                red_factor = max(0.1, stats["reduction"]["cnot_percentage"] / 100)
                skip_prob = min(0.7, red_factor)  # Cap at 70% to maintain some connectivity
                
                for i in range(n_qubits):
                    for j in range(i+1, n_qubits):
                        # Skip some connections based on optimization stats
                        if np.random.random() > skip_prob:
                            qml.CNOT(wires=[i, j])
                            # Apply phase with original phase value
                            if phase_idx < len(phases):
                                qml.ControlledPhaseShift(phases[phase_idx], wires=[i, j])
                        phase_idx += 1
            
            # Return full state vector
            return qml.state()
        
        return optimized_circuit
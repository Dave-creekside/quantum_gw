import pennylane as qml
import numpy as np
import time

# Try to use the GPU device
try:
    dev_gpu = qml.device("lightning.gpu", wires=10)
    print("🟢 Successfully initialized GPU device")
    
    # Define a circuit
    @qml.qnode(dev_gpu)
    def circuit_gpu(params):
        for i in range(10):
            qml.RX(params[i], wires=i)
            qml.RY(params[i+10], wires=i)
        
        for i in range(9):
            qml.CNOT(wires=[i, i+1])
        
        return qml.state()
    
    # Use a CPU device for comparison
    dev_cpu = qml.device("lightning.qubit", wires=10)
    
    @qml.qnode(dev_cpu)
    def circuit_cpu(params):
        for i in range(10):
            qml.RX(params[i], wires=i)
            qml.RY(params[i+10], wires=i)
        
        for i in range(9):
            qml.CNOT(wires=[i, i+1])
        
        return qml.state()
    
    # Generate random parameters
    params = np.random.random(20)
    
    # Time GPU execution
    start_gpu = time.time()
    state_gpu = circuit_gpu(params)
    end_gpu = time.time()
    
    # Time CPU execution
    start_cpu = time.time()
    state_cpu = circuit_cpu(params)
    end_cpu = time.time()
    
    # Print results
    print(f"\nGPU execution time: {end_gpu - start_gpu:.6f} seconds")
    print(f"CPU execution time: {end_cpu - start_cpu:.6f} seconds")
    print(f"Speedup factor: {(end_cpu - start_cpu) / (end_gpu - start_gpu):.2f}x")
    
    # Verify results are the same
    if np.allclose(state_gpu, state_cpu):
        print("✅ GPU and CPU results match!")
    else:
        print("❌ Warning: GPU and CPU results differ")
    
except Exception as e:
    print(f"❌ Failed to initialize GPU device: {e}")
    print("\nFallback to CPU device is recommended.")

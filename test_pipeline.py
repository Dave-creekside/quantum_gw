# Run this in your Python terminal or save as a script
import numpy as np
from qgw_detector.data.ligo import fetch_gw_data, preprocess_gw_data
from qgw_detector.quantum.circuits import QuantumGWDetector
import matplotlib.pyplot as plt

# Get LIGO data
print("Fetching LIGO data...")
times, strain, sample_rate = fetch_gw_data("GW150914")
proc_times, proc_strain = preprocess_gw_data(times, strain, sample_rate)

# Stage 1: 4-qubit star detector
print("\n=== Stage 1: 4-qubit star detector ===")
detector1 = QuantumGWDetector(n_qubits=4, entanglement_type="star")
times1, strain1, states1 = detector1.process_gw_data(proc_times, proc_strain, downsample_factor=200)
metric1 = detector1.calculate_detection_metric(states1)
qfi1 = detector1.calculate_qfi(states1, times1)
snr1 = np.max(qfi1)/np.std(qfi1) if np.std(qfi1) > 0 else 0
print(f"Stage 1 SNR: {snr1:.4f}")

# Stage 2: 6-qubit full detector (using output from stage 1)
print("\n=== Stage 2: 6-qubit full detector ===")
detector2 = QuantumGWDetector(n_qubits=6, entanglement_type="full")
times2, strain2, states2 = detector2.process_gw_data(times1, metric1, downsample_factor=1)
metric2 = detector2.calculate_detection_metric(states2)
qfi2 = detector2.calculate_qfi(states2, times2)
snr2 = np.max(qfi2)/np.std(qfi2) if np.std(qfi2) > 0 else 0
print(f"Stage 2 SNR: {snr2:.4f}")

# Stage 3: 4-qubit linear detector (using output from stage 2)
print("\n=== Stage 3: 4-qubit linear detector ===")
detector3 = QuantumGWDetector(n_qubits=4, entanglement_type="linear")
times3, strain3, states3 = detector3.process_gw_data(times2, metric2, downsample_factor=1)
metric3 = detector3.calculate_detection_metric(states3)
qfi3 = detector3.calculate_qfi(states3, times3)
snr3 = np.max(qfi3)/np.std(qfi3) if np.std(qfi3) > 0 else 0
print(f"Stage 3 SNR: {snr3:.4f}")

# Compare results
print("\n=== Pipeline Results Summary ===")
print(f"Stage 1 (4-star) SNR:   {snr1:.4f}")
print(f"Stage 2 (6-full) SNR:   {snr2:.4f}")
print(f"Stage 3 (4-linear) SNR: {snr3:.4f}")
print(f"Improvement from stage 1 to 3: {snr3/snr1:.2f}x")

# Optional: Create a visualization of the pipeline
plt.figure(figsize=(12, 15))

# Plot strain data
plt.subplot(4, 1, 1)
plt.plot(proc_times, proc_strain * 1e21, 'k-')
plt.title("LIGO Data (Input)")
plt.ylabel("Strain (×10²¹)")
plt.grid(True)

# Plot stage 1 output
plt.subplot(4, 1, 2)
plt.plot(times1, metric1, 'b-')
plt.title("Stage 1: 4-qubit star detector output")
plt.ylabel("Detection Metric")
plt.grid(True)

# Plot stage 2 output
plt.subplot(4, 1, 3)
plt.plot(times2, metric2, 'g-')
plt.title("Stage 2: 6-qubit full detector output")
plt.ylabel("Detection Metric")
plt.grid(True)

# Plot stage 3 output
plt.subplot(4, 1, 4)
plt.plot(times3, metric3, 'r-')
plt.title("Stage 3: 4-qubit linear detector output")
plt.xlabel("Time (s)")
plt.ylabel("Detection Metric")
plt.grid(True)

plt.tight_layout()
plt.savefig("pipeline_test_results.png", dpi=300)
plt.close()

print("\nVisualization saved to 'pipeline_test_results.png'")
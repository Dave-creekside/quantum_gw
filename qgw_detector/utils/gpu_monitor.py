# qgw_detector/utils/gpu_monitor.py
import time
import os
import numpy as np
import torch
import subprocess
from datetime import datetime

class GPUMonitor:
    """Class for monitoring and logging GPU performance metrics"""
    
    def __init__(self, log_dir="logs", log_to_file=True):
        """Initialize the GPU monitor"""
        self.gpu_available = torch.cuda.is_available()
        self.log_to_file = log_to_file
        
        # Create log directory if needed
        if log_to_file:
            self.log_dir = log_dir
            os.makedirs(log_dir, exist_ok=True)
            
            # Create a log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log_file = os.path.join(log_dir, f"gpu_log_{timestamp}.txt")
            
            # Initialize log file with header
            with open(self.log_file, "w") as f:
                f.write("Timestamp,Operation,DeviceName,MemoryAllocated(MB),MemoryReserved(MB),")
                f.write("MaxMemoryAllocated(MB),Utilization(%),Temperature(C)\n")
        
        # Initialize storage for tracking performance over time
        self.timestamps = []
        self.memory_allocated = []
        self.utilization = []
        self.temperature = []
        
        # Print initial GPU information
        if self.gpu_available:
            print("\n=== GPU INFORMATION ===")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  CUDA Capability: {torch.cuda.get_device_capability(i)}")
                
            print(f"Current device: {torch.cuda.current_device()}")
            print("=====================\n")
        else:
            print("No GPU available. Using CPU only.")
    
    def get_gpu_info(self):
        """Get detailed GPU information using NVIDIA tools and PyTorch"""
        info = {"available": self.gpu_available}
        
        if not self.gpu_available:
            return info
        
        # Get PyTorch GPU info
        info["device_count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(info["current_device"])
        info["memory_allocated"] = torch.cuda.memory_allocated() / (1024**2)  # MB
        info["memory_reserved"] = torch.cuda.memory_reserved() / (1024**2)    # MB
        info["max_memory_allocated"] = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        
        # Try to get additional info from nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            if result.returncode == 0:
                util, mem_used, mem_total, temp = result.stdout.strip().split(',')
                info["utilization"] = float(util.strip())
                info["temperature"] = float(temp.strip())
                info["nvidia_smi_memory_used"] = float(mem_used.strip())
                info["nvidia_smi_memory_total"] = float(mem_total.strip())
        except Exception as e:
            # nvidia-smi might not be available or accessible
            info["error"] = f"Error fetching nvidia-smi data: {str(e)}"
        
        return info
    
    def log(self, operation="", reset_max=False, print_info=True):
        """Log current GPU utilization with an optional operation name"""
        if not self.gpu_available:
            if print_info:
                print("GPU not available for monitoring")
            return None
        
        # Get current timestamp
        timestamp = datetime.now()
        
        # Get GPU info
        info = self.get_gpu_info()
        
        # Store data for tracking
        self.timestamps.append(timestamp)
        self.memory_allocated.append(info["memory_allocated"])
        self.utilization.append(info.get("utilization", 0))
        self.temperature.append(info.get("temperature", 0))
        
        # Print information if requested
        if print_info:
            print("\n=== GPU UTILIZATION " + "="*50)
            if operation:
                print(f"Operation: {operation}")
            print(f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            print(f"Device: {info['device_name']}")
            print(f"Memory allocated: {info['memory_allocated']:.2f} MB")
            print(f"Memory reserved: {info['memory_reserved']:.2f} MB")
            print(f"Max memory allocated: {info['max_memory_allocated']:.2f} MB")
            
            if "utilization" in info:
                print(f"GPU utilization: {info['utilization']}%")
                print(f"GPU temperature: {info['temperature']}°C")
                print(f"Memory usage: {info['nvidia_smi_memory_used']}/{info['nvidia_smi_memory_total']} MB")
            
            print("="*70)
        
        # Log to file if enabled
        if self.log_to_file:
            with open(self.log_file, "a") as f:
                f.write(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]},")
                f.write(f"{operation},")
                f.write(f"{info['device_name']},")
                f.write(f"{info['memory_allocated']:.2f},")
                f.write(f"{info['memory_reserved']:.2f},")
                f.write(f"{info['max_memory_allocated']:.2f},")
                f.write(f"{info.get('utilization', '')},")
                f.write(f"{info.get('temperature', '')}\n")
        
        # Reset max memory stats if requested
        if reset_max:
            torch.cuda.reset_max_memory_stats()
            if print_info:
                print("Max memory statistics reset")
        
        return info
    
    def summary(self):
        """Print summary statistics of GPU usage"""
        if not self.gpu_available or len(self.memory_allocated) == 0:
            print("No GPU monitoring data available")
            return
        
        print("\n=== GPU MONITORING SUMMARY " + "="*40)
        print(f"Total monitoring points: {len(self.timestamps)}")
        print(f"Monitoring period: {(self.timestamps[-1] - self.timestamps[0]).total_seconds():.2f} seconds")
        
        # Memory statistics
        peak_memory = max(self.memory_allocated)
        avg_memory = np.mean(self.memory_allocated)
        print(f"Peak memory usage: {peak_memory:.2f} MB")
        print(f"Average memory usage: {avg_memory:.2f} MB")
        
        # Utilization statistics (if available)
        if len(self.utilization) > 0 and any(u > 0 for u in self.utilization):
            peak_util = max(self.utilization)
            avg_util = np.mean(self.utilization)
            print(f"Peak GPU utilization: {peak_util:.2f}%")
            print(f"Average GPU utilization: {avg_util:.2f}%")
        
        # Temperature statistics (if available)
        if len(self.temperature) > 0 and any(t > 0 for t in self.temperature):
            peak_temp = max(self.temperature)
            avg_temp = np.mean(self.temperature)
            print(f"Peak temperature: {peak_temp:.2f}°C")
            print(f"Average temperature: {avg_temp:.2f}°C")
        
        print("="*70)
        
        if self.log_to_file:
            print(f"Detailed logs saved to: {self.log_file}")
    
    def plot_metrics(self, save_path=None):
        """Plot GPU metrics over time"""
        import matplotlib.pyplot as plt
        
        if not self.gpu_available or len(self.timestamps) < 2:
            print("Insufficient GPU monitoring data for plotting")
            return
        
        # Convert timestamps to seconds from start
        start_time = self.timestamps[0]
        time_seconds = [(t - start_time).total_seconds() for t in self.timestamps]
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot memory usage
        axes[0].plot(time_seconds, self.memory_allocated, 'b-', linewidth=2)
        axes[0].set_ylabel('Memory (MB)')
        axes[0].set_title('GPU Memory Allocated')
        axes[0].grid(True)
        
        # Plot utilization if available
        if len(self.utilization) > 0 and any(u > 0 for u in self.utilization):
            axes[1].plot(time_seconds, self.utilization, 'g-', linewidth=2)
            axes[1].set_ylabel('Utilization (%)')
            axes[1].set_title('GPU Utilization')
            axes[1].grid(True)
        else:
            axes[1].text(0.5, 0.5, 'Utilization data not available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[1].transAxes)
        
        # Plot temperature if available
        if len(self.temperature) > 0 and any(t > 0 for t in self.temperature):
            axes[2].plot(time_seconds, self.temperature, 'r-', linewidth=2)
            axes[2].set_ylabel('Temperature (°C)')
            axes[2].set_title('GPU Temperature')
            axes[2].grid(True)
        else:
            axes[2].text(0.5, 0.5, 'Temperature data not available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[2].transAxes)
        
        # Add common x-axis label
        axes[2].set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"GPU metrics plot saved to {save_path}")
        
        plt.show()

# Simple test function
def test_gpu_monitor():
    """Test the GPU monitoring class"""
    monitor = GPUMonitor()
    
    print("Testing GPU monitoring...")
    
    # Log initial state
    monitor.log("Initial state")
    
    # Create a PyTorch tensor on GPU (if available)
    if torch.cuda.is_available():
        # Allocate some memory
        print("Allocating 100MB tensor...")
        tensor = torch.zeros((25000, 1000), device="cuda")
        monitor.log("After tensor allocation")
        
        # Allocate more memory
        print("Allocating another 200MB tensor...")
        tensor2 = torch.zeros((50000, 1000), device="cuda")
        monitor.log("After second tensor allocation")
        
        # Free first tensor
        print("Freeing first tensor...")
        del tensor
        torch.cuda.empty_cache()
        monitor.log("After freeing first tensor")
        
        # Free second tensor
        print("Freeing second tensor...")
        del tensor2
        torch.cuda.empty_cache()
        monitor.log("After freeing all tensors", reset_max=True)
    
    # Print summary
    monitor.summary()
    
    # Plot metrics
    monitor.plot_metrics()
    
    return monitor

if __name__ == "__main__":
    test_gpu_monitor()
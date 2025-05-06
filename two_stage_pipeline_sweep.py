# two_stage_pipeline_sweep.py
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime
import itertools

from qgw_detector.data.ligo import fetch_gw_data, preprocess_gw_data
from qgw_detector.quantum.circuits import QuantumGWDetector

def run_two_stage_pipeline(event_name, pipeline_config, output_dir, 
                         downsample_factor=200, scale_factor=1e21):
    """
    Run a two-stage pipeline configuration and save results
    
    Args:
        event_name: Name of gravitational wave event
        pipeline_config: List of (qubit, topology) tuples [stage1, stage2]
        output_dir: Directory to save results
        downsample_factor: Factor to downsample data
        scale_factor: Scale factor to amplify strain values
    
    Returns:
        dict: Pipeline results
    """
    pipeline_str = "_".join([f"{q}-{t}" for q, t in pipeline_config])
    print(f"\n{'='*80}")
    print(f" TWO-STAGE PIPELINE: {pipeline_str} ")
    print(f"{'='*80}")
    
    # Get LIGO data (only do this once per event)
    if not hasattr(run_two_stage_pipeline, 'data_cache'):
        print("Fetching LIGO data...")
        times, strain, sample_rate = fetch_gw_data(event_name)
        proc_times, proc_strain = preprocess_gw_data(times, strain, sample_rate)
        run_two_stage_pipeline.data_cache = {
            'times': proc_times,
            'strain': proc_strain
        }
    
    # Get cached data
    current_times = run_two_stage_pipeline.data_cache['times']
    current_strain = run_two_stage_pipeline.data_cache['strain']
    
    # For visualization
    all_stages_data = [{
        'name': 'LIGO',
        'times': current_times,
        'data': current_strain * 1e21,
        'label': 'Strain (×10²¹)'
    }]
    
    # Store results for each stage
    pipeline_results = {
        'event_name': event_name,
        'pipeline_config': pipeline_str,
        'stages': []
    }
    
    # Process through each stage (just 2 stages now)
    for i, (n_qubits, topology) in enumerate(pipeline_config):
        print(f"\n=== Stage {i+1}: {n_qubits}-qubit {topology} detector ===")
        
        # Create detector
        detector = QuantumGWDetector(
            n_qubits=n_qubits,
            entanglement_type=topology,
            use_gpu=True
        )
        
        # Apply stage-specific downsample factor (only first stage)
        stage_downsample = downsample_factor if i == 0 else 1
        
        # Process data
        start_time = time.time()
        ds_times, ds_strain, quantum_states = detector.process_gw_data(
            current_times, current_strain, 
            downsample_factor=stage_downsample,
            scale_factor=scale_factor
        )
        process_time = time.time() - start_time
        
        # Calculate metrics
        qfi_values = detector.calculate_qfi(quantum_states, ds_times)
        detection_metric = detector.calculate_detection_metric(quantum_states)
        
        # Calculate SNR and other stats
        max_qfi = float(np.max(qfi_values))
        mean_qfi = float(np.mean(qfi_values))
        std_qfi = float(np.std(qfi_values))
        snr = max_qfi / std_qfi if std_qfi > 0 else 0
        
        # Store stage results
        stage_results = {
            'stage': i+1,
            'qubits': n_qubits,
            'topology': topology,
            'execution_time': process_time,
            'max_qfi': max_qfi,
            'mean_qfi': mean_qfi,
            'qfi_snr': snr
        }
        pipeline_results['stages'].append(stage_results)
        
        # Store data for visualization
        all_stages_data.append({
            'name': f"Stage {i+1}: {n_qubits}-{topology}",
            'times': ds_times,
            'data': detection_metric,
            'label': 'Detection Metric'
        })
        
        # Use this stage's output as input to the next stage
        current_times = ds_times
        current_strain = detection_metric
        
        print(f"Stage {i+1} complete:")
        print(f"  QFI SNR: {snr:.4f}")
        print(f"  Process time: {process_time:.2f} seconds")
    
    # Calculate pipeline-level metrics
    first_stage_snr = pipeline_results['stages'][0]['qfi_snr']
    final_stage_snr = pipeline_results['stages'][-1]['qfi_snr']
    improvement = final_stage_snr / first_stage_snr if first_stage_snr > 0 else 0
    
    pipeline_results['summary'] = {
        'first_stage_snr': first_stage_snr,
        'final_stage_snr': final_stage_snr,
        'improvement_factor': improvement,
        'total_execution_time': sum(stage['execution_time'] for stage in pipeline_results['stages'])
    }
    
    # Create visualization
    fig_path = os.path.join(output_dir, f"pipeline_{pipeline_str}.png")
    
    # Create multi-panel plot
    n_stages = len(all_stages_data)
    plt.figure(figsize=(14, 4 * n_stages))
    
    for i, stage_data in enumerate(all_stages_data):
        plt.subplot(n_stages, 1, i+1)
        plt.plot(stage_data['times'], stage_data['data'])
        plt.title(stage_data['name'])
        plt.ylabel(stage_data['label'])
        plt.grid(True)
        
        if i == n_stages - 1:
            plt.xlabel("Time (s)")
    
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results as JSON
    results_path = os.path.join(output_dir, f"results_{pipeline_str}.json")
    with open(results_path, 'w') as f:
        json.dump(pipeline_results, f, indent=2)
    
    print(f"\nPipeline results saved to: {results_path}")
    print(f"Pipeline visualization saved to: {fig_path}")
    
    # Print summary
    print("\n" + "="*80)
    print(" PIPELINE SUMMARY ")
    print("="*80)
    print(f"Pipeline: {pipeline_str}")
    print(f"First stage SNR: {first_stage_snr:.4f}")
    print(f"Final stage SNR: {final_stage_snr:.4f}")
    print(f"Improvement factor: {improvement:.2f}x")
    print(f"Total execution time: {pipeline_results['summary']['total_execution_time']:.2f} seconds")
    print("="*80 + "\n")
    
    return pipeline_results

def generate_two_stage_configurations():
    """Generate all viable two-stage pipeline configurations with expanded qubit range"""
    # Full topology is limited to 4 and 6 qubits
    full_configs = [(4, 'full'), (6, 'full')]
    
    # Star and linear topologies can use 4, 6, and 8 qubits
    star_linear_configs = []
    for qubits in [4, 6, 8]:
        for topology in ['star', 'linear']:
            star_linear_configs.append((qubits, topology))
    
    # All possible stage configurations
    all_configs = full_configs + star_linear_configs
    
    # Generate all 2-stage pipelines
    pipelines = list(itertools.product(all_configs, repeat=2))
    
    return pipelines

def run_two_stage_pipeline_sweep(event_name="GW150914", 
                               downsample_factor=200, 
                               scale_factor=1e21):
    """Run a comprehensive test of all viable two-stage pipeline configurations"""
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"data/two_stage_sweep_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate pipeline configurations
    pipeline_configs = generate_two_stage_configurations()
    print(f"Testing {len(pipeline_configs)} two-stage pipeline configurations")
    
    # Track all results
    all_results = []
    
    # Run each pipeline
    for i, pipeline_config in enumerate(pipeline_configs):
        print(f"\nRunning pipeline {i+1}/{len(pipeline_configs)}")
        try:
            result = run_two_stage_pipeline(
                event_name=event_name,
                pipeline_config=pipeline_config,
                output_dir=output_dir,
                downsample_factor=downsample_factor,
                scale_factor=scale_factor
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error running pipeline: {e}")
            # Log the error
            with open(os.path.join(output_dir, "errors.log"), "a") as f:
                pipeline_str = "_".join([f"{q}-{t}" for q, t in pipeline_config])
                f.write(f"Error with pipeline {pipeline_str}: {str(e)}\n")
    
    # Create summary table
    summary_table = []
    for result in all_results:
        summary_table.append({
            'pipeline': result['pipeline_config'],
            'final_snr': result['summary']['final_stage_snr'],
            'improvement': result['summary']['improvement_factor'],
            'execution_time': result['summary']['total_execution_time']
        })
    
    # Sort by final SNR (best first)
    summary_table.sort(key=lambda x: x['final_snr'], reverse=True)
    
    # Save summary table
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary_table, f, indent=2)
    
    # Create ranking visualization
    plt.figure(figsize=(14, 10))
    
    # Plot SNR ranking (top 15)
    pipelines = [item['pipeline'] for item in summary_table[:15]]
    snrs = [item['final_snr'] for item in summary_table[:15]]
    
    plt.subplot(2, 1, 1)
    plt.barh(range(len(pipelines)), snrs, color='skyblue')
    plt.yticks(range(len(pipelines)), pipelines)
    plt.xlabel('Final Stage SNR')
    plt.title('Top 15 Two-Stage Pipelines by SNR')
    plt.grid(True, axis='x')
    
    # Plot improvement ranking (top 15)
    summary_table.sort(key=lambda x: x['improvement'], reverse=True)
    pipelines = [item['pipeline'] for item in summary_table[:15]]
    improvements = [item['improvement'] for item in summary_table[:15]]
    
    plt.subplot(2, 1, 2)
    plt.barh(range(len(pipelines)), improvements, color='lightgreen')
    plt.yticks(range(len(pipelines)), pipelines)
    plt.xlabel('Improvement Factor')
    plt.title('Top 15 Two-Stage Pipelines by Improvement Factor')
    plt.grid(True, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pipeline_rankings.png"), dpi=300, bbox_inches='tight')
    
    # Create a detailed performance table as text
    with open(os.path.join(output_dir, "detailed_ranking.txt"), "w") as f:
        f.write("DETAILED TWO-STAGE PIPELINE PERFORMANCE RANKING\n")
        f.write("="*50 + "\n\n")
        
        # Sort by SNR
        summary_table.sort(key=lambda x: x['final_snr'], reverse=True)
        f.write("TOP 20 BY FINAL SNR:\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Rank':4} {'Pipeline':30} {'SNR':8} {'Improvement':12} {'Time (s)':8}\n")
        f.write("-"*50 + "\n")
        
        for i, item in enumerate(summary_table[:20]):
            f.write(f"{i+1:4} {item['pipeline']:30} {item['final_snr']:8.4f} {item['improvement']:12.2f}x {item['execution_time']:8.2f}\n")
        
        f.write("\n\n")
        
        # Sort by improvement
        summary_table.sort(key=lambda x: x['improvement'], reverse=True)
        f.write("TOP 20 BY IMPROVEMENT FACTOR:\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Rank':4} {'Pipeline':30} {'Improvement':12} {'SNR':8} {'Time (s)':8}\n")
        f.write("-"*50 + "\n")
        
        for i, item in enumerate(summary_table[:20]):
            f.write(f"{i+1:4} {item['pipeline']:30} {item['improvement']:12.2f}x {item['final_snr']:8.4f} {item['execution_time']:8.2f}\n")
    
    print(f"\nComprehensive test complete!")
    print(f"All results saved to: {output_dir}")
    print(f"\nTop 5 pipeline configurations by final SNR:")
    for i in range(min(5, len(summary_table))):
        print(f"{i+1}. {summary_table[i]['pipeline']} - SNR: {summary_table[i]['final_snr']:.4f}, Improvement: {summary_table[i]['improvement']:.2f}x")

if __name__ == "__main__":
    run_two_stage_pipeline_sweep(event_name="GW150914", downsample_factor=200)
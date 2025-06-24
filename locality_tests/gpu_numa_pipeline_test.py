#!/usr/bin/env python3
"""
GPU NUMA Pipeline Test with Double Buffering

Test script to evaluate end-to-end pipeline performance with overlapping
H2D, Compute, and D2H stages using double buffering across NUMA nodes.

Usage with numactl:
  numactl --cpunodebind=0 --membind=0 python gpu_numa_pipeline_test.py --gpu-id=5
  numactl --cpunodebind=2 --membind=2 python gpu_numa_pipeline_test.py --gpu-id=3
"""

import torch
import torch.cuda.nvtx as nvtx
import time
import argparse
import numpy as np
import os
import psutil
import sys
import threading
from collections import defaultdict
from enum import Enum

class PipelineStage(Enum):
    H2D = "h2d"
    COMPUTE = "compute"
    D2H = "d2h"

def get_numa_info():
    """Get current process NUMA binding info"""
    try:
        pid = os.getpid()
        proc = psutil.Process(pid)
        cpu_affinity = proc.cpu_affinity()
        return {
            'pid': pid,
            'cpu_affinity': cpu_affinity,
            'cpu_count': len(cpu_affinity),
            'cpu_ranges': _get_cpu_ranges(cpu_affinity)
        }
    except:
        return {'pid': os.getpid(), 'cpu_affinity': 'unknown', 'cpu_count': 'unknown'}

def _get_cpu_ranges(cpu_list):
    """Convert CPU list to readable ranges"""
    if not cpu_list or cpu_list == 'unknown':
        return 'unknown'

    sorted_cpus = sorted(cpu_list)
    ranges = []
    start = sorted_cpus[0]
    end = start

    for cpu in sorted_cpus[1:]:
        if cpu == end + 1:
            end = cpu
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = end = cpu

    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    return ','.join(ranges)

def get_gpu_info(gpu_id):
    """Get GPU information"""
    try:
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}

        if gpu_id >= torch.cuda.device_count():
            return {'error': f'GPU {gpu_id} not available. Available: 0-{torch.cuda.device_count()-1}'}

        with torch.cuda.device(gpu_id):
            props = torch.cuda.get_device_properties(gpu_id)
            return {
                'name': props.name,
                'major': props.major,
                'minor': props.minor,
                'total_memory': props.total_memory,
                'multi_processor_count': props.multi_processor_count,
                'memory_mb': props.total_memory / (1024 * 1024),
                'compute_capability': f"{props.major}.{props.minor}"
            }
    except Exception as e:
        return {'error': str(e)}

def create_matmul_workload(batch_size, tensor_shape, compute_iterations, gpu_id):
    """Create matmul workload tensors for compute simulation"""
    C, H, W = tensor_shape

    # Create matrices for matmul that will simulate NN workload
    # Flatten spatial dimensions and create weight matrix
    input_features = H * W
    hidden_features = max(512, input_features // 4)  # Reasonable hidden size

    # Weight matrices for multiple layers
    weights = []
    for i in range(max(1, compute_iterations // 50)):  # Multiple layers based on iterations
        weight = torch.randn(input_features if i == 0 else hidden_features, 
                           hidden_features, device=f'cuda:{gpu_id}')
        weights.append(weight)

    return weights, input_features, hidden_features

class DoubleBufferedPipeline:
    """Double buffered pipeline for H2D -> Compute -> D2H"""

    def __init__(self, batch_size, tensor_shape, gpu_id, compute_iterations, pin_memory=True):
        self.batch_size = batch_size
        self.tensor_shape = tensor_shape
        self.gpu_id = gpu_id
        self.compute_iterations = compute_iterations
        self.pin_memory = pin_memory

        # Create CUDA streams for pipeline stages
        self.h2d_stream = torch.cuda.Stream(device=gpu_id)
        self.compute_stream = torch.cuda.Stream(device=gpu_id)
        self.d2h_stream = torch.cuda.Stream(device=gpu_id)

        # Double buffers on GPU
        self.gpu_buffer_a = torch.zeros(batch_size, *tensor_shape, device=f'cuda:{gpu_id}')
        self.gpu_buffer_b = torch.zeros(batch_size, *tensor_shape, device=f'cuda:{gpu_id}')

        # CPU result buffers
        self.cpu_result_buffer_a = [torch.zeros(*tensor_shape) for _ in range(batch_size)]
        self.cpu_result_buffer_b = [torch.zeros(*tensor_shape) for _ in range(batch_size)]

        if pin_memory:
            for i in range(batch_size):
                self.cpu_result_buffer_a[i] = self.cpu_result_buffer_a[i].pin_memory()
                self.cpu_result_buffer_b[i] = self.cpu_result_buffer_b[i].pin_memory()

        # Create matmul workload
        self.weights, self.input_features, self.hidden_features = create_matmul_workload(
            batch_size, tensor_shape, compute_iterations, gpu_id
        )

        # Pipeline state
        self.current_buffer = 'A'  # 'A' or 'B'
        self.batch_results = []

    def get_current_buffers(self):
        """Get current GPU and CPU buffers"""
        if self.current_buffer == 'A':
            return self.gpu_buffer_a, self.cpu_result_buffer_a
        else:
            return self.gpu_buffer_b, self.cpu_result_buffer_b

    def get_next_buffers(self):
        """Get next GPU and CPU buffers"""
        if self.current_buffer == 'A':
            return self.gpu_buffer_b, self.cpu_result_buffer_b
        else:
            return self.gpu_buffer_a, self.cpu_result_buffer_a

    def swap_buffers(self):
        """Swap current buffer"""
        self.current_buffer = 'B' if self.current_buffer == 'A' else 'A'

    def h2d_transfer(self, cpu_batch, batch_idx, nvtx_prefix):
        """Perform H2D transfer to current buffer"""
        gpu_buffer, _ = self.get_current_buffers()

        with torch.cuda.stream(self.h2d_stream):
            with nvtx.range(f"{nvtx_prefix}_h2d_batch_{batch_idx}"):
                for i, tensor in enumerate(cpu_batch):
                    with nvtx.range(f"h2d_tensor_{i}"):
                        gpu_buffer[i].copy_(tensor, non_blocking=True)

    def compute_workload(self, batch_idx, nvtx_prefix):
        """Perform compute workload on current buffer"""
        gpu_buffer, _ = self.get_current_buffers()

        with torch.cuda.stream(self.compute_stream):
            with nvtx.range(f"{nvtx_prefix}_compute_batch_{batch_idx}"):
                # Wait for H2D to complete
                self.compute_stream.wait_stream(self.h2d_stream)

                # Flatten tensors for matmul
                batch_flattened = gpu_buffer.view(self.batch_size, -1)

                # Multiple matmul operations to simulate NN compute
                x = batch_flattened
                for iteration in range(self.compute_iterations):
                    with nvtx.range(f"matmul_iter_{iteration}"):
                        weight_idx = min(iteration // 50, len(self.weights) - 1)
                        x = torch.matmul(x, self.weights[weight_idx])
                        if iteration % 10 == 0:  # Add some nonlinearity occasionally
                            x = torch.relu(x)

                # Reshape back to original tensor shape (truncate if needed)
                if x.shape[1] >= self.tensor_shape[0] * self.tensor_shape[1] * self.tensor_shape[2]:
                    result = x[:, :self.tensor_shape[0] * self.tensor_shape[1] * self.tensor_shape[2]]
                    gpu_buffer.copy_(result.view(self.batch_size, *self.tensor_shape))
                else:
                    # Pad if compute result is smaller
                    padded = torch.zeros_like(batch_flattened)
                    padded[:, :x.shape[1]] = x
                    gpu_buffer.copy_(padded.view(self.batch_size, *self.tensor_shape))

    def d2h_transfer(self, batch_idx, nvtx_prefix):
        """Perform D2H transfer from current buffer"""
        gpu_buffer, cpu_result_buffer = self.get_current_buffers()

        with torch.cuda.stream(self.d2h_stream):
            with nvtx.range(f"{nvtx_prefix}_d2h_batch_{batch_idx}"):
                # Wait for compute to complete
                self.d2h_stream.wait_stream(self.compute_stream)

                for i in range(len(cpu_result_buffer)):
                    with nvtx.range(f"d2h_tensor_{i}"):
                        cpu_result_buffer[i].copy_(gpu_buffer[i], non_blocking=True)

    def wait_for_completion(self, stages=None):
        """Wait for specified pipeline stages to complete"""
        if stages is None:
            stages = [PipelineStage.H2D, PipelineStage.COMPUTE, PipelineStage.D2H]

        if PipelineStage.H2D in stages:
            self.h2d_stream.synchronize()
        if PipelineStage.COMPUTE in stages:
            self.compute_stream.synchronize()
        if PipelineStage.D2H in stages:
            self.d2h_stream.synchronize()

def run_pipeline_test(
    gpu_id=0,
    tensor_shape=(3, 224, 224),
    num_samples=1000,
    batch_size=10,
    warmup_samples=100,
    memory_size_mb=512,
    compute_iterations=100,
    skip_warmup=False,
    deterministic=False,
    nsys_report=None,
    pin_memory=True,
    fill_pattern='random',
    pipeline_modes=None,
    sync_frequency=10
):
    """
    Run comprehensive pipeline performance test with double buffering

    Args:
        gpu_id: GPU device ID
        tensor_shape: Tensor shape as (C, H, W)
        num_samples: Number of samples to test
        batch_size: Batch size for pipeline
        warmup_samples: Number of warmup samples
        memory_size_mb: CPU memory pool size in MB
        compute_iterations: Number of compute iterations per batch
        skip_warmup: Skip warmup phase
        deterministic: Enable deterministic algorithms
        nsys_report: NSYS report name for automation
        pin_memory: Use pinned memory for transfers
        fill_pattern: Memory fill pattern
        pipeline_modes: List of pipeline modes to test
        sync_frequency: How often to synchronize for timing measurements
    """

    # Set deterministic behavior if requested
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(42)
        np.random.seed(42)

    # Default pipeline modes
    if pipeline_modes is None:
        pipeline_modes = ['sequential', 'overlapped_simple', 'overlapped_double_buffer']

    numa_info = get_numa_info()
    gpu_info = get_gpu_info(gpu_id)

    print(f"=== GPU NUMA Pipeline Performance Test ===")
    print(f"Process PID: {numa_info['pid']}")
    print(f"CPU Affinity: {numa_info['cpu_ranges']}")
    print(f"GPU ID: {gpu_id}")
    if 'error' in gpu_info:
        print(f"GPU Error: {gpu_info['error']}")
        sys.exit(1)
    print(f"GPU: {gpu_info['name']} ({gpu_info['memory_mb']:.0f} MB)")
    print(f"Compute Capability: {gpu_info['compute_capability']}")
    print(f"Tensor Shape: {tensor_shape}")
    print(f"Batch Size: {batch_size}")
    print(f"Total Samples: {num_samples}")
    print(f"Warmup Samples: {warmup_samples if not skip_warmup else 0}")
    print(f"CPU Memory Pool: {memory_size_mb} MB ({fill_pattern})")
    print(f"Compute Iterations: {compute_iterations}")
    print(f"Pin Memory: {pin_memory}")
    print(f"Pipeline Modes: {pipeline_modes}")
    print(f"Sync Frequency: {sync_frequency}")
    print(f"Deterministic: {deterministic}")
    if nsys_report:
        print(f"NSYS Report: {nsys_report}")
    print("=" * 60)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    torch.cuda.set_device(gpu_id)

    # Allocate CPU memory pool
    print(f"Allocating {memory_size_mb} MB CPU memory pool...")
    if fill_pattern == 'random':
        memory_pool = torch.randn(memory_size_mb * 1024 * 1024 // 4)
    elif fill_pattern == 'sequential':
        memory_pool = torch.arange(memory_size_mb * 1024 * 1024 // 4, dtype=torch.float32)
    else:  # zeros
        memory_pool = torch.zeros(memory_size_mb * 1024 * 1024 // 4)

    print(f"CPU memory pool allocated: {memory_pool.element_size() * memory_pool.nelement() / 1024 / 1024:.2f} MB")

    # Pre-generate test data
    print("Pre-generating test data...")
    total_samples = (0 if skip_warmup else warmup_samples) + num_samples

    cpu_tensors = []
    for i in range(total_samples):
        if fill_pattern == 'random':
            tensor = torch.randn(*tensor_shape)
        elif fill_pattern == 'sequential':
            tensor = torch.arange(np.prod(tensor_shape), dtype=torch.float32).reshape(tensor_shape) + i
        else:  # zeros
            tensor = torch.zeros(*tensor_shape)

        if pin_memory:
            tensor = tensor.pin_memory()

        cpu_tensors.append(tensor)

        # Touch memory pool periodically
        if i % 50 == 0:
            _ = memory_pool[:(1024*1024)].sum()

    print(f"Generated {len(cpu_tensors)} CPU tensors")

    # Results storage
    results = defaultdict(list)

    # Run tests for each pipeline mode
    for pipeline_mode in pipeline_modes:
        print(f"\n--- Running {pipeline_mode.upper()} Pipeline Test ---")

        if not skip_warmup and warmup_samples > 0:
            print(f"Warmup phase: {warmup_samples} samples...")
            _run_pipeline_mode(
                pipeline_mode, cpu_tensors[:warmup_samples], batch_size, gpu_id,
                compute_iterations, f"warmup_{pipeline_mode}", pin_memory, sync_frequency,
                is_warmup=True
            )

        print(f"Test phase: {num_samples} samples...")
        start_idx = 0 if skip_warmup else warmup_samples
        test_tensors = cpu_tensors[start_idx:start_idx + num_samples]

        test_results = _run_pipeline_mode(
            pipeline_mode, test_tensors, batch_size, gpu_id,
            compute_iterations, f"test_{pipeline_mode}", pin_memory, sync_frequency,
            is_warmup=False
        )

        results[pipeline_mode] = test_results

    # Print results summary
    _print_pipeline_results_summary(results, pipeline_modes)

    print("\n=== Pipeline Test Completed ===")
    if nsys_report:
        print(f"Run nsys analysis: nsys stats {nsys_report}")

def _run_pipeline_mode(pipeline_mode, tensors, batch_size, gpu_id, compute_iterations, 
                      nvtx_prefix, pin_memory, sync_frequency, is_warmup=False):
    """Run a specific pipeline mode"""
    results = []
    num_batches = (len(tensors) + batch_size - 1) // batch_size

    if pipeline_mode == 'sequential':
        results = _run_sequential_pipeline(
            tensors, batch_size, gpu_id, compute_iterations, nvtx_prefix, sync_frequency, is_warmup
        )
    elif pipeline_mode == 'overlapped_simple':
        results = _run_overlapped_simple_pipeline(
            tensors, batch_size, gpu_id, compute_iterations, nvtx_prefix, sync_frequency, is_warmup
        )
    elif pipeline_mode == 'overlapped_double_buffer':
        results = _run_overlapped_double_buffer_pipeline(
            tensors, batch_size, gpu_id, compute_iterations, nvtx_prefix, pin_memory, sync_frequency, is_warmup
        )

    return results

def _run_sequential_pipeline(tensors, batch_size, gpu_id, compute_iterations, nvtx_prefix, sync_frequency, is_warmup):
    """Run sequential pipeline: H2D -> Compute -> D2H for each batch"""
    results = []

    # Pre-allocate GPU memory
    gpu_batch = torch.zeros(batch_size, *tensors[0].shape, device=f'cuda:{gpu_id}')
    cpu_result_batch = [torch.zeros(*tensors[0].shape) for _ in range(batch_size)]

    # Create compute workload
    weights, input_features, hidden_features = create_matmul_workload(
        batch_size, tensors[0].shape, compute_iterations, gpu_id
    )

    with nvtx.range(f"{nvtx_prefix}_sequential"):
        for batch_idx in range(0, len(tensors), batch_size):
            batch_end = min(batch_idx + batch_size, len(tensors))
            current_batch_size = batch_end - batch_idx
            batch_tensors = tensors[batch_idx:batch_end]

            with nvtx.range(f"{nvtx_prefix}_batch_{batch_idx // batch_size}"):
                start_time = time.time()

                # H2D Transfer
                with nvtx.range(f"h2d_seq_{batch_idx // batch_size}"):
                    for i, tensor in enumerate(batch_tensors):
                        gpu_batch[i].copy_(tensor, non_blocking=True)
                    torch.cuda.synchronize()

                h2d_time = time.time()

                # Compute
                with nvtx.range(f"compute_seq_{batch_idx // batch_size}"):
                    batch_flattened = gpu_batch[:current_batch_size].view(current_batch_size, -1)
                    x = batch_flattened
                    for iteration in range(compute_iterations):
                        weight_idx = min(iteration // 50, len(weights) - 1)
                        x = torch.matmul(x, weights[weight_idx])
                        if iteration % 10 == 0:
                            x = torch.relu(x)

                    # Reshape back
                    if x.shape[1] >= np.prod(tensors[0].shape):
                        result = x[:, :np.prod(tensors[0].shape)]
                        gpu_batch[:current_batch_size].copy_(result.view(current_batch_size, *tensors[0].shape))

                    torch.cuda.synchronize()

                compute_time = time.time()

                # D2H Transfer
                with nvtx.range(f"d2h_seq_{batch_idx // batch_size}"):
                    for i in range(current_batch_size):
                        cpu_result_batch[i].copy_(gpu_batch[i], non_blocking=True)
                    torch.cuda.synchronize()

                end_time = time.time()

                if not is_warmup:
                    results.append({
                        'mode': 'sequential',
                        'batch_idx': batch_idx // batch_size,
                        'batch_size': current_batch_size,
                        'total_duration': end_time - start_time,
                        'h2d_duration': h2d_time - start_time,
                        'compute_duration': compute_time - h2d_time,
                        'd2h_duration': end_time - compute_time
                    })

                # Progress reporting
                if ((batch_idx // batch_size) + 1) % sync_frequency == 0:
                    progress = batch_end / len(tensors) * 100
                    print(f"  Progress: {progress:.1f}% ({batch_end}/{len(tensors)})")

    return results

def _run_overlapped_simple_pipeline(tensors, batch_size, gpu_id, compute_iterations, nvtx_prefix, sync_frequency, is_warmup):
    """Run overlapped pipeline with separate streams"""
    results = []

    # Create pipeline
    pipeline = DoubleBufferedPipeline(batch_size, tensors[0].shape, gpu_id, compute_iterations)

    with nvtx.range(f"{nvtx_prefix}_overlapped_simple"):
        for batch_idx in range(0, len(tensors), batch_size):
            batch_end = min(batch_idx + batch_size, len(tensors))
            current_batch_size = batch_end - batch_idx
            batch_tensors = tensors[batch_idx:batch_end]
            batch_num = batch_idx // batch_size

            with nvtx.range(f"{nvtx_prefix}_batch_{batch_num}"):
                start_time = time.time()

                # Launch all stages with stream dependencies
                pipeline.h2d_transfer(batch_tensors, batch_num, nvtx_prefix)
                pipeline.compute_workload(batch_num, nvtx_prefix)
                pipeline.d2h_transfer(batch_num, nvtx_prefix)

                # Wait for pipeline completion
                pipeline.wait_for_completion()

                end_time = time.time()

                if not is_warmup:
                    results.append({
                        'mode': 'overlapped_simple',
                        'batch_idx': batch_num,
                        'batch_size': current_batch_size,
                        'total_duration': end_time - start_time
                    })

                # Progress reporting
                if (batch_num + 1) % sync_frequency == 0:
                    progress = batch_end / len(tensors) * 100
                    print(f"  Progress: {progress:.1f}% ({batch_end}/{len(tensors)})")

    return results

def _run_overlapped_double_buffer_pipeline(tensors, batch_size, gpu_id, compute_iterations, 
                                          nvtx_prefix, pin_memory, sync_frequency, is_warmup):
    """Run fully overlapped double buffered pipeline"""
    results = []

    # Create pipeline
    pipeline = DoubleBufferedPipeline(batch_size, tensors[0].shape, gpu_id, compute_iterations, pin_memory)

    with nvtx.range(f"{nvtx_prefix}_double_buffer"):
        num_batches = (len(tensors) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(tensors))
            current_batch_size = batch_end - batch_start
            batch_tensors = tensors[batch_start:batch_end]

            with nvtx.range(f"{nvtx_prefix}_batch_{batch_idx}"):
                start_time = time.time()

                if batch_idx == 0:
                    # First batch: just start the pipeline
                    pipeline.h2d_transfer(batch_tensors, batch_idx, nvtx_prefix)
                    pipeline.compute_workload(batch_idx, nvtx_prefix)
                    pipeline.d2h_transfer(batch_idx, nvtx_prefix)
                else:
                    # Overlapped execution: start next batch while finishing previous
                    # Swap to next buffer
                    pipeline.swap_buffers()

                    # Start H2D for current batch (in new buffer)
                    pipeline.h2d_transfer(batch_tensors, batch_idx, nvtx_prefix)

                    # Wait for previous batch D2H to complete and record results
                    prev_batch_idx = batch_idx - 1
                    pipeline.d2h_stream.synchronize()

                    # Start compute for current batch
                    pipeline.compute_workload(batch_idx, nvtx_prefix)

                    # Start D2H for current batch
                    pipeline.d2h_transfer(batch_idx, nvtx_prefix)

                # For last batch, wait for completion
                if batch_idx == num_batches - 1:
                    pipeline.wait_for_completion()

                end_time = time.time()

                if not is_warmup and batch_idx > 0:  # Skip first batch timing (no overlap yet)
                    results.append({
                        'mode': 'overlapped_double_buffer',
                        'batch_idx': batch_idx,
                        'batch_size': current_batch_size,
                        'total_duration': end_time - start_time
                    })

                # Progress reporting
                if (batch_idx + 1) % sync_frequency == 0:
                    progress = batch_end / len(tensors) * 100
                    print(f"  Progress: {progress:.1f}% ({batch_end}/{len(tensors)})")

    return results

def _print_pipeline_results_summary(results, pipeline_modes):
    """Print comprehensive pipeline results summary"""
    print("\n=== Pipeline Results Summary ===")

    for mode in pipeline_modes:
        if mode not in results or not results[mode]:
            continue

        mode_results = results[mode]
        total_durations = [r['total_duration'] for r in mode_results]

        print(f"\n{mode.upper()} Pipeline Statistics:")
        print(f"  Batches: {len(mode_results)}")
        print(f"  Total Duration (s): mean={np.mean(total_durations):.6f}, std={np.std(total_durations):.6f}")
        print(f"  Total Duration (s): min={np.min(total_durations):.6f}, max={np.max(total_durations):.6f}")

        # Mode-specific details
        if mode == 'sequential' and mode_results:
            h2d_durations = [r['h2d_duration'] for r in mode_results if 'h2d_duration' in r]
            compute_durations = [r['compute_duration'] for r in mode_results if 'compute_duration' in r]
            d2h_durations = [r['d2h_duration'] for r in mode_results if 'd2h_duration' in r]

            if h2d_durations:
                print(f"  H2D Duration (s): mean={np.mean(h2d_durations):.6f}")
            if compute_durations:
                print(f"  Compute Duration (s): mean={np.mean(compute_durations):.6f}")
            if d2h_durations:
                print(f"  D2H Duration (s): mean={np.mean(d2h_durations):.6f}")

            if h2d_durations and compute_durations and d2h_durations:
                total_stages = np.mean(h2d_durations) + np.mean(compute_durations) + np.mean(d2h_durations)
                overlap_efficiency = (total_stages - np.mean(total_durations)) / total_stages * 100
                print(f"  Overlap Efficiency: {overlap_efficiency:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='GPU NUMA Pipeline Performance Test with Double Buffering')

    # Core parameters
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--shape', nargs=3, type=int, default=[3, 224, 224],
                        help='Tensor shape as C H W (default: 3 224 224)')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples to process (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size (default: 10)')
    parser.add_argument('--warmup-samples', type=int, default=100,
                        help='Number of warmup samples (default: 100)')
    parser.add_argument('--memory-size-mb', type=int, default=512,
                        help='CPU memory pool size in MB (default: 512)')
    parser.add_argument('--compute-iterations', type=int, default=100,
                        help='Compute iterations per batch (default: 100)')

    # Pipeline control
    parser.add_argument('--pipeline-modes', nargs='+',
                        choices=['sequential', 'overlapped_simple', 'overlapped_double_buffer'],
                        default=['sequential', 'overlapped_simple', 'overlapped_double_buffer'],
                        help='Pipeline modes to test')
    parser.add_argument('--sync-frequency', type=int, default=10,
                        help='Synchronization frequency for progress reporting (default: 10)')

    # Test control
    parser.add_argument('--skip-warmup', action='store_true',
                        help='Skip warmup phase for quick tests')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic algorithms in PyTorch')

    # Memory and performance options
    parser.add_argument('--no-pin-memory', action='store_true',
                        help='Disable pinned memory (default: enabled)')
    parser.add_argument('--fill-pattern', choices=['random', 'sequential', 'zeros'], default='random',
                        help='Memory fill pattern (default: random)')

    # Profiling and automation
    parser.add_argument('--nsys-report', type=str, default=None,
                        help='NSYS report name for automation')

    args = parser.parse_args()

    run_pipeline_test(
        gpu_id=args.gpu_id,
        tensor_shape=tuple(args.shape),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        warmup_samples=args.warmup_samples,
        memory_size_mb=args.memory_size_mb,
        compute_iterations=args.compute_iterations,
        skip_warmup=args.skip_warmup,
        deterministic=args.deterministic,
        nsys_report=args.nsys_report,
        pin_memory=not args.no_pin_memory,
        fill_pattern=args.fill_pattern,
        pipeline_modes=args.pipeline_modes,
        sync_frequency=args.sync_frequency
    )

if __name__ == '__main__':
    main()

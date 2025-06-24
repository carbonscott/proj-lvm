#!/usr/bin/env python3
"""
GPU NUMA H2D/D2H Performance Test

Test script to evaluate Host-to-Device and Device-to-Host memory transfer
performance across different NUMA node bindings.

Usage with numactl:
  numactl --cpunodebind=0 --membind=0 python gpu_numa_h2d_d2h_test.py --gpu-id=5
  numactl --cpunodebind=2 --membind=2 python gpu_numa_h2d_d2h_test.py --gpu-id=3
"""

import torch
import torch.cuda.nvtx as nvtx
import time
import argparse
import numpy as np
import os
import psutil
import sys
from collections import defaultdict

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
                'memory_mb': props.total_memory / (1024 * 1024)
            }
    except Exception as e:
        return {'error': str(e)}

def allocate_cpu_memory_pool(memory_size_mb, fill_pattern='random'):
    """Allocate CPU memory pool with specified pattern"""
    print(f"Allocating {memory_size_mb} MB CPU memory pool...")

    if fill_pattern == 'random':
        memory_pool = torch.randn(memory_size_mb * 1024 * 1024 // 4)  # 4 bytes per float32
    elif fill_pattern == 'sequential':
        memory_pool = torch.arange(memory_size_mb * 1024 * 1024 // 4, dtype=torch.float32)
    else:  # zeros
        memory_pool = torch.zeros(memory_size_mb * 1024 * 1024 // 4)

    actual_mb = memory_pool.element_size() * memory_pool.nelement() / 1024 / 1024
    print(f"CPU memory pool allocated: {actual_mb:.2f} MB")
    return memory_pool

def run_h2d_d2h_test(
    gpu_id=0,
    tensor_shape=(3, 224, 224),
    num_samples=1000,
    batch_size=10,
    warmup_samples=100,
    memory_size_mb=512,
    skip_warmup=False,
    deterministic=False,
    nsys_report=None,
    pin_memory=True,
    fill_pattern='random',
    test_modes=None
):
    """
    Run comprehensive H2D/D2H performance test

    Args:
        gpu_id: GPU device ID
        tensor_shape: Tensor shape as (C, H, W)
        num_samples: Number of samples to test
        batch_size: Batch size for transfers
        warmup_samples: Number of warmup samples
        memory_size_mb: CPU memory pool size in MB
        skip_warmup: Skip warmup phase
        deterministic: Enable deterministic algorithms
        nsys_report: NSYS report name for automation
        pin_memory: Use pinned memory for transfers
        fill_pattern: Memory fill pattern ('random', 'sequential', 'zeros')
        test_modes: List of test modes to run
    """

    # Set deterministic behavior if requested
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(42)
        np.random.seed(42)

    # Default test modes
    if test_modes is None:
        test_modes = ['h2d_only', 'd2h_only', 'h2d_d2h_sequential', 'h2d_d2h_concurrent']

    numa_info = get_numa_info()
    gpu_info = get_gpu_info(gpu_id)

    print(f"=== GPU NUMA H2D/D2H Performance Test ===")
    print(f"Process PID: {numa_info['pid']}")
    print(f"CPU Affinity: {numa_info['cpu_ranges']}")
    print(f"GPU ID: {gpu_id}")
    if 'error' in gpu_info:
        print(f"GPU Error: {gpu_info['error']}")
        sys.exit(1)
    print(f"GPU: {gpu_info['name']} ({gpu_info['memory_mb']:.0f} MB)")
    print(f"Tensor Shape: {tensor_shape}")
    print(f"Batch Size: {batch_size}")
    print(f"Total Samples: {num_samples}")
    print(f"Warmup Samples: {warmup_samples if not skip_warmup else 0}")
    print(f"CPU Memory Pool: {memory_size_mb} MB ({fill_pattern})")
    print(f"Pin Memory: {pin_memory}")
    print(f"Test Modes: {test_modes}")
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
    cpu_memory_pool = allocate_cpu_memory_pool(memory_size_mb, fill_pattern)

    # Pre-generate test data
    print("Pre-generating test data...")
    total_samples = (0 if skip_warmup else warmup_samples) + num_samples

    # Estimate memory requirements
    tensor_size_bytes = np.prod(tensor_shape) * 4  # float32
    estimated_mb = (total_samples * tensor_size_bytes) / (1024 * 1024)
    print(f"Estimated memory for {total_samples} tensors: {estimated_mb:.1f} MB")

    # Generate CPU tensors
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
            _ = cpu_memory_pool[:(1024*1024)].sum()

    print(f"Generated {len(cpu_tensors)} CPU tensors")

    # Pre-allocate GPU memory
    print("Pre-allocating GPU memory...")
    gpu_batch = torch.zeros(batch_size, *tensor_shape, device=f'cuda:{gpu_id}')
    gpu_result_batch = torch.zeros(batch_size, *tensor_shape, device=f'cuda:{gpu_id}')
    print(f"GPU memory allocated: {(gpu_batch.numel() + gpu_result_batch.numel()) * 4 / (1024*1024):.1f} MB")

    # Results storage
    results = defaultdict(list)

    # Run tests for each mode
    for test_mode in test_modes:
        print(f"\n--- Running {test_mode.upper()} Test ---")

        if not skip_warmup and warmup_samples > 0:
            print(f"Warmup phase: {warmup_samples} samples...")
            _run_test_mode(
                test_mode, cpu_tensors[:warmup_samples], gpu_batch, gpu_result_batch,
                batch_size, gpu_id, f"warmup_{test_mode}", is_warmup=True
            )
            torch.cuda.synchronize()

        print(f"Test phase: {num_samples} samples...")
        start_idx = 0 if skip_warmup else warmup_samples
        test_tensors = cpu_tensors[start_idx:start_idx + num_samples]

        test_results = _run_test_mode(
            test_mode, test_tensors, gpu_batch, gpu_result_batch,
            batch_size, gpu_id, f"test_{test_mode}", is_warmup=False
        )

        results[test_mode] = test_results
        torch.cuda.synchronize()

    # Print results summary
    _print_results_summary(results, test_modes)

    print("\n=== Test Completed ===")
    if nsys_report:
        print(f"Run nsys analysis: nsys stats {nsys_report}")

def _run_test_mode(test_mode, tensors, gpu_batch, gpu_result_batch, batch_size, gpu_id, nvtx_prefix, is_warmup=False):
    """Run a specific test mode"""
    results = []

    with nvtx.range(f"{nvtx_prefix}_mode"):
        for batch_start in range(0, len(tensors), batch_size):
            batch_end = min(batch_start + batch_size, len(tensors))
            current_batch_size = batch_end - batch_start

            batch_tensors = tensors[batch_start:batch_end]
            batch_idx = batch_start // batch_size

            with nvtx.range(f"{nvtx_prefix}_batch_{batch_idx}"):
                if test_mode == 'h2d_only':
                    result = _test_h2d_only(batch_tensors, gpu_batch, current_batch_size, batch_idx, nvtx_prefix)
                elif test_mode == 'd2h_only':
                    result = _test_d2h_only(batch_tensors, gpu_result_batch, current_batch_size, batch_idx, nvtx_prefix)
                elif test_mode == 'h2d_d2h_sequential':
                    result = _test_h2d_d2h_sequential(batch_tensors, gpu_batch, gpu_result_batch, current_batch_size, batch_idx, nvtx_prefix)
                elif test_mode == 'h2d_d2h_concurrent':
                    result = _test_h2d_d2h_concurrent(batch_tensors, gpu_batch, gpu_result_batch, current_batch_size, batch_idx, nvtx_prefix)

                if not is_warmup:
                    results.append(result)

                # Progress reporting
                if (batch_idx + 1) % 50 == 0 or batch_end >= len(tensors):
                    progress = batch_end / len(tensors) * 100
                    print(f"  Progress: {progress:.1f}% ({batch_end}/{len(tensors)})")

    return results

def _test_h2d_only(batch_tensors, gpu_batch, current_batch_size, batch_idx, nvtx_prefix):
    """Test H2D transfers only"""
    with nvtx.range(f"{nvtx_prefix}_h2d_batch_{batch_idx}"):
        start_time = time.time()

        for i, tensor in enumerate(batch_tensors):
            with nvtx.range(f"h2d_tensor_{i}"):
                gpu_batch[i].copy_(tensor, non_blocking=True)

        torch.cuda.synchronize()
        end_time = time.time()

        total_bytes = sum(t.numel() * 4 for t in batch_tensors)
        return {
            'mode': 'h2d_only',
            'batch_idx': batch_idx,
            'batch_size': current_batch_size,
            'duration': end_time - start_time,
            'bytes': total_bytes,
            'throughput_gb_s': total_bytes / (end_time - start_time) / (1024**3)
        }

def _test_d2h_only(batch_tensors, gpu_result_batch, current_batch_size, batch_idx, nvtx_prefix):
    """Test D2H transfers only"""
    # Fill GPU memory with data first
    for i in range(current_batch_size):
        gpu_result_batch[i].fill_(float(batch_idx * 100 + i))

    with nvtx.range(f"{nvtx_prefix}_d2h_batch_{batch_idx}"):
        start_time = time.time()

        for i in range(current_batch_size):
            with nvtx.range(f"d2h_tensor_{i}"):
                batch_tensors[i].copy_(gpu_result_batch[i], non_blocking=True)

        torch.cuda.synchronize()
        end_time = time.time()

        total_bytes = current_batch_size * gpu_result_batch[0].numel() * 4
        return {
            'mode': 'd2h_only',
            'batch_idx': batch_idx,
            'batch_size': current_batch_size,
            'duration': end_time - start_time,
            'bytes': total_bytes,
            'throughput_gb_s': total_bytes / (end_time - start_time) / (1024**3)
        }

def _test_h2d_d2h_sequential(batch_tensors, gpu_batch, gpu_result_batch, current_batch_size, batch_idx, nvtx_prefix):
    """Test H2D followed by D2H sequentially"""
    start_time = time.time()

    # H2D phase
    with nvtx.range(f"{nvtx_prefix}_h2d_sequential_{batch_idx}"):
        for i, tensor in enumerate(batch_tensors):
            gpu_batch[i].copy_(tensor, non_blocking=True)
        torch.cuda.synchronize()

    h2d_end = time.time()

    # D2H phase
    with nvtx.range(f"{nvtx_prefix}_d2h_sequential_{batch_idx}"):
        for i in range(current_batch_size):
            batch_tensors[i].copy_(gpu_batch[i], non_blocking=True)
        torch.cuda.synchronize()

    end_time = time.time()

    total_bytes = sum(t.numel() * 4 for t in batch_tensors) * 2  # Both directions
    return {
        'mode': 'h2d_d2h_sequential',
        'batch_idx': batch_idx,
        'batch_size': current_batch_size,
        'duration': end_time - start_time,
        'h2d_duration': h2d_end - start_time,
        'd2h_duration': end_time - h2d_end,
        'bytes': total_bytes,
        'throughput_gb_s': total_bytes / (end_time - start_time) / (1024**3)
    }

def _test_h2d_d2h_concurrent(batch_tensors, gpu_batch, gpu_result_batch, current_batch_size, batch_idx, nvtx_prefix):
    """Test H2D and D2H with overlapping streams"""
    # Create separate streams for H2D and D2H
    h2d_stream = torch.cuda.Stream()
    d2h_stream = torch.cuda.Stream()

    start_time = time.time()

    with nvtx.range(f"{nvtx_prefix}_concurrent_{batch_idx}"):
        # Start H2D transfers
        with torch.cuda.stream(h2d_stream):
            with nvtx.range(f"h2d_concurrent_{batch_idx}"):
                for i, tensor in enumerate(batch_tensors):
                    gpu_batch[i].copy_(tensor, non_blocking=True)

        # Start D2H transfers (using previous batch data in gpu_result_batch)
        with torch.cuda.stream(d2h_stream):
            with nvtx.range(f"d2h_concurrent_{batch_idx}"):
                for i in range(current_batch_size):
                    batch_tensors[i].copy_(gpu_result_batch[i], non_blocking=True)

        # Wait for both streams
        h2d_stream.synchronize()
        d2h_stream.synchronize()

    end_time = time.time()

    total_bytes = sum(t.numel() * 4 for t in batch_tensors) * 2  # Both directions
    return {
        'mode': 'h2d_d2h_concurrent',
        'batch_idx': batch_idx,
        'batch_size': current_batch_size,
        'duration': end_time - start_time,
        'bytes': total_bytes,
        'throughput_gb_s': total_bytes / (end_time - start_time) / (1024**3)
    }

def _print_results_summary(results, test_modes):
    """Print comprehensive results summary"""
    print("\n=== Results Summary ===")

    for mode in test_modes:
        if mode not in results or not results[mode]:
            continue

        mode_results = results[mode]
        durations = [r['duration'] for r in mode_results]
        throughputs = [r['throughput_gb_s'] for r in mode_results]

        print(f"\n{mode.upper()} Statistics:")
        print(f"  Batches: {len(mode_results)}")
        print(f"  Duration (s): mean={np.mean(durations):.6f}, std={np.std(durations):.6f}")
        print(f"  Duration (s): min={np.min(durations):.6f}, max={np.max(durations):.6f}")
        print(f"  Throughput (GB/s): mean={np.mean(throughputs):.3f}, std={np.std(throughputs):.3f}")
        print(f"  Throughput (GB/s): min={np.min(throughputs):.3f}, max={np.max(throughputs):.3f}")

        # Mode-specific details
        if mode == 'h2d_d2h_sequential' and mode_results:
            h2d_durations = [r['h2d_duration'] for r in mode_results if 'h2d_duration' in r]
            d2h_durations = [r['d2h_duration'] for r in mode_results if 'd2h_duration' in r]
            if h2d_durations and d2h_durations:
                print(f"  H2D Duration (s): mean={np.mean(h2d_durations):.6f}")
                print(f"  D2H Duration (s): mean={np.mean(d2h_durations):.6f}")

def main():
    parser = argparse.ArgumentParser(description='GPU NUMA H2D/D2H Performance Test')

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

    # Test control
    parser.add_argument('--test-modes', nargs='+', 
                        choices=['h2d_only', 'd2h_only', 'h2d_d2h_sequential', 'h2d_d2h_concurrent'],
                        default=['h2d_only', 'd2h_only', 'h2d_d2h_sequential', 'h2d_d2h_concurrent'],
                        help='Test modes to run')
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

    run_h2d_d2h_test(
        gpu_id=args.gpu_id,
        tensor_shape=tuple(args.shape),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        warmup_samples=args.warmup_samples,
        memory_size_mb=args.memory_size_mb,
        skip_warmup=args.skip_warmup,
        deterministic=args.deterministic,
        nsys_report=args.nsys_report,
        pin_memory=not args.no_pin_memory,
        fill_pattern=args.fill_pattern,
        test_modes=args.test_modes
    )

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
NUMA Network Test - Data Pusher

Test script to evaluate network socket performance across NUMA nodes.
Usage with numactl:
  numactl --cpunodebind=2 --membind=2 python numa_network_test_pusher.py --nic-numa=2
  numactl --cpunodebind=0 --membind=0 python numa_network_test_pusher.py --nic-numa=2
"""

import torch
import time
import argparse
import numpy as np
from pynng import Push0
import os
import psutil

def get_numa_info():
    """Get current process NUMA binding info"""
    try:
        pid = os.getpid()
        proc = psutil.Process(pid)
        cpu_affinity = proc.cpu_affinity()
        return {
            'pid': pid,
            'cpu_affinity': cpu_affinity,
            'cpu_count': len(cpu_affinity)
        }
    except:
        return {'pid': os.getpid(), 'cpu_affinity': 'unknown', 'cpu_count': 'unknown'}

def tensor_to_bytes(tensor):
    """Convert PyTorch tensor to bytes efficiently"""
    tensor_cpu = tensor.cpu()
    tensor_np = tensor_cpu.numpy()
    return tensor_np.tobytes()

def run_network_push_test(
    address,
    tensor_shape=(1, 224, 224),
    num_samples=1000,
    batch_buffer_size=10,
    warmup_samples=100,
    memory_size_mb=100,
    nic_numa_node=None
):
    """
    Run network push test with detailed timing measurements

    Args:
        address: Network address to push to
        tensor_shape: Shape of tensors to generate
        num_samples: Total samples to push
        batch_buffer_size: Size for both batching and memory buffer
        warmup_samples: Number of samples for warmup
        memory_size_mb: Size of memory pool in MB for NUMA testing
        nic_numa_node: Expected NUMA node of the NIC (for reporting)
    """

    numa_info = get_numa_info()

    print(f"=== NUMA Network Push Test ===")
    print(f"Process PID: {numa_info['pid']}")
    print(f"CPU Affinity: {numa_info['cpu_affinity']}")
    print(f"NIC NUMA Node: {nic_numa_node}")
    print(f"Push Address: {address}")
    print(f"Tensor Shape: {tensor_shape}")
    print(f"Total Samples: {num_samples}")
    print(f"Batch-Buffer Size: {batch_buffer_size}")
    print(f"Warmup Samples: {warmup_samples}")
    print(f"Memory Pool Size: {memory_size_mb} MB")
    print("=" * 50)

    with Push0(listen=address) as sock:
        print(f"Listening at {address}, waiting for puller to connect...")

        # Allocate memory on current NUMA node to test memory locality
        print(f"Allocating {memory_size_mb} MB memory pool on current NUMA node...")
        memory_pool = torch.randn(memory_size_mb * 1024 * 1024 // 4)  # 4 bytes per float32
        print(f"Memory pool allocated: {memory_pool.element_size() * memory_pool.nelement() / 1024 / 1024:.2f} MB")


        # Calculate pre-allocation memory requirements
        total_samples = warmup_samples + num_samples
        tensor_size_bytes = np.prod(tensor_shape) * 4  # float32 = 4 bytes
        estimated_total_mb = (total_samples * tensor_size_bytes) / (1024 * 1024)

        # Pre-generate tensors to minimize generation overhead during test
        print("Pre-generating test data...")
        print(f"Estimated memory for {total_samples} tensors: {estimated_total_mb:.1f} MB")

        # Warning for large allocations
        if estimated_total_mb > 1000:  # 1GB threshold
            print(f"WARNING: Large memory allocation ({estimated_total_mb:.1f} MB) may cause memory pressure!")
            print(f"Consider reducing --num-samples or tensor size if system has limited RAM")

        test_tensors = []
        total_samples = warmup_samples + num_samples

        for i in range(total_samples):
            tensor = torch.randn(*tensor_shape)

            # Touch memory pool to ensure it's in local cache
            if i % 10 == 0:
                _ = memory_pool[:(1024*1024)].sum()  # Touch 4MB of the pool

            tensor_bytes = tensor_to_bytes(tensor)

            metadata = {
                'index': i,
                'shape': tensor_shape,
                'is_warmup': i < warmup_samples,
                'numa_info': numa_info,
                'nic_numa': nic_numa_node
            }
            metadata_bytes = str(metadata).encode('utf-8')
            test_tensors.append(metadata_bytes + b'\n' + tensor_bytes)

        print(f"Generated {len(test_tensors)} tensors")

        # Warmup phase
        print("Starting warmup phase...")
        warmup_start = time.time()
        for i in range(warmup_samples):
            sock.send(test_tensors[i])
        warmup_time = time.time() - warmup_start
        print(f"Warmup completed: {warmup_samples} samples in {warmup_time:.3f}s")

        # Main test phase
        print("Starting main test phase...")
        batch_times = []
        total_bytes = 0
        memory_access_times = []

        test_start = time.time()

        for batch_start in range(warmup_samples, total_samples, batch_buffer_size):
            batch_end = min(batch_start + batch_buffer_size, total_samples)
            current_batch_size = batch_end - batch_start

            batch_start_time = time.time()

            for i in range(batch_start, batch_end):
                # Memory access test
                mem_start = time.time()
                memory_sum = memory_pool[:(512*1024)].sum()  # Touch 2MB
                mem_time = time.time() - mem_start
                memory_access_times.append(mem_time)

                sock.send(test_tensors[i])
                total_bytes += len(test_tensors[i])

            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)

            if len(batch_times) % 10 == 0:
                avg_batch_time = np.mean(batch_times[-10:])
                avg_mem_time = np.mean(memory_access_times[-10*current_batch_size:])
                throughput = (current_batch_size * len(test_tensors[0])) / avg_batch_time / (1024*1024)
                print(f"Batch {len(batch_times)}: {avg_batch_time:.4f}s, "
                      f"{throughput:.2f} MB/s, mem_access: {avg_mem_time*1000:.3f}ms")

        total_test_time = time.time() - test_start

        # Results
        print("\n=== Test Results ===")
        print(f"Total test time: {total_test_time:.3f}s")
        print(f"Samples pushed: {num_samples}")
        print(f"Total bytes: {total_bytes / (1024*1024):.2f} MB")
        print(f"Average throughput: {total_bytes / total_test_time / (1024*1024):.2f} MB/s")
        print(f"Average sample rate: {num_samples / total_test_time:.2f} samples/s")

        if batch_times:
            print(f"Batch statistics:")
            print(f"  Mean batch time: {np.mean(batch_times):.4f}s")
            print(f"  Std batch time: {np.std(batch_times):.4f}s")
            print(f"  Min batch time: {np.min(batch_times):.4f}s")
            print(f"  Max batch time: {np.max(batch_times):.4f}s")

        if memory_access_times:
            print(f"Memory access statistics:")
            print(f"  Mean access time: {np.mean(memory_access_times)*1000:.4f}ms")
            print(f"  Std access time: {np.std(memory_access_times)*1000:.4f}ms")
            print(f"  Min access time: {np.min(memory_access_times)*1000:.4f}ms")
            print(f"  Max access time: {np.max(memory_access_times)*1000:.4f}ms")

        print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description='NUMA Network Push Test')
    parser.add_argument('--address', default='tcp://0.0.0.0:5555',
                        help='Network address to push to')
    parser.add_argument('--shape', nargs=3, type=int, default=[1, 224, 224],
                        help='Tensor shape (C H W)')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples to push')
    parser.add_argument('--batch-buffer-size', type=int, default=10,
                        help='Size for both batching and memory buffer')
    parser.add_argument('--warmup-samples', type=int, default=100,
                        help='Number of warmup samples')
    parser.add_argument('--memory-size-mb', type=int, default=100,
                        help='Size of memory pool in MB')
    parser.add_argument('--nic-numa', type=int, default=None,
                        help='NUMA node where the NIC is located')

    args = parser.parse_args()

    run_network_push_test(
        args.address,
        tuple(args.shape),
        args.num_samples,
        args.batch_buffer_size,
        args.warmup_samples,
        args.memory_size_mb,
        args.nic_numa
    )

if __name__ == '__main__':
    main()

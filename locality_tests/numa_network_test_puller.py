#!/usr/bin/env python3
"""
NUMA Network Test - Data Puller

Test script to evaluate network socket performance across NUMA nodes.
Usage with numactl:
  numactl --cpunodebind=2 --membind=2 python numa_network_test_puller.py --nic-numa=2
  numactl --cpunodebind=0 --membind=0 python numa_network_test_puller.py --nic-numa=2
"""

import time
import argparse
import numpy as np
import torch
import os
import psutil
from pynng import Pull0, Timeout

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

def bytes_to_tensor(data, shape):
    """Convert bytes to PyTorch tensor with known shape"""
    newline_index = data.find(b'\n')
    if newline_index != -1:
        tensor_data = data[newline_index + 1:]
        tensor_np = np.frombuffer(tensor_data, dtype=np.float32).reshape(shape)
        return torch.from_numpy(tensor_np)
    return None

def run_network_pull_test(
    address,
    expected_samples=1000,
    timeout_ms=5000,
    nic_numa_node=None
):
    """
    Run network pull test with detailed timing measurements
    """

    numa_info = get_numa_info()

    print(f"=== NUMA Network Pull Test ===")
    print(f"Process PID: {numa_info['pid']}")
    print(f"CPU Affinity: {numa_info['cpu_affinity']}")
    print(f"NIC NUMA Node: {nic_numa_node}")
    print(f"Pull Address: {address}")
    print(f"Expected Samples: {expected_samples}")
    print(f"Timeout: {timeout_ms}ms")
    print("=" * 50)

    with Pull0(dial=address) as sock:
        sock.recv_timeout = timeout_ms
        print(f"Connected to {address}")

        # Statistics tracking
        received_count = 0
        warmup_count = 0
        test_count = 0
        total_bytes = 0
        test_bytes = 0

        # Timing
        first_recv_time = None
        test_start_time = None
        receive_times = []

        # Batch timing (for consistent measurement)
        batch_size = 10
        batch_times = []
        batch_start_time = None
        batch_count = 0

        print("Starting to receive data...")

        while True:
            try:
                recv_start = time.time()
                data = sock.recv()
                recv_time = time.time() - recv_start

                if first_recv_time is None:
                    first_recv_time = time.time()

                # Parse metadata
                newline_index = data.find(b'\n')
                if newline_index != -1:
                    metadata_str = data[:newline_index].decode('utf-8')
                    try:
                        metadata = eval(metadata_str)
                        is_warmup = metadata.get('is_warmup', False)
                        sample_index = metadata.get('index', received_count)
                        tensor_shape = metadata.get('shape', (3, 224, 224))

                        received_count += 1
                        total_bytes += len(data)
                        receive_times.append(recv_time)

                        if is_warmup:
                            warmup_count += 1
                        else:
                            if test_start_time is None:
                                test_start_time = time.time()
                                batch_start_time = test_start_time
                                print("Test phase started")

                            test_count += 1
                            test_bytes += len(data)

                            # Batch timing
                            if batch_start_time is not None:
                                batch_count += 1
                                if batch_count >= batch_size:
                                    batch_time = time.time() - batch_start_time
                                    batch_times.append(batch_time)

                                    if len(batch_times) % 10 == 0:
                                        avg_batch_time = np.mean(batch_times[-10:])
                                        throughput = (batch_size * len(data)) / avg_batch_time / (1024*1024)
                                        print(f"Batch {len(batch_times)}: {avg_batch_time:.4f}s, {throughput:.2f} MB/s")

                                    batch_start_time = time.time()
                                    batch_count = 0

                        # Verify tensor reconstruction (occasionally)
                        if received_count <= 5 or received_count % 200 == 0:
                            tensor = bytes_to_tensor(data, tensor_shape)
                            if tensor is not None:
                                print(f"Sample {sample_index}: shape={tensor.shape}, "
                                      f"min/max/mean={tensor.min():.3f}/{tensor.max():.3f}/{tensor.mean():.3f}")

                        # Check if we've received expected test samples
                        if test_count >= expected_samples:
                            print(f"Received all {expected_samples} test samples")
                            break

                    except Exception as e:
                        print(f"Error parsing metadata: {e}")
                        continue

            except Timeout:
                print("Timeout waiting for data")
                break
            except KeyboardInterrupt:
                print("Interrupted by user")
                break
            except Exception as e:
                print(f"Error receiving data: {e}")
                break

        # Final results
        current_time = time.time()

        print("\n=== Test Results ===")
        print(f"Total samples received: {received_count}")
        print(f"Warmup samples: {warmup_count}")
        print(f"Test samples: {test_count}")
        print(f"Total bytes: {total_bytes / (1024*1024):.2f} MB")
        print(f"Test bytes: {test_bytes / (1024*1024):.2f} MB")

        if test_start_time and test_count > 0:
            test_duration = current_time - test_start_time
            print(f"Test duration: {test_duration:.3f}s")
            print(f"Test throughput: {test_bytes / test_duration / (1024*1024):.2f} MB/s")
            print(f"Test sample rate: {test_count / test_duration:.2f} samples/s")

        if receive_times:
            print(f"Receive time statistics:")
            print(f"  Mean: {np.mean(receive_times):.6f}s")
            print(f"  Std: {np.std(receive_times):.6f}s")
            print(f"  Min: {np.min(receive_times):.6f}s")
            print(f"  Max: {np.max(receive_times):.6f}s")

        if batch_times:
            print(f"Batch time statistics:")
            print(f"  Mean: {np.mean(batch_times):.4f}s")
            print(f"  Std: {np.std(batch_times):.4f}s")
            print(f"  Min: {np.min(batch_times):.4f}s")
            print(f"  Max: {np.max(batch_times):.4f}s")

        print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description='NUMA Network Pull Test')
    parser.add_argument('--address', default='tcp://127.0.0.1:5555',
                        help='Network address to pull from')
    parser.add_argument('--expected-samples', type=int, default=1000,
                        help='Expected number of test samples')
    parser.add_argument('--timeout', type=int, default=5000,
                        help='Socket timeout in milliseconds')
    parser.add_argument('--nic-numa', type=int, default=None,
                        help='NUMA node where the NIC is located')

    args = parser.parse_args()

    run_network_pull_test(
        args.address,
        args.expected_samples,
        args.timeout,
        args.nic_numa
    )

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
NUMA IPC Test - Data Puller

Test script to evaluate IPC socket performance across NUMA nodes.
Usage with numactl:
  numactl --cpunodebind=2 --membind=2 python numa_ipc_test_puller.py
  numactl --cpunodebind=3 --membind=3 python numa_ipc_test_puller.py
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

def run_ipc_pull_test(
    ipc_path,
    expected_samples=1000,
    timeout_ms=10000,
    batch_buffer_size=10,
    memory_size_mb=100,
    process_data=True
):
    """
    Run IPC pull test with memory allocation to test NUMA effects

    Args:
        ipc_path: IPC socket path
        expected_samples: Expected number of test samples
        timeout_ms: Socket timeout in milliseconds
        batch_buffer_size: Size for both batching and processing buffer
        memory_size_mb: Size of memory pool in MB for NUMA testing
        process_data: Whether to process received tensors
    """

    numa_info = get_numa_info()

    print(f"=== NUMA IPC Pull Test ===")
    print(f"Process PID: {numa_info['pid']}")
    print(f"CPU Affinity: {numa_info['cpu_affinity']}")
    print(f"IPC Path: {ipc_path}")
    print(f"Expected Samples: {expected_samples}")
    print(f"Timeout: {timeout_ms}ms")
    print(f"Batch-Buffer Size: {batch_buffer_size}")
    print(f"Memory Pool Size: {memory_size_mb} MB")
    print(f"Process Data: {process_data}")
    print("=" * 50)

    # Allocate memory on current NUMA node to test memory locality
    print(f"Allocating {memory_size_mb} MB memory pool on current NUMA node...")
    memory_pool = torch.randn(memory_size_mb * 1024 * 1024 // 4)  # 4 bytes per float32
    processing_buffer = torch.zeros(batch_buffer_size, 1, 224, 224)  # Will be resized when we know actual shape
    print(f"Memory pool allocated: {memory_pool.element_size() * memory_pool.nelement() / 1024 / 1024:.2f} MB")

    ipc_address = f"ipc://{ipc_path}"

    with Pull0(dial=ipc_address) as sock:
        sock.recv_timeout = timeout_ms
        print(f"Connected to {ipc_address}")

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
        processing_times = []
        memory_access_times = []

        # Batch timing
        batch_times = []
        batch_start_time = None
        batch_count = 0
        batch_bytes = 0
        processing_buffer_initialized = False

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
                        tensor_shape = metadata.get('shape', (1, 224, 224))

                        # Initialize processing buffer with correct shape on first real tensor
                        if not processing_buffer_initialized and not is_warmup:
                            processing_buffer = torch.zeros(batch_buffer_size, *tensor_shape)
                            processing_buffer_initialized = True
                            print(f"Processing buffer initialized: {processing_buffer.shape}")

                        # Memory access test
                        mem_start = time.time()
                        memory_sum = memory_pool[:(512*1024)].sum()  # Touch 2MB
                        mem_time = time.time() - mem_start
                        memory_access_times.append(mem_time)

                        # Data processing test
                        if process_data and processing_buffer_initialized:
                            proc_start = time.time()
                            tensor = bytes_to_tensor(data, tensor_shape)
                            if tensor is not None:
                                # Simulate processing: copy to buffer and do some computation
                                buffer_idx = received_count % processing_buffer.shape[0]
                                processing_buffer[buffer_idx] = tensor
                                # Simple computation to force memory access
                                result = processing_buffer[buffer_idx].mean()
                            proc_time = time.time() - proc_start
                            processing_times.append(proc_time)

                        received_count += 1
                        total_bytes += len(data)
                        receive_times.append(recv_time)

                        if is_warmup:
                            warmup_count += 1
                        else:
                            if test_start_time is None:
                                test_start_time = time.time()
                                batch_start_time = test_start_time
                                batch_bytes = 0

                            test_count += 1
                            test_bytes += len(data)

                            # Batch timing
                            if batch_start_time is not None:
                                batch_count += 1
                                batch_bytes += len(data)
                                if batch_count >= batch_buffer_size:
                                    batch_time = time.time() - batch_start_time
                                    batch_times.append(batch_time)

                                    if len(batch_times) % 10 == 0:
                                        avg_batch_time = np.mean(batch_times[-10:])
                                        avg_mem_time = np.mean(memory_access_times[-10*batch_buffer_size:])
                                        avg_proc_time = np.mean(processing_times[-10*batch_buffer_size:]) if processing_times else 0

                                        # NOTE: Throughput calculation assumes all messages are the same size
                                        throughput = batch_bytes / avg_batch_time / (1024*1024)
                                        print(f"Batch {len(batch_times)}: {avg_batch_time:.4f}s, "
                                              f"{throughput:.2f} MB/s, mem: {avg_mem_time*1000:.3f}ms, "
                                              f"proc: {avg_proc_time*1000:.3f}ms")

                                    batch_start_time = time.time()
                                    batch_count = 0
                                    batch_bytes = 0

                        # Verify tensor reconstruction (occasionally)
                        if received_count <= 5 or received_count % 200 == 0:
                            if process_data:
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

        if memory_access_times:
            print(f"Memory access statistics:")
            print(f"  Mean: {np.mean(memory_access_times)*1000:.4f}ms")
            print(f"  Std: {np.std(memory_access_times)*1000:.4f}ms")
            print(f"  Min: {np.min(memory_access_times)*1000:.4f}ms")
            print(f"  Max: {np.max(memory_access_times)*1000:.4f}ms")

        if processing_times and process_data:
            print(f"Processing time statistics:")
            print(f"  Mean: {np.mean(processing_times)*1000:.4f}ms")
            print(f"  Std: {np.std(processing_times)*1000:.4f}ms")
            print(f"  Min: {np.min(processing_times)*1000:.4f}ms")
            print(f"  Max: {np.max(processing_times)*1000:.4f}ms")

        if batch_times:
            print(f"Batch time statistics:")
            print(f"  Mean: {np.mean(batch_times):.4f}s")
            print(f"  Std: {np.std(batch_times):.4f}s")
            print(f"  Min: {np.min(batch_times):.4f}s")
            print(f"  Max: {np.max(batch_times):.4f}s")

        print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description='NUMA IPC Pull Test')
    parser.add_argument('--ipc-path', default='/tmp/numa_ipc_test',
                        help='IPC socket path')
    parser.add_argument('--expected-samples', type=int, default=1000,
                        help='Expected number of test samples')
    parser.add_argument('--timeout', type=int, default=10000,
                        help='Socket timeout in milliseconds')
    parser.add_argument('--batch-buffer-size', type=int, default=10,
                        help='Size for both batching and processing buffer')
    parser.add_argument('--memory-size-mb', type=int, default=100,
                        help='Size of memory pool in MB')
    parser.add_argument('--no-process', action='store_true',
                        help='Skip tensor processing (just receive)')

    args = parser.parse_args()

    run_ipc_pull_test(
        args.ipc_path,
        args.expected_samples,
        args.timeout,
        args.batch_buffer_size,
        args.memory_size_mb,
        not args.no_process
    )

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import subprocess
import re
import os
import sys
import json
from pathlib import Path

def get_numa_topology():
    """Get NUMA topology information."""
    numa_info = {}
    try:
        output = subprocess.check_output(['numactl', '--hardware'], universal_newlines=True)

        # Parse number of nodes
        node_match = re.search(r'available: (\d+) nodes', output)
        if node_match:
            numa_info['node_count'] = int(node_match.group(1))

        # Parse CPU assignment and memory for each node
        for node in range(numa_info['node_count']):
            cpu_match = re.search(r'node ' + str(node) + r' cpus: ([\d\s]+)', output)
            if cpu_match:
                numa_info[f'node{node}_cpus'] = [int(cpu) for cpu in cpu_match.group(1).split()]

            mem_match = re.search(r'node ' + str(node) + r' size: (\d+) MB', output)
            if mem_match:
                numa_info[f'node{node}_memory'] = int(mem_match.group(1))

        return numa_info
    except subprocess.CalledProcessError:
        print("Error: Failed to get NUMA topology. Is numactl installed?")
        sys.exit(1)

def get_gpu_topology():
    """Get GPU topology information and NUMA affinity."""
    gpu_info = {}
    try:
        # Get number of GPUs - use nvidia-smi -L instead which is more reliable
        gpu_list_output = subprocess.check_output(['nvidia-smi', '-L'], universal_newlines=True)
        gpu_count = len(gpu_list_output.strip().split('\n'))
        gpu_info['gpu_count'] = gpu_count

        # Get PCIe bus ID and NUMA node for each GPU
        for gpu in range(gpu_info['gpu_count']):
            bus_id = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=pci.bus_id', '--format=csv,noheader'], universal_newlines=True).strip()
            gpu_info[f'gpu{gpu}_bus_id'] = bus_id

            # Clean up bus ID format for lspci
            clean_bus_id = bus_id.split(':')[-2] + ':' + bus_id.split(':')[-1]

            # Get NUMA node for this GPU using lspci
            try:
                numa_output = subprocess.check_output(['lspci', '-vv', '-s', clean_bus_id], universal_newlines=True)
                numa_match = re.search(r'NUMA node: (\d+)', numa_output)
                if numa_match:
                    gpu_info[f'gpu{gpu}_numa_node'] = int(numa_match.group(1))
                else:
                    gpu_info[f'gpu{gpu}_numa_node'] = 0  # Default to node 0 if not found
            except subprocess.CalledProcessError:
                # Fall back to nvidia-smi topo if available in newer drivers
                try:
                    topo_output = subprocess.check_output(['nvidia-smi', 'topo', '-n', str(gpu)], universal_newlines=True)
                    numa_match = re.search(r'NUMA node (\d+)', topo_output)
                    if numa_match:
                        gpu_info[f'gpu{gpu}_numa_node'] = int(numa_match.group(1))
                    else:
                        gpu_info[f'gpu{gpu}_numa_node'] = 0
                except subprocess.CalledProcessError:
                    gpu_info[f'gpu{gpu}_numa_node'] = 0

            # Get GPU name
            gpu_name = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=name', '--format=csv,noheader'], universal_newlines=True).strip()
            gpu_info[f'gpu{gpu}_name'] = gpu_name

        return gpu_info
    except subprocess.CalledProcessError:
        print("Warning: No NVIDIA GPUs detected or nvidia-smi not available.")
        return {'gpu_count': 0}

def determine_optimal_binding(numa_info, gpu_info, target_gpus=None):
    """Determine optimal NUMA binding based on GPU usage."""
    if gpu_info['gpu_count'] == 0:
        # No GPUs, use all NUMA nodes with interleaving
        return {
            'cpunodebind': None,
            'membind': None,
            'interleave': list(range(numa_info['node_count'])),
            'strategy': 'cpu-only-interleave'
        }

    # Parse target GPUs
    if target_gpus is None:
        # Use all GPUs
        target_gpus = list(range(gpu_info['gpu_count']))
    elif isinstance(target_gpus, str):
        # Parse CUDA_VISIBLE_DEVICES format (e.g., "0,1,3")
        target_gpus = [int(g) for g in target_gpus.split(',') if g.strip()]

    # Collect NUMA nodes for target GPUs
    gpu_numa_nodes = {}
    for gpu in target_gpus:
        if gpu < gpu_info['gpu_count']:
            node = gpu_info[f'gpu{gpu}_numa_node']
            if node not in gpu_numa_nodes:
                gpu_numa_nodes[node] = []
            gpu_numa_nodes[node].append(gpu)

    if not gpu_numa_nodes:
        # No valid GPUs specified
        return {
            'cpunodebind': None,
            'membind': None,
            'interleave': list(range(numa_info['node_count'])),
            'strategy': 'fallback-interleave'
        }

    # Single NUMA node case (ideal)
    if len(gpu_numa_nodes) == 1:
        node = list(gpu_numa_nodes.keys())[0]
        return {
            'cpunodebind': node,
            'membind': node,
            'interleave': None,
            'strategy': 'single-node-bind'
        }

    # Multiple NUMA nodes case
    # Find node with most GPUs
    primary_node = max(gpu_numa_nodes.keys(), key=lambda node: len(gpu_numa_nodes[node]))

    # If we have a dominant node (more GPUs than others), prefer it
    if len(gpu_numa_nodes[primary_node]) > sum(len(gpus) for node, gpus in gpu_numa_nodes.items() if node != primary_node):
        return {
            'cpunodebind': None,  # Allow threads on all nodes
            'membind': None,
            'preferred': primary_node,
            'strategy': 'dominant-node-preferred'
        }
    else:
        # Balanced distribution across nodes - interleave memory
        nodes = list(gpu_numa_nodes.keys())
        return {
            'cpunodebind': None,
            'membind': None,
            'interleave': nodes,
            'strategy': 'multi-node-interleave'
        }

def generate_numactl_command(binding):
    """Generate the numactl command string based on binding."""
    cmd_parts = ["numactl"]

    if binding.get('cpunodebind') is not None:
        cmd_parts.extend(["--cpunodebind", str(binding['cpunodebind'])])

    if binding.get('membind') is not None:
        cmd_parts.extend(["--membind", str(binding['membind'])])

    if binding.get('preferred') is not None:
        cmd_parts.extend(["--preferred", str(binding['preferred'])])

    if binding.get('interleave') is not None:
        cmd_parts.extend(["--interleave", ','.join(str(n) for n in binding['interleave'])])

    return ' '.join(cmd_parts)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Determine optimal NUMA binding for PyTorch with GPUs')
    parser.add_argument('--gpus', type=str, help='Target GPUs (comma-separated, e.g., "0,1,3")')
    parser.add_argument('--strategy', type=str, choices=['single', 'interleave', 'preferred', 'auto'], 
                      default='auto', help='Binding strategy')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Check for CUDA_VISIBLE_DEVICES in environment
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    target_gpus = args.gpus or cuda_devices

    # Get topology information
    numa_info = get_numa_topology()
    gpu_info = get_gpu_topology()

    # Determine optimal binding
    binding = determine_optimal_binding(numa_info, gpu_info, target_gpus)

    # Override with user-specified strategy if provided
    if args.strategy != 'auto':
        if args.strategy == 'single' and numa_info['node_count'] > 0:
            # Bind to primary GPU node or node 0
            primary_node = 0
            if gpu_info['gpu_count'] > 0 and target_gpus:
                first_gpu = int(target_gpus.split(',')[0]) if isinstance(target_gpus, str) else target_gpus[0]
                if first_gpu < gpu_info['gpu_count']:
                    primary_node = gpu_info[f'gpu{first_gpu}_numa_node']

            binding = {
                'cpunodebind': primary_node,
                'membind': primary_node,
                'interleave': None,
                'strategy': 'user-specified-single'
            }
        elif args.strategy == 'interleave':
            binding = {
                'cpunodebind': None,
                'membind': None,
                'interleave': list(range(numa_info['node_count'])),
                'strategy': 'user-specified-interleave'
            }
        elif args.strategy == 'preferred' and gpu_info['gpu_count'] > 0 and target_gpus:
            # Prefer the node of the first GPU
            first_gpu = int(target_gpus.split(',')[0]) if isinstance(target_gpus, str) else target_gpus[0]
            if first_gpu < gpu_info['gpu_count']:
                primary_node = gpu_info[f'gpu{first_gpu}_numa_node']
                binding = {
                    'cpunodebind': None,
                    'membind': None,
                    'preferred': primary_node,
                    'strategy': 'user-specified-preferred'
                }

    # Print system information
    print("\n=== System NUMA Topology ===")
    print(f"NUMA Nodes: {numa_info['node_count']}")
    for node in range(numa_info['node_count']):
        print(f"Node {node}: {len(numa_info[f'node{node}_cpus'])} CPUs, {numa_info[f'node{node}_memory']/1024:.1f} GB memory")

    if gpu_info['gpu_count'] > 0:
        print("\n=== GPU NUMA Affinity ===")
        for gpu in range(gpu_info['gpu_count']):
            print(f"GPU {gpu} ({gpu_info[f'gpu{gpu}_name']}): NUMA Node {gpu_info[f'gpu{gpu}_numa_node']}")

    print("\n=== Binding Decision ===")
    print(f"Strategy: {binding['strategy']}")

    if binding.get('cpunodebind') is not None:
        print(f"CPU Binding: Node {binding['cpunodebind']}")
    else:
        print("CPU Binding: None (using all nodes)")

    if binding.get('membind') is not None:
        print(f"Memory Binding: Node {binding['membind']}")
    elif binding.get('preferred') is not None:
        print(f"Memory Preference: Node {binding['preferred']}")
    elif binding.get('interleave') is not None:
        print(f"Memory Interleaving: Nodes {', '.join(str(n) for n in binding['interleave'])}")
    else:
        print("Memory Binding: None (using all nodes)")

    # Generate and print the numactl command
    numactl_cmd = generate_numactl_command(binding)
    if numactl_cmd == "numactl":
        print("\nNo NUMA binding needed. Run your application normally.")
    else:
        print("\nRecommended command prefix:")
        print(f"{numactl_cmd} your_command")

        # Print examples
        print("\nExamples:")
        print(f"  {numactl_cmd} python train.py")
        print(f"  {numactl_cmd} python -m torch.distributed.launch --nproc_per_node=X train.py")

if __name__ == "__main__":
    main()

# NUMA Awareness in HPC Environments (Slurm & MPI)

## What is NUMA?

Non-Uniform Memory Access (NUMA) is a computer memory design where memory access time depends on the memory location relative to the processor. In multi-socket systems or systems with multiple GPUs, data transfer between CPUs/GPUs and memory can have different latencies and bandwidths depending on which NUMA node they reside in.

**Why it matters**: Proper NUMA binding can improve performance by 10-40% for memory or I/O intensive workloads, especially those with CPU-GPU transfers.

## Slurm NUMA Awareness

Slurm has built-in support for NUMA topology detection and binding:

### Key Slurm NUMA Options

| Option | Description | Example |
|--------|-------------|---------|
| `--cpu-bind` | Control how tasks are bound to CPUs | `--cpu-bind=cores` |
| `--mem-bind` | Control memory binding policy | `--mem-bind=local` |
| `--distribution` | Control task distribution | `--distribution=block:block` |
| `--gpus-per-task` | Allocate GPUs per task | `--gpus-per-task=1` |
| `--gpu-bind` | Control GPU binding policy | `--gpu-bind=closest` |

### Common Slurm NUMA Commands

```bash
# Basic NUMA-aware CPU binding
srun --cpu-bind=cores ./myprogram

# NUMA-aware memory binding
srun --cpu-bind=cores --mem-bind=local ./myprogram

# GPU with NUMA awareness
srun --cpu-bind=cores --mem-bind=local --gpus-per-task=1 ./myprogram

# Comprehensive NUMA awareness
srun --cpu-bind=cores --mem-bind=local --distribution=block:block --gpus-per-task=1 --gpu-bind=closest ./myprogram
MPI NUMA AwarenessAll major MPI implementations provide NUMA-aware process binding:OpenMPI# Basic NUMA binding
mpirun --bind-to numa ./myprogram

# Map processes by NUMA domain
mpirun --map-by numa ./myprogram

# Combined mapping and binding
mpirun --map-by numa --bind-to numa ./myprogram

# Specify threads per NUMA domain
mpirun --map-by numa:pe=4 ./myprogram
Intel MPI# Enable process pinning
export I_MPI_PIN=1

# Pin to NUMA domains
export I_MPI_PIN_DOMAIN=numa
mpirun -n 8 ./myprogram

# Fine-grained control
export I_MPI_PIN_PROCESSOR_LIST=0,2,4,6
mpirun -n 4 ./myprogram
MPICH# Bind to NUMA domains
mpirun -bind-to numa ./myprogram

# Specify number of cores per rank
mpirun -bind-to numa:4 ./myprogram
Combining Slurm and MPIWhen using MPI applications on Slurm-managed clusters, there are several approaches:Let Slurm Handle Binding (Recommended)# Slurm handles binding, MPI respects it
srun --cpu-bind=cores mpirun --bind-to none ./myprogram
Let MPI Handle Binding# Slurm allocates resources, MPI handles binding
salloc -N 2 -n 8
mpirun --bind-to numa ./myprogram
GPU-Accelerated MPI Applications# Slurm allocates GPUs and handles binding
srun --cpu-bind=cores --gpus-per-task=1 --gpu-bind=closest mpirun --bind-to none ./myprogram
Best Practices

Check your topology first:
numactl --hardware
nvidia-smi topo -m  # For NVIDIA GPUs



For CPU-only workloads:

Bind tasks to cores or NUMA domains
Use --mem-bind=local to ensure memory locality



For GPU workloads:

Ensure CPU threads are bound to the same NUMA node as the GPU
Use --gpu-bind=closest in Slurm



For hybrid MPI+OpenMP:

Bind MPI ranks to NUMA domains
Let OpenMP threads stay within NUMA domains



Verify your binding:
# Add this to your job script
numastat -n
hwloc-ps


Example Job ScriptsBasic Slurm NUMA-Aware Job#!/bin/bash
#SBATCH --job-name=numa_test
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-bind=local

srun --cpu-bind=cores ./myprogram
GPU-Accelerated Job#!/bin/bash
#SBATCH --job-name=gpu_numa
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest

srun --cpu-bind=cores ./my_gpu_program
MPI Job with NUMA Awareness#!/bin/bash
#SBATCH --job-name=mpi_numa
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4

# Let Slurm handle binding
srun --cpu-bind=cores --distribution=block:block mpirun --bind-to none ./mpi_program
```

## Example - AMD EPYC 7713 64-Core Processor

```
$ numactl --hardware
available: 8 nodes (0-7)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
node 0 size: 63965 MB
node 0 free: 19875 MB
node 1 cpus: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
node 1 size: 64507 MB
node 1 free: 21081 MB
node 2 cpus: 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
node 2 size: 64507 MB
node 2 free: 6504 MB
node 3 cpus: 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
node 3 size: 64495 MB
node 3 free: 4817 MB
node 4 cpus: 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79
node 4 size: 64507 MB
node 4 free: 43357 MB
node 5 cpus: 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
node 5 size: 64465 MB
node 5 free: 10066 MB
node 6 cpus: 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111
node 6 size: 64507 MB
node 6 free: 12027 MB
node 7 cpus: 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
node 7 size: 64506 MB
node 7 free: 38407 MB
node distances:
node   0   1   2   3   4   5   6   7
  0:  10  12  12  12  32  32  32  32
  1:  12  10  12  12  32  32  32  32
  2:  12  12  10  12  32  32  32  32
  3:  12  12  12  10  32  32  32  32
  4:  32  32  32  32  10  12  12  12
  5:  32  32  32  32  12  10  12  12
  6:  32  32  32  32  12  12  10  12
  7:  32  32  32  32  12  12  12  10
```

It looks like 

```
+------------------------------------------------------------------------------------+
|                                                                                    |
|         DUAL-SOCKET AMD EPYC 7713 NUMA ARCHITECTURE (8 NUMA NODES)                 |
|                                                                                    |
+--------------------------------------+      +--------------------------------------+
|            SOCKET 0                  |      |            SOCKET 1                  |
|  +-------------+    +-------------+  |      |  +-------------+    +-------------+  |
|  | NUMA NODE 0 |    | NUMA NODE 1 |  |      |  | NUMA NODE 4 |    | NUMA NODE 5 |  |
|  |             |<-->|             |  |      |  |             |<-->|             |  |
|  | CPUs: 0-15  |d=12| CPUs: 16-31 |  |      |  | CPUs: 64-79 |d=12| CPUs: 80-95 |  |
|  | ~64GB RAM   |    | ~64GB RAM   |  |      |  | ~64GB RAM   |    | ~64GB RAM   |  |
|  | 2 CCDs      |    | 2 CCDs      |  |      |  | 2 CCDs      |    | 2 CCDs      |  |
|  +------^------+    +------^------+  |      |  +------^------+    +------^------+  |
|         |                 |          |      |         |                 |          |
|      d=12              d=12          |      |      d=12              d=12          |
|         |                 |          |      |         |                 |          |
|  +------v------+    +------v------+  |      |  +------v------+    +------v------+  |
|  | NUMA NODE 2 |    | NUMA NODE 3 |  |      |  | NUMA NODE 6 |    | NUMA NODE 7 |  |
|  |             |<-->|             |  |      |  |             |<-->|             |  |
|  | CPUs: 32-47 |d=12| CPUs: 48-63 |  |      |  | CPUs: 96-111|d=12| CPUs:112-127|  |
|  | ~64GB RAM   |    | ~64GB RAM   |  |      |  | ~64GB RAM   |    | ~64GB RAM   |  |
|  | 2 CCDs      |    | 2 CCDs      |  |      |  | 2 CCDs      |    | 2 CCDs      |  |
|  +-------------+    +-------------+  |      |  +-------------+    +-------------+  |
|                                      |      |                                      |
+---------------||---------------------+      +---------------||---------------------+
                ||                                            ||
                ||============= d=32 (Infinity Fabric) =======||
                ||                                            ||
+-----------------------------------------------------------------------------------+
|                                                                                   |
|                               NUMA DISTANCE LEGEND                                |
|                                                                                   |
|  • d=10: Memory access within same NUMA node (baseline local access)              |
|  • d=12: Memory access between NUMA nodes in same socket (~20% higher latency)    |
|  • d=32: Memory access between sockets (~220% higher latency)                     |
|                                                                                   |
|  NOTE: Each socket has 64 cores organized as 4 NUMA domains with 16 cores each    |
|        Each NUMA domain has 2 CCDs (Core Complex Dies) with 8 cores per CCD       |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

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


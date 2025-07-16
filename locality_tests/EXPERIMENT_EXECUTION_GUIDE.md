# GPU-NUMA Locality Experiments - Execution Guide

This guide provides step-by-step instructions for running GPU-NUMA locality experiments using the unified Hydra-based framework.

## 🚀 Quick Start

### Prerequisites
- NVIDIA GPUs with CUDA support
- Python environment with PyTorch, Hydra, pandas
- `nsys` profiling tools installed
- `numactl` for NUMA binding

### Basic Execution
```bash
# Run the core NUMA locality study (48 experiments)
python hydra_experiment_runner.py experiment=numa_study

# Quick framework validation (12 experiments) 
python hydra_experiment_runner.py experiment=quick_test

# Model scaling analysis (36 experiments)
python hydra_experiment_runner.py experiment=scaling_study
```

## 📊 Experiment Configurations

### 1. NUMA Locality Study (`numa_study`)
**Purpose**: Study NUMA locality effects across different ViT model sizes and GPU-NUMA combinations

**Configuration**: 4 GPUs × 2 NUMA nodes × 6 models = **48 experiments**
- **GPUs**: 0, 1, 2, 3
- **NUMA Nodes**: 0 (local), 2 (remote)  
- **Samples**: 1000 per experiment
- **Duration**: ~30-40 minutes

**Models Tested**:
| Model Name | Patch | Depth | Dim | Purpose |
|------------|-------|-------|-----|---------|
| `vit0x0x0` | 32 | 0 | 0 | Memory transfer baseline |
| `vit32x2x128` | 32 | 2 | 128 | Memory-bound workload |
| `vit32x4x256` | 32 | 4 | 256 | Light compute |
| `vit32x6x512` | 32 | 6 | 512 | Balanced compute/memory |
| `vit16x8x768` | 16 | 8 | 768 | Moderate compute-bound |
| `vit16x12x1024` | 16 | 12 | 1024 | Heavy compute-bound |

### 2. Quick Test (`quick_test`)
**Purpose**: Framework validation and rapid debugging

**Configuration**: 1 GPU × 1 NUMA node × 3 models = **3 experiments**
- **GPUs**: 0
- **NUMA Nodes**: 0
- **Samples**: 100 per experiment (fast execution)
- **Duration**: ~2-5 minutes

**Models Tested**:
- `vit0x0x0`: No-op baseline
- `vit32x2x256`: Light compute
- `vit32x4x512`: Medium compute

### 3. Model Scaling Study (`scaling_study`)
**Purpose**: Systematic study of model complexity scaling effects

**Configuration**: 1 GPU × 1 NUMA node × 9 models = **9 experiments**
- **GPUs**: 0 (single GPU for clean scaling analysis)
- **NUMA Nodes**: 0 (local NUMA for best performance baseline)
- **Samples**: 2000 per experiment (high precision)
- **Duration**: ~45-60 minutes

**Models Tested**:
- **Depth Scaling**: vit32x2x512, vit32x4x512, vit32x6x512, vit32x8x512, vit32x12x512
- **Dimension Scaling**: vit32x6x256, vit32x6x512, vit32x6x768, vit32x6x1024

## 🔧 Advanced Usage

### Parameter Overrides
```bash
# Test on specific GPUs
python hydra_experiment_runner.py experiment=numa_study experiment.hardware.gpu_ids=[0,1]

# Increase sample size for higher precision
python hydra_experiment_runner.py experiment=quick_test experiment.test.num_samples=500

# Test on different NUMA nodes
python hydra_experiment_runner.py experiment=numa_study experiment.hardware.numa_nodes=[0,1,2]

# Enable model compilation
python hydra_experiment_runner.py experiment=numa_study experiment.test.compile_model=true
```

### Multi-Run Parameter Sweeps
```bash
# Compare compiled vs non-compiled models
python hydra_experiment_runner.py -m experiment=scaling_study experiment.test.compile_model=true,false

# NUMA comparison with compilation study
python hydra_experiment_runner.py -m experiment=numa_study experiment.test.compile_model=true,false experiment.hardware.numa_nodes="[0],[2]"

# Custom model parameter sweep
python hydra_experiment_runner.py -m experiment.vit_configs.depth=2,4,6,8 experiment.hardware.gpu_ids="[0],[1]"
```

### Configuration Customization
```bash
# Use custom experiment configuration
python hydra_experiment_runner.py +experiment=custom experiment.vit_configs.patch_size=16

# Override multiple parameters
python hydra_experiment_runner.py experiment=numa_study experiment.test.batch_size=20 experiment.test.timeout_s=900
```

## 📁 Output Organization

### Directory Structure
```
outputs/
├── numa_locality_study/           # NUMA study results
│   └── 2025-07-12/
│       └── 14-30-15/
│           ├── nsys_reports/      # .nsys-rep profiling files
│           ├── logs/              # stdout/stderr logs per run
│           ├── results.json       # Detailed experiment results
│           └── experiment_summary.json
├── quick_test/                    # Quick test results
└── model_scaling_study/           # Scaling study results
```

### Key Output Files
- **`results.json`**: Complete experiment metadata and results
- **`experiment_summary.json`**: High-level statistics and success rates  
- **`nsys_reports/*.nsys-rep`**: NVIDIA profiling data for each run
- **`logs/*.stdout/.stderr`**: Execution logs per experiment

## 🔍 Analysis Pipeline

### 1. Run Experiments
```bash
python hydra_experiment_runner.py experiment=numa_study
```

### 2. Convert Nsys to SQLite
```bash
# Convert all .nsys-rep files to SQLite databases
find outputs/ -name "*.nsys-rep" -exec nsys export --type=sqlite {} \;
```

### 3. Analyze Results
```bash
# Run unified analysis script
python analyze_latency.py --db-directory outputs/numa_locality_study/2025-07-12/14-30-15/nsys_reports/
```

## 🎯 Reproducible Experiment Scenarios

### Scenario 1: Full NUMA Locality Analysis
```bash
# Complete NUMA study (reproduces original 48 experiments)
python hydra_experiment_runner.py experiment=numa_study

# Expected: 48 successful runs, ~35 minutes, 100% success rate
# Generates: pipeline_gpu{0-3}_numa{0,2}_vit{model} files
```

### Scenario 2: Performance Baseline
```bash
# Quick validation (reproduces original validation)
python hydra_experiment_runner.py experiment=quick_test experiment.hardware.gpu_ids=[0,1,2,3]

# Expected: 12 successful runs, ~5 minutes
# Purpose: Verify framework on all GPUs
```

### Scenario 3: Scaling Analysis
```bash
# Model complexity study (reproduces original scaling)
python hydra_experiment_runner.py experiment=scaling_study

# Expected: 9 successful runs, ~45 minutes
# Purpose: Understand compute vs memory scaling
```

### Scenario 4: Compilation Optimization Study
```bash
# Compare compiled vs non-compiled (reproduces optimization study)
python hydra_experiment_runner.py -m experiment=scaling_study experiment.test.compile_model=true,false

# Expected: 18 runs (9×2), ~90 minutes
# Purpose: Measure torch.compile benefits
```

### Scenario 5: Comprehensive Analysis
```bash
# Full parameter sweep (reproduces comprehensive study)
python hydra_experiment_runner.py -m experiment=numa_study experiment.test.compile_model=true,false experiment.hardware.numa_nodes="[0],[2]" experiment.test.num_samples=500

# Expected: 96 runs (4×2×6×2), ~2-3 hours
# Purpose: Complete NUMA×compilation×GPU×model matrix
```

## 🚨 Troubleshooting

### Common Issues
1. **GPU Memory Error**: Reduce `batch_size` or `memory_size_mb`
2. **NUMA Binding Fails**: Check `numactl --hardware` for available nodes
3. **Nsys Permission Error**: Run with appropriate privileges
4. **Timeout**: Increase `experiment.test.timeout_s`

### Monitoring Progress
```bash
# Check experiment status
tail -f outputs/*/logs/experiment.log

# Monitor GPU usage
nvidia-smi -l 1

# Check NUMA topology
numactl --hardware
```

### Resume Interrupted Experiments
```bash
# Hydra automatically resumes with resume=true (default)
python hydra_experiment_runner.py experiment=numa_study

# Force fresh start
python hydra_experiment_runner.py experiment=numa_study experiment.resume=false
```

## 📈 Expected Results

### Performance Characteristics
- **Memory-bound models** (vit0x0x0, vit32x2x128): High NUMA locality penalty (30-47%)
- **Compute-bound models** (vit16x12x1024): Low NUMA locality penalty (10-15%)
- **Balanced models** (vit32x6x512): Moderate NUMA locality penalty (20-30%)

### Success Metrics
- **Success Rate**: Should achieve 100% (zero failures)
- **Throughput Range**: 300-7000 samples/second depending on model
- **Execution Time**: 2 minutes (quick_test) to 3 hours (comprehensive)

## 🔄 Integration with Analysis

### Post-Experiment Workflow
1. **Execute experiments** using this guide
2. **Convert nsys reports** to SQLite databases  
3. **Run analysis** with `analyze_latency.py`
4. **Generate reports** and visualizations
5. **Compare results** across different configurations

### File Naming Convention
All experiments now generate files with unified naming:
```
pipeline_gpu{N}_numa{N}_vit{patch}x{depth}x{dim}.nsys-rep
```

This ensures seamless integration with the unified analysis pipeline.
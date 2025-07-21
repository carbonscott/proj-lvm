# ViT Inference Optimization - Experiment Execution Guide

This guide provides step-by-step instructions for running ViT inference optimization experiments using the unified Hydra-based framework across all research questions (RQ1-RQ5).

## 🚀 Quick Start

### Prerequisites
- NVIDIA GPUs with CUDA support
- Python environment with PyTorch, Hydra, pandas
- `nsys` profiling tools installed
- `numactl` for NUMA binding

### Basic Execution
```bash
# Run all research questions (RQ1-RQ5) - 157 experiments total
./run_all_experiments.sh

# Run specific research questions
./run_all_experiments.sh rq1 rq2  # Only RQ1 and RQ2

# Individual research question execution
python hydra_experiment_runner.py experiment=rq1_model_architecture_impact
python hydra_experiment_runner.py experiment=rq2_resolution_scaling
python hydra_experiment_runner.py experiment=rq3_compilation_optimization
python hydra_experiment_runner.py experiment=rq4_numa_locality_effects
python hydra_experiment_runner.py experiment=rq5_pipeline_design
```

## 📊 Research Questions and Experiment Configurations

### RQ1: Model Architecture Impact (39 configurations)
**Purpose**: Study how ViT architectural parameters affect inference performance

**Configuration**: Comprehensive ViT parameter sweep
- **Patch sizes**: {8, 16, 32}
- **Depths**: {6, 12, 18, 24}  
- **Dimensions**: {384, 768, 1024, 1280}
- **Samples**: 2000 per experiment
- **Duration**: ~45-60 minutes

**Key Models Tested**:
- Small models: vit32x6x384, vit16x6x384, vit8x6x384
- Medium models: vit32x12x768, vit16x12x768, vit8x12x768
- Large models: vit32x18x1024, vit16x18x1024, vit32x24x1280

### RQ2: Resolution Scaling (17 configurations)
**Purpose**: Study end-to-end pipeline performance vs input resolution

**Configuration**: Resolution scaling across model sizes
- **Input resolutions**: {256², 512², 1024², 2048²}
- **Models**: Representative models from small to large
- **Samples**: 1500 per experiment (reduced for large inputs)
- **Duration**: ~30-45 minutes

**Key Models Tested**:
- Small: vit32x6x384 at all resolutions
- Medium: vit32x12x768 at 256px, 512px, 1024px
- Large: vit32x18x1024 at 256px, 512px

### RQ3: Compilation Optimization (32 configurations)
**Purpose**: Study torch.compile effectiveness across model characteristics and compilation modes

**Configuration**: Systematic evaluation across compilation modes
- **Models**: Representative small to large models
- **Compilation modes**: default, reduce-overhead, max-autotune vs uncompiled
- **Samples**: 2000 per experiment
- **Duration**: ~90-120 minutes (includes compilation overhead)

**Key Comparisons**:
- vit16x6x384: uncompiled vs compiled vs compiled_reduce-overhead vs compiled_max-autotune
- vit32x12x768: uncompiled vs compiled vs compiled_reduce-overhead vs compiled_max-autotune  
- vit32x18x1024: uncompiled vs compiled vs compiled_reduce-overhead vs compiled_max-autotune

**Advanced Compilation Modes**:
- **reduce-overhead**: Targets kernel launch overhead via automatic CUDA graphs
- **max-autotune**: Most aggressive optimization strategy
- **Breakthrough**: Advanced modes provide 30-40% performance improvement for ViT inference

**Execution**:
```bash
# Full RQ3 with all 32 configurations (including advanced compilation modes)
python hydra_experiment_runner.py experiment=rq3_compilation_optimization
```

#### Targeted Execution: Advanced Compilation Modes Only

**Running only the 6 advanced compilation mode experiments** (reduce-overhead and max-autotune):

**Option 1: Create Override Config (Recommended)**
Create `conf/experiment/rq3_advanced_only.yaml`:
```yaml
# @package _global_
defaults:
  - rq3_compilation_optimization

experiment:
  name: "rq3_advanced_compilation_modes_only"
  vit_configs:
    - name: "vit16x6x384_compiled_reduce-overhead"
      patch_size: 16
      depth: 6
      heads: 6
      dim: 384
      mlp_dim: 1536
      compile_model: true
      compile_mode: "reduce-overhead"
    
    - name: "vit16x6x384_compiled_max-autotune"
      patch_size: 16
      depth: 6
      heads: 6
      dim: 384
      mlp_dim: 1536
      compile_model: true
      compile_mode: "max-autotune"
    
    - name: "vit32x12x768_compiled_reduce-overhead"
      patch_size: 32
      depth: 12
      heads: 12
      dim: 768
      mlp_dim: 3072
      compile_model: true
      compile_mode: "reduce-overhead"
    
    - name: "vit32x12x768_compiled_max-autotune"
      patch_size: 32
      depth: 12
      heads: 12
      dim: 768
      mlp_dim: 3072
      compile_model: true
      compile_mode: "max-autotune"
    
    - name: "vit32x18x1024_compiled_reduce-overhead"
      patch_size: 32
      depth: 18
      heads: 16
      dim: 1024
      mlp_dim: 4096
      compile_model: true
      compile_mode: "reduce-overhead"
    
    - name: "vit32x18x1024_compiled_max-autotune"
      patch_size: 32
      depth: 18
      heads: 16
      dim: 1024
      mlp_dim: 4096
      compile_model: true
      compile_mode: "max-autotune"
```

**Usage:**
```bash
# Run only the 6 advanced compilation mode experiments (~15-20 minutes)
python hydra_experiment_runner.py experiment=rq3_advanced_only
```

**Benefits:**
- ✅ No code modifications required
- ✅ Preserves main RQ3 config for full pipeline runs
- ✅ Reusable for future targeted experiments
- ✅ Clean separation of concerns

### RQ4: NUMA Locality Effects (52 configurations)
**Purpose**: Study NUMA binding impact on inference performance

**Configuration**: NUMA local vs remote binding across models
- **NUMA nodes**: 0 (local), 2 (remote)
- **Models**: Lightweight to heavy compute models
- **Batch sizes**: {1, 8, 32} for sensitivity analysis
- **Samples**: 2000 per experiment
- **Duration**: ~60-90 minutes

**Key Models Tested**:
- Memory-bound: vit32x2x256, vit32x6x384
- Balanced: vit32x12x768, vit16x12x768
- Compute-bound: vit32x18x1024, vit16x24x1280

### RQ5: Pipeline Design (23 configurations)
**Purpose**: Study optimal pipeline design for asynchronous operations

**Configuration**: Synchronous vs asynchronous pipeline variants
- **Pipeline modes**: synchronous, asynchronous
- **Queue depths**: {2, 4, 8, 16}
- **Preprocessing**: CPU vs GPU
- **Batch sizes**: {1, 4, 32} for latency vs throughput
- **Samples**: 3000 per experiment (pipeline stability)
- **Duration**: ~45-75 minutes

**Key Comparisons**:
- vit32x6x384_sync_baseline vs vit32x6x384_async_cpu_preproc
- vit16x12x768_async_queue2 vs vit16x12x768_async_queue8
- vit32x18x1024_async_batch1_latency vs vit32x18x1024_async_batch32_throughput

## 🔧 Advanced Usage

### Parameter Overrides
```bash
# Test on specific GPUs
python hydra_experiment_runner.py experiment=rq4_numa_locality_effects experiment.hardware.gpu_ids=[0,1]

# Increase sample size for higher precision
python hydra_experiment_runner.py experiment=rq1_model_architecture_impact experiment.test.num_samples=3000

# Test on different NUMA nodes
python hydra_experiment_runner.py experiment=rq4_numa_locality_effects experiment.hardware.numa_nodes=[0,1,2]

# Enable model compilation with advanced modes
python gpu_numa_pipeline_test.py --compile-model --compile-mode reduce-overhead --vit-patch-size 16 --vit-depth 6 --vit-dim 384
python gpu_numa_pipeline_test.py --compile-model --compile-mode max-autotune --vit-patch-size 32 --vit-depth 12 --vit-dim 768
```

### Multi-Run Parameter Sweeps
```bash
# Compare compiled vs non-compiled models (RQ3)
python hydra_experiment_runner.py -m experiment=rq3_compilation_optimization experiment.test.compile_model=true,false

# NUMA comparison with compilation study (RQ4)
python hydra_experiment_runner.py -m experiment=rq4_numa_locality_effects experiment.test.compile_model=true,false experiment.hardware.numa_nodes="[0],[2]"

# Batch size sweep for architecture study (RQ1)
python hydra_experiment_runner.py -m experiment=rq1_model_architecture_impact experiment.test.batch_size=1,2,4,8,16,32,64

# Resolution scaling sweep (RQ2)  
python hydra_experiment_runner.py -m experiment=rq2_resolution_scaling experiment.test.tensor_shape="[3,256,256],[3,512,512],[3,1024,1024]"

# Pipeline queue depth optimization (RQ5)
python hydra_experiment_runner.py -m experiment=rq5_pipeline_design experiment.test.queue_depth=2,4,8,16
```

### Configuration Customization
```bash
# Use custom experiment configuration
python hydra_experiment_runner.py +experiment=custom experiment.vit_configs.patch_size=16

# Override multiple parameters
python hydra_experiment_runner.py experiment=rq1_model_architecture_impact experiment.test.batch_size=20 experiment.test.timeout_s=900
```

## 📁 Output Organization

### Directory Structure
```
experiments/
├── rq1_model_architecture_impact/    # RQ1 architecture study results
│   └── YYYY-MM-DD/
│       └── HH-MM-SS/
│           ├── nsys_reports/          # .nsys-rep profiling files
│           ├── logs/                  # stdout/stderr logs per run
│           ├── results.json           # Detailed experiment results
│           └── experiment_summary.json
├── rq2_resolution_scaling/            # RQ2 resolution study results
├── rq3_compilation_optimization/      # RQ3 compilation study results
├── rq4_numa_locality_effects/        # RQ4 NUMA study results
└── rq5_pipeline_design/               # RQ5 pipeline study results
```

### Key Output Files
- **`results.json`**: Complete experiment metadata and results
- **`experiment_summary.json`**: High-level statistics and success rates  
- **`nsys_reports/*.nsys-rep`**: NVIDIA profiling data for each run
- **`logs/*.stdout/.stderr`**: Execution logs per experiment

## 🔍 Analysis Pipeline

### 1. Run Experiments
```bash
# Run all research questions
./run_all_experiments.sh

# Or individual research questions
python hydra_experiment_runner.py experiment=rq1_model_architecture_impact
python hydra_experiment_runner.py experiment=rq2_resolution_scaling
python hydra_experiment_runner.py experiment=rq3_compilation_optimization
python hydra_experiment_runner.py experiment=rq4_numa_locality_effects
python hydra_experiment_runner.py experiment=rq5_pipeline_design
```

### 2. Convert Nsys to SQLite
```bash
# Convert all .nsys-rep files to SQLite databases
python batch_nsys_to_sqlite.py experiments/ --verbose
```

### 3. Analyze Results
```bash
# Run comprehensive analysis by research question
cd experiments/rq1_model_architecture_impact/*/*/nsys_reports
python ../../../../../analyze_latency.py --console-summary --output-dir ../../../../../rq1_analysis_results *.sqlite

cd experiments/rq2_resolution_scaling/*/*/nsys_reports  
python ../../../../../analyze_latency.py --console-summary --output-dir ../../../../../rq2_analysis_results *.sqlite

cd experiments/rq3_compilation_optimization/*/*/nsys_reports
python ../../../../../analyze_latency.py --console-summary --output-dir ../../../../../rq3_analysis_results *.sqlite

cd experiments/rq4_numa_locality_effects/*/*/nsys_reports
python ../../../../../analyze_latency.py --console-summary --output-dir ../../../../../rq4_analysis_results *.sqlite

cd experiments/rq5_pipeline_design/*/*/nsys_reports
python ../../../../../analyze_latency.py --console-summary --output-dir ../../../../../rq5_analysis_results *.sqlite
```

## 🎯 Reproducible Experiment Scenarios

### Scenario 1: Full Architecture Impact Analysis (RQ1)
```bash
# Complete architecture study (39 configurations)
python hydra_experiment_runner.py experiment=rq1_model_architecture_impact

# Expected: 39 successful runs, ~45-60 minutes, 100% success rate
# Purpose: Understand how ViT parameters affect performance
```

### Scenario 2: Resolution Scaling Study (RQ2)
```bash
# Resolution scaling analysis (17 configurations)
python hydra_experiment_runner.py experiment=rq2_resolution_scaling

# Expected: 17 successful runs, ~30-45 minutes
# Purpose: Study performance vs input resolution
```

### Scenario 3: Compilation Optimization Study (RQ3)
```bash
# Compare compiled vs non-compiled (26 configurations)
python hydra_experiment_runner.py experiment=rq3_compilation_optimization

# Expected: 26 runs, ~60-90 minutes
# Purpose: Measure torch.compile effectiveness
```

### Scenario 4: NUMA Locality Effects (RQ4)
```bash
# NUMA binding study (52 configurations)
python hydra_experiment_runner.py experiment=rq4_numa_locality_effects

# Expected: 52 runs, ~60-90 minutes
# Purpose: Understand NUMA binding impact
```

### Scenario 5: Pipeline Design Optimization (RQ5)
```bash
# Pipeline design study (23 configurations)
python hydra_experiment_runner.py experiment=rq5_pipeline_design

# Expected: 23 runs, ~45-75 minutes
# Purpose: Optimize asynchronous pipeline strategies
```

### Scenario 6: Comprehensive Analysis (All RQs)
```bash
# Full research question sweep (163 configurations)
./run_all_experiments.sh

# Expected: 163 runs, ~4-6 hours
# Purpose: Complete ViT optimization study across all research questions
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
tail -f logs/rq1_run.log     # For RQ1
tail -f logs/rq2_run.log     # For RQ2
tail -f logs/rq3_run.log     # For RQ3
tail -f logs/rq4_run.log     # For RQ4
tail -f logs/rq5_run.log     # For RQ5

# Monitor GPU usage
nvidia-smi -l 1

# Check NUMA topology
numactl --hardware
```

### Resume Interrupted Experiments
```bash
# Hydra automatically resumes with resume=true (default)
python hydra_experiment_runner.py experiment=rq1_model_architecture_impact

# Force fresh start
python hydra_experiment_runner.py experiment=rq1_model_architecture_impact experiment.resume=false
```

## 📈 Expected Results

### Performance Characteristics
- **Small models** (384-dim): ~30% GPU utilization regardless of optimizations
- **Medium models** (768-dim): ~98% GPU utilization with clear performance threshold
- **Large models** (1024+ dim): ~99% GPU utilization with minimal optimization benefit
- **Resolution scaling**: Well-designed models maintain efficiency across 256px-2048px
- **Compilation effects**: Minimal impact (<1% difference) for ViT inference
- **NUMA effects**: Subtle differences (~1.35x max) rather than dramatic 3x differences

### Success Metrics
- **Success Rate**: Should achieve 100% (zero failures)
- **Throughput Range**: 300-7000 samples/second depending on model
- **Execution Time**: 30 minutes (RQ2) to 6 hours (all RQs)
- **Total Configurations**: 163 across all research questions (includes advanced compilation modes)

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
gpu{N}_numa{N}_vit{patch}x{depth}x{dim}[_suffix].nsys-rep
```

Examples:
- RQ1: `gpu0_numa0_vit32x12x768.nsys-rep`
- RQ2: `gpu0_numa0_vit32x12x768_256px.nsys-rep`
- RQ3: `gpu0_numa0_vit32x12x768_compiled_reduce-overhead.nsys-rep`
- RQ3: `gpu0_numa0_vit32x12x768_compiled_max-autotune.nsys-rep`
- RQ4: `gpu0_numa0_vit32x12x768_numa_local.nsys-rep`
- RQ5: `gpu0_numa0_vit32x12x768_async_cpu_preproc.nsys-rep`

This ensures seamless integration with the unified analysis pipeline.
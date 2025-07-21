# ViT Inference Pipeline Optimization Research

## Project Overview

This research investigates the optimization of Vision Transformer (ViT) inference pipelines, systematically analyzing how model architectural parameters, hardware configurations, compilation strategies, and memory locality optimizations affect end-to-end inference performance.

**Primary Goal**: Maximize throughput for ViT inference pipelines across diverse model configurations and hardware setups.

## Research Questions Addressed

We completed 5 research questions with **157 experiments** conducted in July 2025:

### RQ1: Model Architecture Impact (39 experiments)
How do ViT architectural parameters (patch size, model depth, hidden dimension) affect inference performance?

### RQ2: Input Resolution Scaling (17 experiments) 
How does end-to-end pipeline performance scale with input resolution for different ViT configurations?

### RQ3: Compilation Optimization Effectiveness (26 experiments)
Under what conditions does torch.compile provide significant performance improvements for ViT inference?

**Compilation Mode Analysis**: Systematic evaluation across torch.compile modes (`default`, `reduce-overhead`, `max-autotune`) to characterize kernel optimization effectiveness for different model sizes and workload characteristics.

### RQ4: NUMA Binding and Memory Locality Effects (52 experiments)
How does NUMA binding affect inference performance across different model and workload characteristics?

### RQ5: Pipeline Design and Asynchronous Operations (23 experiments)
What is the optimal pipeline design for maximizing throughput using asynchronous preprocessing and inference overlap?

## Experimental Configuration

**Model Parameter Space**:
- Patch sizes: {8, 16, 32}
- Model depths: {6, 12, 18, 24} 
- Hidden dimensions: {384, 768, 1024, 1280}
- Input resolutions: {256², 512², 1024², 2048²}
- Hardware: NVIDIA L40S, Precision: bfloat16

## Data Location Guide

### Raw Experimental Data
```
experiments/
├── rq1_model_architecture_impact/2025-07-14/    # Model architecture experiments
├── rq2_resolution_scaling/2025-07-14/           # Resolution scaling experiments  
├── rq3_compilation_optimization/2025-07-14/     # torch.compile comparison
├── rq4_numa_locality_effects/2025-07-14/       # NUMA binding experiments
└── rq5_pipeline_design/2025-07-14/             # Pipeline optimization tests
```

Each experiment directory contains:
- `nsys_reports/`: Profiling data (.nsys-rep and .sqlite files)
- `logs/`: Detailed execution logs
- `results.json`: Experiment metadata and configurations

### Analysis Results 
```
rq1_analysis_results/    # Individual performance analysis for each model config
rq2_analysis_results/    # Resolution scaling analysis 
rq3_analysis_results/    # Compilation effectiveness analysis
rq4_analysis_results/    # NUMA binding impact analysis
rq5_analysis_results/    # Pipeline design optimization analysis
```

Each analysis file follows pattern: `gpu0_numa0_vit{patch}x{depth}x{hidden}_{variant}_performance_analysis.txt`

## Key Verified Findings

### 1. Architecture Threshold Effect
- **768+ hidden dimension models**: Achieve 97-99% compute utilization and pipeline efficiency
- **384 hidden dimension models**: Limited to 85-88% efficiency regardless of other optimizations
- **Clear performance threshold** around 768 hidden dimensions

### 2. torch.compile Impact Assessment
- **Minimal performance benefit**: Often 1-2% slower than uncompiled models
- **Not cost-effective**: Compilation overhead outweighs benefits for ViT inference
- **Recommendation**: Skip torch.compile for ViT inference workloads

### 3. NUMA Effects Characterization  
- **Minimal impact**: ~1.01-1.02x performance difference between local/remote NUMA
- **Batch sensitivity**: Single-batch workloads most sensitive to NUMA effects
- **Practical impact**: NUMA optimization provides minimal ROI

### 4. Memory Bandwidth Consistency
- **Consistent performance**: 21-23 GB/s memory bandwidth across all configurations
- **Hardware bottleneck**: Appears to be system limit rather than model-dependent
- **Scaling behavior**: Memory usage scales predictably with model size and resolution

### 5. Resolution Scaling Characteristics
- **Efficient scaling**: Well-designed models maintain performance across 256px-2048px
- **Memory predictability**: Linear scaling behavior observed
- **Architecture dependency**: Patch count matters more than absolute resolution

### 6. Pipeline Design Optimization
- **Simple configurations optimal**: Basic async/sync pipelines perform similarly  
- **Queue depth**: Simple settings (queue=2) often outperform complex configurations
- **Preprocessing strategy**: CPU preprocessing slightly outperforms GPU preprocessing

## Actionable Guidelines

**For Model Selection**:
- Focus on ≥768 hidden dimensions for efficient inference
- Prefer smaller patches for better GPU utilization
- Deeper models generally maintain better performance

**For System Configuration**:
- Skip torch.compile optimization for ViT inference
- Use simple pipeline configurations 
- NUMA optimization not critical for performance

**For Resolution Planning**:
- Well-designed models scale efficiently across resolutions
- Memory requirements scale predictably
- Batch size adjustments minimal across resolution changes

## Data Verification Notes

⚠️ **Important**: Some summary documents may contain outdated claims. For accurate findings, refer to:
- Individual analysis files in `rq*_analysis_results/` directories
- Raw experiment data in `experiments/*/2025-07-14/` folders
- See `notes.md` for data source guidance

## Analysis Pipeline

To reproduce analysis:
```bash
# Convert profiling data 
python batch_nsys_to_sqlite.py experiments/ --verbose

# Run analysis by research question
cd experiments/rq1_model_architecture_impact/*/*/nsys_reports
python ../../../../../analyze_latency.py --console-summary --output-dir ../../../../../rq1_analysis_results *.sqlite
```

**Performance Expectations**: 
- nsys-to-SQLite conversion: ~30 minutes for 157 files
- Analysis per RQ: 5-15 minutes depending on configuration count

---

*This document serves as a project guide for anyone to understand the completed ViT optimization research, locate experimental results, and build upon the verified findings.*

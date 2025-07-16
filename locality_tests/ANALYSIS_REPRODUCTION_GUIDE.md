# ViT Inference Analysis: Complete Reproduction Guide

This document provides step-by-step commands to reproduce the comprehensive analysis of 157 ViT inference configurations across 5 research questions.

## Prerequisites

- NVIDIA Nsight Systems with `nsys` command available
- Python environment with required analysis tools
- Access to experimental data directory: `/sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/`

## Phase 1: Data Discovery and Verification

### 1.1 Verify Complete Dataset
```bash
# Count total nsys reports across all experiments
find /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/experiments -name "*.nsys-rep" | wc -l
# Expected: 157

# Count reports by research question
for rq in rq1_model_architecture_impact rq2_resolution_scaling rq3_compilation_optimization rq4_numa_locality_effects rq5_pipeline_design; do 
    echo "=== $rq ==="
    find /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/experiments/$rq -name "*.nsys-rep" | wc -l
done
```

**Expected Results:**
- RQ1 (Model Architecture): 39 reports
- RQ2 (Resolution Scaling): 17 reports  
- RQ3 (Compilation): 26 reports
- RQ4 (NUMA Effects): 52 reports
- RQ5 (Pipeline Design): 23 reports
- **Total: 157 reports**

### 1.2 Verify Alternative Data Location
```bash
# Check if experiments exist in additional working directory
find /sdf/data/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/experiments -name "*.nsys-rep" | wc -l
# Should also show 157
```

## Phase 2: Convert nsys Reports to SQLite

### 2.1 Batch Conversion Process
```bash
# Navigate to base directory
cd /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests

# Convert all nsys reports to SQLite format (takes ~30 minutes)
python batch_nsys_to_sqlite.py experiments/ --verbose
```

**Expected Output Pattern:**
```
Found 157 nsys report(s) to convert

Converting to SQLite...
============================================================
[1/157] Processing gpu0_numa0_vit16x12x1024.nsys-rep...
  ✓ Successfully converted: gpu0_numa0_vit16x12x1024.nsys-rep → gpu0_numa0_vit16x12x1024.sqlite
[2/157] Processing gpu0_numa0_vit16x12x1280.nsys-rep...
  ✓ Successfully converted: gpu0_numa0_vit16x12x1280.nsys-rep → gpu0_numa0_vit16x12x1280.sqlite
...
============================================================
Conversion completed: 157 successful, 0 failed
```

### 2.2 Verify Conversion Success
```bash
# Check that all SQLite files were created
find experiments -name "*.sqlite" | wc -l
# Expected: 157
```

## Phase 3: Systematic Analysis by Research Question

### 3.1 RQ1: Model Architecture Impact (39 configs)

**Research Focus:** How ViT architectural parameters affect inference performance

```bash
# Navigate to RQ1 nsys reports directory
cd /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/experiments/rq1_model_architecture_impact/2025-07-14/16-30-54/nsys_reports

# Run comprehensive analysis
python ../../../../../analyze_latency.py --console-summary --output-dir ../../../../../rq1_analysis_results *.sqlite
```

**Manual Inspection Points:**
- **Console Output Patterns:**
  - 384-dim models: Temporal Compression ~40-50%, Compute Utilization ~30-35%
  - 768+ dim models: Temporal Compression ~98-100%, Compute Utilization ~95-98%
- **Critical Files to Examine:**
  ```bash
  # Poor performance example
  cat ../../../../../rq1_analysis_results/gpu0_numa0_vit32x6x384_performance_analysis.txt
  
  # Good performance example  
  cat ../../../../../rq1_analysis_results/gpu0_numa0_vit16x12x1024_performance_analysis.txt
  ```
- **Expected Findings:**
  - Clear performance threshold at ~768 hidden dimensions
  - Patch count matters more than patch size
  - Memory bandwidth consistent ~21-23 GB/s across models

### 3.2 RQ2: Resolution Scaling (17 configs)

**Research Focus:** End-to-end pipeline performance vs input resolution

```bash
# Navigate to RQ2 directory
cd /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/experiments/rq2_resolution_scaling/2025-07-14/16-46-49/nsys_reports

# Run analysis
python ../../../../../analyze_latency.py --console-summary --output-dir ../../../../../rq2_analysis_results *.sqlite
```

**Manual Inspection Points:**
- **Resolution Patterns:** Look for `_256px`, `_512px`, `_1024px`, `_2048px` in filenames
- **Expected Findings:**
  - Well-designed models maintain efficiency across resolutions
  - ViT 8x6x384 (many patches): ~101% efficiency
  - ViT 32x6x384 (few patches): ~40% efficiency at all resolutions
- **Key Comparison:**
  ```bash
  # Compare same model at different resolutions
  grep "Temporal Compression Ratio" ../../../../../rq2_analysis_results/*_256px_* 
  grep "Temporal Compression Ratio" ../../../../../rq2_analysis_results/*_1024px_*
  ```

### 3.3 RQ3: Compilation Optimization (26 configs)

**Research Focus:** torch.compile effectiveness across model characteristics

```bash
# Navigate to RQ3 directory
cd /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/experiments/rq3_compilation_optimization/2025-07-14/16-52-40/nsys_reports

# Run analysis
python ../../../../../analyze_latency.py --console-summary --output-dir ../../../../../rq3_analysis_results *.sqlite
```

**Manual Inspection Points:**
- **Compiled vs Uncompiled Pairs:** Look for `_compiled` vs `_uncompiled` suffixes
- **Expected Findings:**
  - Minimal differences: <1% performance improvement from compilation
  - Large models: 99.9% vs 100.0% efficiency (negligible)
  - Small models: 42.9% vs 42.9% efficiency (no improvement)
- **Systematic Comparison:**
  ```bash
  # Extract performance metrics for comparison
  grep "Duration:" ../../../../../rq3_analysis_results/*_compiled_* | sort
  grep "Duration:" ../../../../../rq3_analysis_results/*_uncompiled_* | sort
  ```

### 3.4 RQ4: NUMA Effects (52 configs)

**Research Focus:** NUMA binding impact on inference performance

```bash
# Navigate to RQ4 directory
cd /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/experiments/rq4_numa_locality_effects/2025-07-14/17-03-11/nsys_reports

# Run analysis (longer due to 52 configurations)
python ../../../../../analyze_latency.py --console-summary --output-dir ../../../../../rq4_analysis_results *.sqlite
```

**Manual Inspection Points:**
- **NUMA Patterns:** Look for `numa0` vs `numa2`, `numa_local` vs `numa_remote`
- **Search for Performance Differences:**
  ```bash
  # Compare NUMA configurations
  grep -r "Duration:" ../../../../../rq4_analysis_results/ | grep -E "(numa_local|numa_remote)" | head -20
  
  # Focus on batch1 (most sensitive to NUMA effects)
  grep -r "Duration:" ../../../../../rq4_analysis_results/ | grep "batch1" | sort
  
  # Find potential 3x differences
  grep -r "Duration:" ../../../../../rq4_analysis_results/ | sort -k2 -n | tail -10
  ```
- **Expected Findings:**
  - NUMA effects more subtle than expected
  - batch1: ~1.35x difference (1190ms vs 1607ms)
  - Larger batches: minimal NUMA impact
  - Memory bandwidth remains consistent across NUMA nodes

### 3.5 RQ5: Pipeline Design (23 configs)

**Research Focus:** Optimal pipeline design for asynchronous preprocessing and inference

```bash
# Navigate to RQ5 directory
cd /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/experiments/rq5_pipeline_design/2025-07-14/17-38-57/nsys_reports

# Run analysis
python ../../../../../analyze_latency.py --console-summary --output-dir ../../../../../rq5_analysis_results *.sqlite
```

**Manual Inspection Points:**
- **Pipeline Patterns:**
  - `sync_baseline` vs `async_*` (synchronous vs asynchronous)
  - `queue2`, `queue8`, `queue16` (queue depth variations)
  - `cpu_preproc` vs `gpu_preproc` (preprocessing strategies)
  - `batch1_latency`, `batch4_balanced`, `batch32_throughput` (latency vs throughput)
- **Expected Findings:**
  - Pipeline optimizations provide <2% variation for well-designed models
  - Queue depth 2 often optimal
  - CPU preprocessing slightly better than GPU
  - Architecture dominates over pipeline strategy

## Phase 4: Cross-Analysis and Pattern Recognition

### 4.1 Verify Analysis Completion
```bash
# Check that all analysis phases completed successfully
find /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/rq1_analysis_results -name "*.txt" | wc -l  # Should be 39
find /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/rq2_analysis_results -name "*.txt" | wc -l  # Should be 17
find /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/rq3_analysis_results -name "*.txt" | wc -l  # Should be 26
find /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/rq4_analysis_results -name "*.txt" | wc -l  # Should be 52
find /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/rq5_analysis_results -name "*.txt" | wc -l  # Should be 23

# Total should be 157
find /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/*_analysis_results -name "*.txt" | wc -l
```

### 4.2 Key Insights Extraction

**Architecture Performance Cliff (RQ1):**
```bash
# Find the performance threshold
grep "Compute Utilization" /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/rq1_analysis_results/*.txt | sort -k3 -n
```

**Resolution Independence (RQ2):**
```bash
# Compare efficiency across resolutions for same model
grep "Temporal Compression Ratio" /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/rq2_analysis_results/*vit32x12x768* | sort
```

**Compilation Ineffectiveness (RQ3):**
```bash
# Compare compiled vs uncompiled performance
for model in vit16x12x768 vit32x12x768 vit32x18x1024; do
    echo "=== $model ==="
    grep "Duration:" /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/rq3_analysis_results/*${model}_compiled_performance* 
    grep "Duration:" /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/rq3_analysis_results/*${model}_uncompiled_performance*
done
```

**NUMA Subtlety (RQ4):**
```bash
# Find NUMA effects in batch1 configurations
grep "Duration:" /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/rq4_analysis_results/*batch1* | sort -k2 -n
```

**Pipeline Optimization Limits (RQ5):**
```bash
# Compare pipeline strategies
grep "Duration:" /sdf/scratch/lcls/ds/prj/prjcwang31/scratch/proj-lvm/run_2025_0714_1347/locality_tests/rq5_analysis_results/*vit16x12x768* | sort -k2 -n
```

## Critical Manual Inspection Points

### 1. Performance Threshold Analysis
**What to Look For:**
- Dramatic performance cliff around 768 hidden dimensions
- 384-dim models: ~30% efficiency regardless of other optimizations
- 768+ dim models: ~98% efficiency across configurations

### 2. Architecture Pattern Recognition  
**Key Comparisons:**
- ViT 8x6x384 (many small patches) vs ViT 32x6x384 (few large patches)
- Performance maintained across resolutions for well-designed models
- Depth scaling: deeper models generally maintain better GPU utilization

### 3. Optimization Technique Effectiveness
**Expected Findings:**
- torch.compile: Minimal impact (<1% difference)
- Pipeline tuning: <2% variation for efficient models
- Queue depth: Often queue=2 > queue=8 > queue=16
- NUMA: Subtle effects (1.2-1.4x max, not 3x)

### 4. Memory Bandwidth Consistency
**Consistent Across All Configurations:**
- H2D bandwidth: 21-23 GB/s
- D2H bandwidth: 21-23 GB/s  
- Minimal variation across NUMA nodes or model sizes

## Summary of Key Findings

**Revolutionary Insights:**
1. **Architecture Dominates Everything:** 768+ dim models achieve 98% efficiency regardless of optimization
2. **Common Optimizations Are Ineffective:** torch.compile, pipeline tuning provide minimal benefits
3. **Hardware Effects Are Subtle:** NUMA differences are 1.35x, not 3x as initially suggested
4. **Resolution Scaling Defies Expectations:** Well-designed models maintain efficiency across 256px-2048px

**Actionable Guidelines:**
- Focus on model architecture first (≥768 dim, many small patches)
- Skip torch.compile for ViT inference
- Use simple pipeline settings (queue=2, batch=4, CPU preprocessing)
- NUMA optimization has minimal ROI for well-designed models

## Troubleshooting

**Common Issues:**
- `nsys command not found`: Ensure NVIDIA Nsight Systems is installed
- `Permission denied`: Check file permissions on experiment directories
- `Analysis timeouts`: Some large models may take longer to analyze
- `Missing files`: Verify all 157 .nsys-rep files exist before conversion

**Performance Expectations:**
- nsys-to-SQLite conversion: ~30 minutes for 157 files
- Each RQ analysis: 5-15 minutes depending on configuration count
- Total analysis time: ~1-2 hours for complete reproduction

This reproduction guide enables independent verification of all findings from the comprehensive ViT inference optimization study.
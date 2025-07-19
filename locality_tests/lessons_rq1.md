# RQ1 Lessons: Model Architecture Impact on ViT Inference Performance

## Executive Summary

**Research Question**: How do ViT architectural parameters (patch size, model depth, hidden dimension) affect inference performance?

**Key Finding**: **Hidden dimension is the dominant factor** determining GPU utilization, with a critical threshold at 768 dimensions. Patch size and depth provide secondary optimization opportunities.

## Critical Performance Threshold

### Hidden Dimension Impact - The 768 Threshold Rule

**Finding**: **≥768 hidden dimensions are required for efficient GPU utilization** in most practical scenarios. The only exception is very small patches (8x8), which can achieve good performance even with 384 dimensions due to their massive computational workload.

#### The Standard 768 Threshold: Patch Sizes 16 and 32

**Patch Size 32 Models (dramatic threshold effect):**
| Hidden Dim | Compute Util | Pipeline Eff | Large Gaps | Duration |
|------------|--------------|--------------|------------|----------|
| **384**    | **33.5%**    | **33.9%**    | 32,739     | 1,192ms  |
| **768**    | **82.1%**    | **84.5%**    | 4,202      | 1,185ms  |
| **1024**   | **95.2%**    | **97.2%**    | 7          | 1,696ms  |

**Patch Size 16 Models (clear threshold effect):**
| Hidden Dim | Compute Util | Pipeline Eff | Large Gaps | Duration |
|------------|--------------|--------------|------------|----------|
| **384**    | **85.5%**    | **88.2%**    | 1,873      | 1,166ms  |
| **768**    | **97.6%**    | **98.6%**    | 3          | 3,433ms  |
| **1024**   | **98.6%**    | **99.3%**    | 5          | 5,471ms  |

#### Special Case Exception: Very Small Patches (8x8)

**Patch Size 8 Models (384 dimensions viable due to large workload):**
| Hidden Dim | Compute Util | Pipeline Eff | Large Gaps | Duration |
|------------|--------------|--------------|------------|----------|
| **384**    | **99.1%**    | **99.6%**    | 6          | 6,623ms  |
| **768**    | **99.6%**    | **99.8%**    | 4          | 17,252ms |
| **1024**   | **99.8%**    | **99.9%**    | 3          | 26,041ms |

**Key Insights**: 
- **General Rule**: Use ≥768 hidden dimensions for efficient GPU utilization
- **Threshold Effect**: 384→768 dimensions show 2-3x utilization improvement for standard patches
- **Special Exception**: Patch size 8 can rescue 384-dimension models due to massive workload (784 vs 49 patches per image)
- **Design Recommendation**: Plan for ≥768 dimensions unless specifically choosing patch size 8

#### Fact Check - Hidden Dimension Threshold
**Analysis Files**:
```
# Patch 32 models (dramatic threshold)
rq1_analysis_results/gpu0_numa0_vit32x12x384_performance_analysis.txt
rq1_analysis_results/gpu0_numa0_vit32x12x768_performance_analysis.txt  
rq1_analysis_results/gpu0_numa0_vit32x12x1024_performance_analysis.txt

# Patch 16 models (clear threshold)
rq1_analysis_results/gpu0_numa0_vit16x12x384_performance_analysis.txt
rq1_analysis_results/gpu0_numa0_vit16x12x768_performance_analysis.txt
rq1_analysis_results/gpu0_numa0_vit16x12x1024_performance_analysis.txt

# Patch 8 models (exception case)
rq1_analysis_results/gpu0_numa0_vit8x12x384_performance_analysis.txt
rq1_analysis_results/gpu0_numa0_vit8x12x768_performance_analysis.txt
rq1_analysis_results/gpu0_numa0_vit8x12x1024_performance_analysis.txt
```

**Raw Profiler Data**:
```
# Patch 32 models
experiments/rq1_model_architecture_impact/2025-07-14/16-30-54/nsys_reports/gpu0_numa0_vit32x12x384.nsys-rep
experiments/rq1_model_architecture_impact/2025-07-14/16-30-54/nsys_reports/gpu0_numa0_vit32x12x768.nsys-rep
experiments/rq1_model_architecture_impact/2025-07-14/16-30-54/nsys_reports/gpu0_numa0_vit32x12x1024.nsys-rep

# Patch 16 models  
experiments/rq1_model_architecture_impact/2025-07-14/16-30-54/nsys_reports/gpu0_numa0_vit16x12x384.nsys-rep
experiments/rq1_model_architecture_impact/2025-07-14/16-30-54/nsys_reports/gpu0_numa0_vit16x12x768.nsys-rep
experiments/rq1_model_architecture_impact/2025-07-14/16-30-54/nsys_reports/gpu0_numa0_vit16x12x1024.nsys-rep

# Patch 8 models
experiments/rq1_model_architecture_impact/2025-07-14/16-30-54/nsys_reports/gpu0_numa0_vit8x12x384.nsys-rep
experiments/rq1_model_architecture_impact/2025-07-14/16-30-54/nsys_reports/gpu0_numa0_vit8x12x768.nsys-rep
experiments/rq1_model_architecture_impact/2025-07-14/16-30-54/nsys_reports/gpu0_numa0_vit8x12x1024.nsys-rep
```

**What to Verify**:
- In analysis files: Look for "Compute utilization" line (line ~40) 
  - Patch 32: 33.5% vs 82.1% vs 95.2% (dramatic jump at 768)
  - Patch 16: 85.5% vs 97.6% vs 98.6% (clear improvement at 768)
  - Patch 8: 99.1% vs 99.6% vs 99.8% (excellent across all dimensions)
- In analysis files: Check "Large gaps (>10μs)" count (line ~54): Shows dramatic reduction at 768+ for patches 16&32
- In .nsys-rep files (Nsight Systems): GPU timeline density correlates with utilization percentages

---

## Patch Size Impact - Smaller Patches Enable Better GPU Saturation

**Finding**: Smaller patches dramatically improve GPU utilization by creating larger computational workloads per image.

### Patch Size Analysis: 12-layer, 768-dimension Models

| Patch Size | Compute Util | Pipeline Eff | Duration | Avg Kernel | Total Kernels |
|------------|--------------|--------------|----------|------------|---------------|
| **8x12x768**  | **99.6%**    | **99.8%**    | 17.2s    | 0.33ms     | 52,131        |
| **16x12x768** | **97.6%**    | **98.6%**    | 3.4s     | 0.06ms     | 52,173        |
| **32x12x768** | **82.1%**    | **84.5%**    | 1.2s     | 0.02ms     | 53,000        |

### Patch Size Analysis: 12-layer, 384-dimension Models

| Patch Size | Compute Util | Pipeline Eff | Duration | Avg Kernel | Total Kernels |
|------------|--------------|--------------|----------|------------|---------------|
| **8x12x384**  | **99.1%**    | **99.6%**    | 6.6s     | 0.13ms     | 52,087        |
| **16x12x384** | **85.5%**    | **88.2%**    | 1.2s     | 0.02ms     | 53,000        |
| **32x12x384** | **33.5%**    | **33.9%**    | 1.2s     | 0.01ms     | 53,000        |

**Key Insight**: Patch size 8 can rescue even 384-dimension models to achieve 99%+ utilization. Small patches create more patches per image (8×8 = 784 patches vs 32×32 = 49 patches), leading to larger workloads.

#### Fact Check - Patch Size Impact
**Analysis Files**:
```
rq1_analysis_results/gpu0_numa0_vit8x12x768_performance_analysis.txt
rq1_analysis_results/gpu0_numa0_vit16x12x768_performance_analysis.txt
rq1_analysis_results/gpu0_numa0_vit32x12x768_performance_analysis.txt
rq1_analysis_results/gpu0_numa0_vit8x12x384_performance_analysis.txt
rq1_analysis_results/gpu0_numa0_vit16x12x384_performance_analysis.txt
rq1_analysis_results/gpu0_numa0_vit32x12x384_performance_analysis.txt
```

**Raw Profiler Data**:
```
experiments/rq1_model_architecture_impact/2025-07-14/16-30-54/nsys_reports/gpu0_numa0_vit8x12x768.nsys-rep
experiments/rq1_model_architecture_impact/2025-07-14/16-30-54/nsys_reports/gpu0_numa0_vit16x12x768.nsys-rep
experiments/rq1_model_architecture_impact/2025-07-14/16-30-54/nsys_reports/gpu0_numa0_vit32x12x768.nsys-rep
```

**What to Verify**:
- In analysis files: "Compute utilization" progression (99.6% → 97.6% → 82.1% for 768-dim)
- In analysis files: "Duration" scaling shows longer execution for smaller patches
- In .nsys-rep files: Timeline density - patch 8 shows much longer, denser GPU activity

---

## Depth Impact - Moderate Improvement When Fundamentals Are Sound

**Finding**: Increasing depth provides gradual performance improvements, but **only when combined with adequate hidden dimensions (≥768)**. Cannot rescue models with insufficient computational density.

### Depth Analysis: 32-patch, 768-dimension Models

| Depth | Compute Util | Pipeline Eff | Total Kernels | Duration |
|-------|--------------|--------------|---------------|----------|
| **6**  | **76.9%**    | **79.1%**    | 27,500        | 653ms    |
| **12** | **82.1%**    | **84.5%**    | 53,000        | 1,185ms  |
| **18** | **83.5%**    | **85.9%**    | 78,500        | 1,733ms  |
| **24** | **85.6%**    | **88.1%**    | 104,000       | 2,239ms  |

### Depth Analysis: 32-patch, 384-dimension Models

| Depth | Compute Util | Pipeline Eff | Total Kernels | Duration |
|-------|--------------|--------------|---------------|----------|
| **6**  | **31.3%**    | **31.8%**    | 27,500        | 651ms    |
| **12** | **33.5%**    | **33.9%**    | 53,000        | 1,192ms  |
| **18** | **34.3%**    | **34.8%**    | 78,500        | 1,735ms  |
| **24** | **35.4%**    | **35.8%**    | 104,000       | 2,223ms  |

**Key Insight**: For 768-dim models, depth improves utilization from 76.9% to 85.6%. For 384-dim models, all depths remain stuck at poor utilization (~31-35%).

#### Fact Check - Depth Impact
**Analysis Files**:
```
rq1_analysis_results/gpu0_numa0_vit32x6x768_performance_analysis.txt
rq1_analysis_results/gpu0_numa0_vit32x12x768_performance_analysis.txt
rq1_analysis_results/gpu0_numa0_vit32x18x768_performance_analysis.txt
rq1_analysis_results/gpu0_numa0_vit32x24x768_performance_analysis.txt
```

**Raw Profiler Data**:
```
experiments/rq1_model_architecture_impact/2025-07-14/16-30-54/nsys_reports/gpu0_numa0_vit32x6x768.nsys-rep
experiments/rq1_model_architecture_impact/2025-07-14/16-30-54/nsys_reports/gpu0_numa0_vit32x12x768.nsys-rep
experiments/rq1_model_architecture_impact/2025-07-14/16-30-54/nsys_reports/gpu0_numa0_vit32x18x768.nsys-rep
experiments/rq1_model_architecture_impact/2025-07-14/16-30-54/nsys_reports/gpu0_numa0_vit32x24x768.nsys-rep
```

**What to Verify**:
- In analysis files: "Total kernels" linear scaling (27K → 53K → 78K → 104K)
- In analysis files: Gradual "Compute utilization" improvement for 768-dim models only
- In .nsys-rep files: Timeline duration scales linearly with depth

---

## Parameter Hierarchy and Interaction Effects

### Importance Ranking
1. **Hidden Dimension** (dominant factor) - Determines if GPU can be saturated
2. **Patch Size** (strong secondary factor) - Controls workload size per image  
3. **Depth** (moderate factor) - Provides incremental improvements when fundamentals are sound

### Critical Interaction Effects

1. **Small patches can rescue low-dimension models**: 8x*x384 achieves 99%+ utilization
2. **Large patches make dimension limitations catastrophic**: 32x*x384 limited to ~33% utilization
3. **Depth cannot overcome inadequate hidden dimensions**: All 384-dim models plateau regardless of depth

---

## Actionable Guidelines

### For High-Performance ViT Inference:

1. **Primary**: Use ≥768 hidden dimensions
   - 384 dimensions create fundamental GPU underutilization
   - 768+ enables 82-99% GPU utilization

2. **Secondary**: Prefer smaller patch sizes (8 > 16 > 32)
   - Smaller patches create larger computational workloads
   - Can partially compensate for low hidden dimensions

3. **Tertiary**: Increase depth when other fundamentals are sound
   - Provides 10-15% utilization improvement for well-configured models
   - Ineffective for rescuing poorly-configured models

### Design Decision Framework:

```
IF hidden_dim >= 768:
    Use any patch size, depth provides incremental benefits
    Expected utilization: 80-99%
    
ELIF hidden_dim == 384:
    MUST use patch_size <= 16 (preferably 8)
    Depth has minimal impact
    Expected utilization: 33-99% (patch-dependent)
    
ELSE:
    Model likely unsuitable for efficient GPU inference
```

---

## Experimental Configuration

**Total Configurations Tested**: 39 ViT variants
- **Patch sizes**: 8, 16, 32
- **Depths**: 6, 12, 18, 24 layers
- **Hidden dimensions**: 384, 768, 1024, 1280
- **Hardware**: NVIDIA L40S GPU, precision: bfloat16
- **Input**: 224×224 ImageNet resolution, batch size 8

**Configuration File**: `conf/experiment/rq1_model_architecture_impact.yaml`

---

## Memory and Performance Characteristics

### Consistent Memory Bandwidth
All configurations achieved **21-23 GB/s memory bandwidth**, suggesting this is a hardware limit rather than model-dependent bottleneck.

### Timeline Scaling
- **Patch 8 models**: 6-17 second execution times (high workload)
- **Patch 16 models**: 1-3 second execution times (medium workload)  
- **Patch 32 models**: 0.6-2 second execution times (low workload)

The results demonstrate that **computational density** (operations per unit time) is the key factor determining GPU utilization efficiency in ViT inference pipelines.
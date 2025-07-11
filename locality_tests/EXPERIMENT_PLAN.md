# GPU NUMA Pipeline Experiment Plan

## 🎯 Objective

Systematically study GPU-NUMA locality effects on ML model inference pipeline performance, with focus on understanding the interplay between memory transfer (H2D/D2H) and compute workloads across different NUMA node placements.

## 🔬 Core Research Questions

1. **NUMA Sensitivity Analysis**: Which ViT model configurations show the strongest NUMA locality effects?
2. **Compute vs Memory Bound Transition**: At what model complexity does compute dominate over memory transfer bottlenecks?
3. **D2H Transfer Scaling**: How does our enhanced D2H transfer (using ViTForProfiling) scale with model complexity?
4. **Hardware Optimization**: What are the optimal GPU-NUMA pairings for different workload characteristics?
5. **Compilation Benefits**: Where does torch.compile provide the most significant performance improvements?

## 📋 Experimental Phase Plan

### **Phase 1: Framework Validation & Baseline** ⚡ *Priority: High*

**Goal**: Validate experimental framework and establish baseline performance metrics.

```bash
# Quick framework test
python hydra_experiment_runner.py experiment=quick_test

# Multi-GPU validation
python hydra_experiment_runner.py experiment=quick_test \
  experiment.hardware.gpu_ids=[0,1,2,3]

# Baseline single-GPU performance
python hydra_experiment_runner.py experiment=quick_test \
  experiment.test.num_samples=500
```

**Expected Outcomes**:
- Framework validation on all available GPUs
- Baseline performance metrics for comparison
- Identification of any hardware-specific issues

---

### **Phase 2: Core NUMA Locality Study** 🎯 *Priority: High*

**Goal**: Understand fundamental NUMA locality effects across different model complexities.

```bash
# Comprehensive NUMA study
python hydra_experiment_runner.py experiment=numa_study

# Focus on key GPU-NUMA combinations
python hydra_experiment_runner.py experiment=numa_study \
  experiment.hardware.gpu_ids=[0,3] \
  experiment.hardware.numa_nodes=[0,2,3]

# Extended sampling for statistical significance
python hydra_experiment_runner.py experiment=numa_study \
  experiment.test.num_samples=2000
```

**Key Metrics to Analyze**:
- Throughput (samples/sec) vs NUMA node distance
- Performance degradation for remote NUMA access
- Model-specific NUMA sensitivity patterns

**Expected Insights**:
- Light models (depth 2-4): Strong NUMA effects (memory-bound)
- Heavy models (depth 12+): Reduced NUMA sensitivity (compute-bound)
- Medium models (depth 6-8): Transition behavior

---

### **Phase 3: Model Complexity Scaling Analysis** 📈 *Priority: High*

**Goal**: Systematically understand how model complexity affects the compute/memory balance.

```bash
# Systematic scaling study
python hydra_experiment_runner.py experiment=scaling_study

# Extended depth scaling
python hydra_experiment_runner.py -m \
  experiment=scaling_study \
  'experiment.vit_configs[0].depth=2,4,6,8,10,12,16,20'

# Dimension scaling analysis
python hydra_experiment_runner.py -m \
  experiment=scaling_study \
  'experiment.vit_configs[0].dim=128,256,384,512,768,1024,1280'
```

**Analysis Focus**:
- Throughput scaling with model parameters
- Memory usage vs compute time relationship
- Identification of scaling regimes (linear, super-linear, saturation)

---

### **Phase 4: Memory Transfer Characterization** 💾 *Priority: Medium*

**Goal**: Understand how different memory transfer patterns affect NUMA sensitivity.

```bash
# Variable image sizes (H2D scaling)
python hydra_experiment_runner.py -m \
  experiment=numa_study \
  experiment.test.tensor_shape="[3,224,224],[3,384,384],[3,512,512],[3,768,768]" \
  experiment.test.batch_size=10,8,6,4  # Adjust for memory constraints

# Batch size effects
python hydra_experiment_runner.py -m \
  experiment=numa_study \
  experiment.test.batch_size=4,8,16,32 \
  experiment.test.memory_size_mb=512,1024

# Memory pool size effects
python hydra_experiment_runner.py -m \
  experiment=numa_study \
  experiment.test.memory_size_mb=256,512,1024,2048
```

**Research Focus**:
- H2D transfer scaling with image resolution
- D2H transfer characteristics (should be constant with ViTForProfiling)
- Memory pressure effects on NUMA sensitivity

---

### **Phase 5: Compilation Optimization Study** ⚡ *Priority: Medium*

**Goal**: Quantify torch.compile benefits across different model configurations and NUMA setups.

```bash
# Compilation benefits across model sizes
python hydra_experiment_runner.py -m \
  experiment=scaling_study \
  experiment.test.compile_model=true,false

# NUMA effects with/without compilation
python hydra_experiment_runner.py -m \
  experiment=numa_study \
  experiment.test.compile_model=true,false

# Small model focus (highest expected benefit)
python hydra_experiment_runner.py -m \
  experiment.test.compile_model=true,false \
  'experiment.vit_configs=[{name:small,patch_size:32,depth:2,heads:4,dim:256,mlp_dim:1024}]'
```

**Expected Results**:
- Larger speedups for smaller models (reduced kernel overhead)
- Compilation overhead vs runtime benefits trade-off
- NUMA interaction with compiled models

---

### **Phase 6: Production Optimization** 🎯 *Priority: Low*

**Goal**: Identify optimal configurations for real-world deployment scenarios.

```bash
# High-throughput configurations
python hydra_experiment_runner.py -m \
  experiment=numa_study \
  experiment.test.batch_size=16,32,64 \
  experiment.max_workers=8

# Long-running stability test
python hydra_experiment_runner.py experiment=numa_study \
  experiment.test.num_samples=5000 \
  experiment.test.timeout_s=1800

# Resource utilization optimization
python hydra_experiment_runner.py -m \
  experiment=numa_study \
  experiment.hardware.gpu_ids="[0],[1],[2],[3],[0,1],[0,2]"  # Multi-GPU patterns
```

## 📊 Analysis and Reporting Strategy

### **Immediate Analysis (After Each Phase)**
```bash
# Check experiment results
cat outputs/*/experiment_summary.json

# View detailed results
jq '.[] | {run_id, duration, success, gpu_id, numa_node}' outputs/*/results.json
```

### **Comprehensive Analysis Script** (Future)
```python
# Load and analyze all experimental results
import pandas as pd
import matplotlib.pyplot as plt

# Aggregate results across experiments
results_df = load_all_experiments()

# Key analyses:
# 1. NUMA sensitivity heatmaps
# 2. Model scaling curves
# 3. Memory vs compute bound classification
# 4. Optimal configuration recommendations
```

## 🚨 Critical Success Metrics

### **Performance Metrics**
- **Throughput**: Samples per second across configurations
- **NUMA Penalty**: Performance degradation for remote NUMA access
- **Scaling Efficiency**: How performance scales with model complexity
- **Memory Utilization**: GPU memory usage patterns

### **Research Deliverables**
- **NUMA Sensitivity Map**: Which models are most/least NUMA-sensitive
- **Crossover Analysis**: Compute vs memory bound transition points
- **Best Practices Guide**: Optimal GPU-NUMA configurations
- **Scaling Laws**: Mathematical models for performance prediction

## ⏱️ Estimated Timeline

| Phase | Duration | Priority | Dependencies |
|-------|----------|----------|--------------|
| Phase 1 | 2-4 hours | High | GPU access |
| Phase 2 | 8-12 hours | High | Phase 1 complete |
| Phase 3 | 6-10 hours | High | Phase 1 complete |
| Phase 4 | 4-8 hours | Medium | Phase 2 complete |
| Phase 5 | 4-6 hours | Medium | Phase 3 complete |
| Phase 6 | 6-12 hours | Low | All phases complete |

**Total Estimated Time**: 30-52 GPU hours

## 🎯 Execution Strategy

### **Recommended Order**
1. **Start immediately**: Phase 1 (Framework validation)
2. **Parallel execution**: Phase 2 + Phase 3 (if sufficient GPU resources)
3. **Sequential**: Phase 4 → Phase 5 → Phase 6

### **Resource Management**
- **Peak GPU usage**: 4 GPUs × 4 workers = up to 16 parallel experiments
- **Storage requirements**: ~1-2 GB per experiment run (nsys reports)
- **Monitor logs**: `tail -f outputs/*/experiment.log`

### **Quality Assurance**
- Validate each phase before proceeding
- Check for statistical significance in results
- Monitor for hardware issues or anomalies
- Archive successful configurations for reproduction

## 🔄 Iterative Refinement

Based on initial results, the plan may be refined to:
- Focus on interesting parameter ranges discovered
- Add targeted experiments for unexpected findings
- Optimize resource allocation based on initial timings
- Prioritize most scientifically valuable experiments

This plan provides a systematic approach to understanding GPU-NUMA effects while maintaining experimental rigor and efficient resource utilization.
# GPU NUMA Pipeline Experiments with Hydra

A modern experiment orchestration framework using Facebook Hydra for configuration management, providing enterprise-grade experiment management capabilities.

## Why Hydra?

Hydra offers significant advantages over custom YAML solutions:

### 🚀 **Built-in Power Features**
- **Hierarchical Configuration**: Compose configs from reusable components
- **Command-line Overrides**: No code changes needed for parameter tweaks
- **Multi-run Support**: Built-in parameter sweeps and grid search
- **Automatic Organization**: Time-stamped experiment directories
- **Type Safety**: Schema validation with structured configs

### 🔬 **Research-Grade Features**
- **Experiment Tracking**: Integration with MLflow, W&B, TensorBoard
- **Reproducibility**: Complete parameter logging and code snapshots
- **Job Scheduling**: Integration with SLURM, Ray, and other schedulers
- **Plugin Ecosystem**: Extensible with community plugins

## Quick Start

### 1. Installation
```bash
pip install -r requirements_hydra.txt
```

### 2. Basic Usage
```bash
# Run default experiment (numa_study)
python hydra_experiment_runner.py

# Run different experiment
python hydra_experiment_runner.py experiment=scaling_study

# Quick test
python hydra_experiment_runner.py experiment=quick_test
```

### 3. Command-line Overrides
```bash
# Override specific parameters
python hydra_experiment_runner.py experiment.test.num_samples=2000

# Override multiple parameters
python hydra_experiment_runner.py experiment=numa_study \
  experiment.hardware.gpu_ids=[0,1] \
  experiment.test.compile_model=true

# Override nested configurations
python hydra_experiment_runner.py experiment.vit_configs[0].depth=8
```

### 4. Parameter Sweeps (Multi-run)
```bash
# Sweep over multiple parameters
python hydra_experiment_runner.py -m \
  experiment.test.batch_size=8,16,32 \
  experiment.hardware.numa_nodes=0,2

# Sweep ViT configurations
python hydra_experiment_runner.py -m \
  'experiment.vit_configs[0].depth=4,6,8,12' \
  'experiment.vit_configs[0].dim=256,512,768'

# Complex sweeps with multiple models
python hydra_experiment_runner.py -m \
  experiment=numa_study,scaling_study \
  experiment.hardware.gpu_ids=[0],[1],[2]
```

## Configuration Architecture

### Hierarchical Structure
```
conf/
├── config.yaml                 # Main entry point
├── experiment/                 # Experiment definitions
│   ├── numa_study.yaml
│   ├── scaling_study.yaml
│   └── quick_test.yaml
└── output/                     # Output configurations
    └── base.yaml
```

### Configuration Composition
Hydra automatically composes configurations:
```yaml
# config.yaml
defaults:
  - experiment: numa_study    # Load numa_study.yaml
  - output: base             # Load base output config
  - _self_                   # Apply local overrides last
```

### Type-Safe Configuration
Using dataclasses for validation:
```python
@dataclass
class ViTConfig:
    name: str
    patch_size: int = 32
    depth: int = 6
    heads: int = 8
    dim: int = 512
    mlp_dim: int = 2048
```

## Advanced Features

### 1. Experiment Tracking Integration
```python
# Add to hydra_experiment_runner.py
import mlflow

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_container(cfg))
        runner = HydraExperimentRunner(cfg)
        results = runner.run_all_experiments()
        mlflow.log_metrics({"success_rate": results.success_rate})
```

### 2. Conditional Configuration
```yaml
# numa_study.yaml
experiment:
  name: "numa_study"
  hardware:
    gpu_ids: ${oc.env:CUDA_VISIBLE_DEVICES,[0,1,2,3]}  # From environment
    numa_nodes: ${if_gpu_count_gt_2:[0,2],[0]}          # Conditional logic
```

### 3. Grid Search Templates
```yaml
# conf/experiment/grid_search.yaml
# @package _global_
defaults:
  - override /hydra/launcher: basic  # or joblib, ray, slurm

experiment:
  name: "grid_search"
  
hydra:
  mode: MULTIRUN
  sweep:
    dir: multirun/${experiment.name}/${now:%Y-%m-%d_%H-%M-%S}
```

### 4. Cluster Integration
```yaml
# conf/experiment/slurm_sweep.yaml
defaults:
  - override /hydra/launcher: submitit_slurm

hydra:
  launcher:
    timeout_min: 60
    cpus_per_task: 4
    gpus_per_node: 1
    tasks_per_node: 1
    mem_gb: 32
    nodes: 1
    name: ${hydra.job.name}
```

## Example Workflows

### 1. Development and Testing
```bash
# Quick validation
python hydra_experiment_runner.py experiment=quick_test

# Single parameter test
python hydra_experiment_runner.py experiment=quick_test \
  experiment.vit_configs[0].depth=4
```

### 2. NUMA Locality Study
```bash
# Full NUMA study
python hydra_experiment_runner.py experiment=numa_study

# Focus on specific GPUs
python hydra_experiment_runner.py experiment=numa_study \
  experiment.hardware.gpu_ids=[0,2]

# Compare local vs remote NUMA
python hydra_experiment_runner.py -m \
  experiment=numa_study \
  experiment.hardware.numa_nodes=[0],[2]
```

### 3. Model Scaling Analysis
```bash
# Systematic depth scaling
python hydra_experiment_runner.py -m \
  experiment=scaling_study \
  'experiment.vit_configs[0].depth=2,4,6,8,12,16'

# Dimension scaling
python hydra_experiment_runner.py -m \
  experiment=scaling_study \
  'experiment.vit_configs[0].dim=128,256,512,768,1024'
```

### 4. Compilation Benefits Study
```bash
# Compare compiled vs non-compiled
python hydra_experiment_runner.py -m \
  experiment=scaling_study \
  experiment.test.compile_model=true,false
```

## Output Organization

Hydra automatically organizes outputs:
```
outputs/
├── numa_locality_study/
│   └── 2024-07-11/
│       └── 14-30-15/           # Timestamped run
│           ├── .hydra/         # Hydra metadata
│           ├── experiment.log  # Structured logging
│           ├── results.json    # Experiment results
│           ├── nsys_reports/   # Profiling data
│           └── logs/           # Individual run logs
└── multirun/                   # Multi-run sweeps
    └── scaling_study/
        └── 2024-07-11_15-45-30/
            ├── 0/              # Parameter combination 0
            ├── 1/              # Parameter combination 1
            └── multirun.yaml   # Sweep summary
```

## Integration with Analysis

### 1. Load Results Programmatically
```python
from hydra import compose, initialize
from omegaconf import OmegaConf
import pandas as pd

# Load experiment results
with initialize(config_path="conf"):
    cfg = compose(config_name="config")
    
# Load results from specific run
results_path = "outputs/numa_study/2024-07-11/14-30-15/results.json"
df = pd.read_json(results_path)

# Analyze results
success_by_numa = df.groupby('numa_node')['success'].mean()
```

### 2. Automated Analysis Pipeline
```python
@hydra.main(config_path="conf", config_name="analyze")
def analyze_results(cfg: DictConfig):
    """Automated analysis of experiment results"""
    
    # Load all results from experiment
    results_pattern = f"outputs/{cfg.experiment_name}/**/results.json"
    all_results = []
    
    for results_file in glob.glob(results_pattern, recursive=True):
        df = pd.read_json(results_file)
        all_results.append(df)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Generate analysis plots
    create_performance_plots(combined_df)
    create_numa_analysis(combined_df)
    
    return combined_df
```

## Comparison: Hydra vs Custom YAML

| Feature | Custom YAML | Hydra |
|---------|-------------|-------|
| Configuration Management | Manual parsing | Built-in with validation |
| Command-line Overrides | Not supported | Native support |
| Parameter Sweeps | Manual implementation | Built-in multi-run |
| Experiment Organization | Manual directory creation | Automatic timestamping |
| Type Safety | No validation | Dataclass schemas |
| Reproducibility | Manual logging | Complete parameter tracking |
| Community Support | Custom maintenance | Active ecosystem |
| Learning Curve | Low | Medium |
| Flexibility | High | Very High |
| Integration | Manual | Rich plugin ecosystem |

## Best Practices

### 1. Configuration Design
- Use structured configs (dataclasses) for type safety
- Organize configs hierarchically by functionality
- Use meaningful defaults in base configurations
- Document configuration options

### 2. Experiment Organization
- Use descriptive experiment names
- Group related experiments under same base name
- Leverage Hydra's automatic timestamping
- Archive completed experiment directories

### 3. Parameter Sweeps
- Start with small sweeps for validation
- Use multi-run for systematic parameter exploration
- Leverage job scheduling for large sweeps
- Monitor resource usage during sweeps

### 4. Development Workflow
- Use `quick_test` for rapid iteration
- Validate configurations before large runs
- Use command-line overrides for experimentation
- Commit configurations to version control

## Troubleshooting

### Common Issues
1. **Configuration not found**: Check `config_path` in `@hydra.main`
2. **Override syntax errors**: Use quotes for complex overrides
3. **Multi-run failures**: Check resource availability and timeouts
4. **Permission errors**: Ensure NUMA and nsys access

### Debugging
- Use `--config-path` and `--config-name` for custom configs
- Add `hydra.verbose=true` for detailed logging
- Check `.hydra/config.yaml` in output directory for resolved config
- Use `hydra.job.chdir=false` to stay in original directory

This Hydra-based framework provides enterprise-grade experiment management with minimal code while offering maximum flexibility for systematic GPU-NUMA research.
# How to Run GPU NUMA Experiments with Hydra

This guide shows you how to run GPU-NUMA pipeline experiments using the Hydra-based orchestration framework.

## 🚀 Quick Start

### 1. Initial Setup
```bash
# Navigate to the Hydra experiment directory
cd /sdf/data/lcls/ds/prj/prjcwang31/results/proj-lvm/hydra-orchestration/locality_tests

# Install required dependencies
pip install -r requirements_hydra.txt

# Verify that required tools are available
which nsys numactl nvidia-smi
```

### 2. Test the Framework
```bash
# Run a quick test to validate everything works
python hydra_experiment_runner.py experiment=quick_test
```

**What this does:**
- Tests 3 ViT configurations (no-op, light, medium)
- Uses GPU 0 and NUMA node 0
- Runs 100 samples with short timeout
- Creates timestamped output directory

### 3. Check the Results
```bash
# Look at the output structure
ls -la outputs/

# Check experiment summary
cat outputs/quick_test/*/experiment_summary.json

# View detailed logs
tail outputs/quick_test/*/experiment.log
```

## 📋 Basic Experiment Commands

### Default NUMA Study
```bash
# Run the comprehensive NUMA locality study
python hydra_experiment_runner.py

# Same as:
python hydra_experiment_runner.py experiment=numa_study
```

### Model Scaling Study
```bash
# Systematic model complexity analysis
python hydra_experiment_runner.py experiment=scaling_study
```

### Quick Development Tests
```bash
# Fast validation runs
python hydra_experiment_runner.py experiment=quick_test
```

## 🔧 Command-Line Customization

### Override Basic Parameters
```bash
# Change number of samples
python hydra_experiment_runner.py experiment=quick_test \
  experiment.test.num_samples=200

# Test different GPUs
python hydra_experiment_runner.py experiment=numa_study \
  experiment.hardware.gpu_ids=[0,1,2]

# Enable torch compilation
python hydra_experiment_runner.py experiment=scaling_study \
  experiment.test.compile_model=true

# Increase parallel workers
python hydra_experiment_runner.py experiment=numa_study \
  experiment.max_workers=8
```

### Modify Test Configuration
```bash
# Change batch size and memory
python hydra_experiment_runner.py experiment=numa_study \
  experiment.test.batch_size=20 \
  experiment.test.memory_size_mb=1024

# Adjust timeouts for large models
python hydra_experiment_runner.py experiment=scaling_study \
  experiment.test.timeout_s=1200

# Test different image sizes
python hydra_experiment_runner.py experiment=quick_test \
  experiment.test.tensor_shape=[3,512,512]
```

### Hardware Configuration
```bash
# Test specific GPU-NUMA combinations
python hydra_experiment_runner.py experiment=numa_study \
  experiment.hardware.gpu_ids=[0] \
  experiment.hardware.numa_nodes=[0,2]

# Single GPU comprehensive test
python hydra_experiment_runner.py experiment=numa_study \
  experiment.hardware.gpu_ids=[3] \
  experiment.hardware.numa_nodes=[0,1,2,3]
```

## 🔄 Parameter Sweeps (Multi-run)

### Basic Sweeps
```bash
# Test multiple batch sizes
python hydra_experiment_runner.py -m \
  experiment=quick_test \
  experiment.test.batch_size=5,10,20

# Compare NUMA nodes
python hydra_experiment_runner.py -m \
  experiment=numa_study \
  experiment.hardware.numa_nodes=[0],[2],[3]

# Test with and without compilation
python hydra_experiment_runner.py -m \
  experiment=scaling_study \
  experiment.test.compile_model=true,false
```

### Advanced Sweeps
```bash
# Multi-dimensional parameter sweep
python hydra_experiment_runner.py -m \
  experiment=numa_study \
  experiment.test.batch_size=8,16 \
  experiment.hardware.gpu_ids=[0],[1] \
  experiment.hardware.numa_nodes=[0],[2]

# ViT configuration sweep (modify specific ViT configs)
python hydra_experiment_runner.py -m \
  experiment=scaling_study \
  'experiment.vit_configs[0].depth=4,6,8,12'

# Complex multi-experiment sweep
python hydra_experiment_runner.py -m \
  experiment=numa_study,scaling_study \
  experiment.test.num_samples=500,1000
```

## 🔍 Checking Configurations

### Preview Configuration
```bash
# See resolved configuration without running
python hydra_experiment_runner.py --cfg job experiment=numa_study

# Check what multi-run will execute
python hydra_experiment_runner.py -m --cfg job \
  experiment.test.batch_size=8,16 \
  experiment.hardware.gpu_ids=[0],[1]

# Print configuration in different formats
python hydra_experiment_runner.py --cfg job --package=experiment experiment=scaling_study
```

### List Available Configurations
```bash
# See available experiment configurations
ls conf/experiment/

# View specific configuration file
cat conf/experiment/numa_study.yaml
```

## 📊 Understanding Output Structure

### Automatic Organization
Every run creates a timestamped directory:
```
outputs/
├── numa_locality_study/           # Experiment name
│   └── 2024-07-11/               # Date
│       └── 15-30-45/             # Time (HH-MM-SS)
│           ├── .hydra/           # Hydra metadata
│           │   ├── config.yaml   # Resolved configuration
│           │   ├── hydra.yaml    # Hydra settings
│           │   └── overrides.yaml # Command-line overrides
│           ├── experiment.log    # Full experiment log
│           ├── results.json      # Detailed run results
│           ├── experiment_summary.json  # Summary statistics
│           ├── nsys_reports/     # NVIDIA nsys profiling files
│           │   ├── gpu0_numa0_noop.nsys-rep
│           │   └── gpu0_numa0_light_small.nsys-rep
│           └── logs/             # Individual run logs
│               ├── gpu0_numa0_noop.stdout
│               ├── gpu0_numa0_noop.stderr
│               └── ...
```

### Multi-run Output
Parameter sweeps create multiple subdirectories:
```
multirun/
└── numa_locality_study/
    └── 2024-07-11_16-15-30/      # Sweep timestamp
        ├── 0/                    # Parameter combination 0
        │   ├── .hydra/
        │   ├── results.json
        │   └── ...
        ├── 1/                    # Parameter combination 1
        ├── 2/                    # Parameter combination 2
        └── multirun.yaml         # Sweep summary
```

## 📈 Monitoring and Analysis

### Real-time Monitoring
```bash
# Watch experiment progress
tail -f outputs/numa_locality_study/*/experiment.log

# Monitor specific run logs
tail -f outputs/numa_locality_study/*/logs/gpu0_numa0_medium_balanced.stdout

# Check system resources during runs
htop
nvidia-smi -l 1
```

### Result Analysis
```bash
# Quick summary
cat outputs/numa_locality_study/*/experiment_summary.json

# Detailed results
jq '.[] | select(.success==true) | {run_id, duration, gpu_id, numa_node}' \
  outputs/numa_locality_study/*/results.json

# Count successful vs failed runs
jq '[.[] | .success] | group_by(.) | map({success: .[0], count: length})' \
  outputs/numa_locality_study/*/results.json
```

## 🛠️ Development and Debugging

### Quick Iteration
```bash
# Minimal test for rapid development
python hydra_experiment_runner.py experiment=quick_test \
  experiment.test.num_samples=20 \
  experiment.test.timeout_s=60

# Test single ViT configuration
python hydra_experiment_runner.py experiment=quick_test \
  experiment.vit_configs=[{name:test,patch_size:32,depth:2,heads:4,dim:256,mlp_dim:1024}]
```

### Debugging
```bash
# Enable verbose Hydra logging
python hydra_experiment_runner.py experiment=quick_test \
  hydra.verbose=true

# Stay in original directory (don't change to output dir)
python hydra_experiment_runner.py experiment=quick_test \
  hydra.job.chdir=false

# Dry run - check configuration only
python hydra_experiment_runner.py --cfg job experiment=numa_study
```

### Configuration Validation
```bash
# Test configuration loading
python -c "
from hydra import compose, initialize
with initialize(config_path='conf'):
    cfg = compose(config_name='config', overrides=['experiment=numa_study'])
    print('Configuration loaded successfully')
    print(f'Experiment: {cfg.experiment.name}')
    print(f'ViT configs: {len(cfg.experiment.vit_configs)}')
"
```

## ⚡ Performance Tips

### Optimal Resource Usage
```bash
# Match workers to available cores
python hydra_experiment_runner.py experiment=numa_study \
  experiment.max_workers=$(nproc)

# Reduce batch size for large models to avoid OOM
python hydra_experiment_runner.py experiment=scaling_study \
  experiment.test.batch_size=5

# Increase timeout for complex models
python hydra_experiment_runner.py experiment=scaling_study \
  experiment.test.timeout_s=1800
```

### Efficient Sweeps
```bash
# Start with small parameter ranges
python hydra_experiment_runner.py -m \
  experiment=quick_test \
  experiment.test.batch_size=5,10

# Then expand to full ranges
python hydra_experiment_runner.py -m \
  experiment=numa_study \
  experiment.test.batch_size=5,10,15,20
```

## 🚨 Troubleshooting

### Common Issues

**Permission Errors:**
```bash
# Check NUMA access
numactl --hardware

# Check nsys permissions
nsys --version
```

**Configuration Errors:**
```bash
# Validate configuration syntax
python hydra_experiment_runner.py --cfg job experiment=numa_study

# Check for typos in overrides
python hydra_experiment_runner.py experiment=numa_study \
  experiment.test.num_samples=1000  # Note: no spaces around =
```

**CUDA/GPU Issues:**
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA in PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
```

### Recovery from Failed Runs
```bash
# Hydra automatically skips completed runs on restart (if resume=true)
# Just re-run the same command
python hydra_experiment_runner.py experiment=numa_study

# Or force restart from beginning
python hydra_experiment_runner.py experiment=numa_study \
  experiment.resume=false
```

## 🎯 Recommended Workflow

1. **Start Small**: Always test with `experiment=quick_test` first
2. **Validate Config**: Use `--cfg job` to check configurations
3. **Single Parameter**: Test individual parameter changes
4. **Small Sweeps**: Start with 2-3 parameter values
5. **Scale Up**: Expand to full parameter ranges
6. **Monitor**: Watch logs and system resources
7. **Analyze**: Check results and summaries

## 📚 Next Steps

- Explore the configuration files in `conf/` to understand structure
- Create custom experiment configurations by copying existing ones
- Integrate with analysis scripts to process results
- Set up automated plotting and reporting from output data

The Hydra framework handles all the complexity of experiment orchestration, letting you focus on the research questions!
#!/usr/bin/env python3
"""
GPU NUMA Experiment Runner with Hydra Configuration Management

A modern experiment orchestration framework using Facebook Hydra for configuration
management, providing powerful features like:
- Hierarchical configuration composition
- Command-line overrides
- Multi-run support with parameter sweeps
- Automatic experiment logging and organization
- Integration with MLflow, Weights & Biases, etc.

Features over custom YAML approach:
- Industry-standard configuration management
- Built-in parameter sweeps and grid search
- Automatic experiment directory creation
- Command-line override capabilities
- Extensible plugin system
- Integration with experiment tracking tools

Usage:
    # Basic run
    python hydra_experiment_runner.py

    # Override configuration
    python hydra_experiment_runner.py experiment=numa_study gpu_ids=[0,1] 
    
    # Parameter sweep
    python hydra_experiment_runner.py -m vit.depth=4,6,8 numa_nodes=0,2
    
    # Custom experiment
    python hydra_experiment_runner.py +experiment=custom vit.patch_size=16
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import hydra
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import pandas as pd


# Configuration schemas using dataclasses for type safety
@dataclass
class ViTConfig:
    name: str
    patch_size: int = 32
    depth: int = 6
    heads: int = 8
    dim: int = 512
    mlp_dim: int = 2048


@dataclass 
class TestConfig:
    num_samples: int = 1000
    batch_size: int = 10
    warmup_samples: int = 100
    memory_size_mb: int = 512
    tensor_shape: List[int] = None
    compile_model: bool = False
    timeout_s: int = 600
    
    def __post_init__(self):
        if self.tensor_shape is None:
            self.tensor_shape = [3, 224, 224]


@dataclass
class HardwareConfig:
    gpu_ids: List[int] = None
    numa_nodes: List[int] = None
    
    def __post_init__(self):
        if self.gpu_ids is None:
            self.gpu_ids = [0]
        if self.numa_nodes is None:
            self.numa_nodes = [0]


@dataclass
class ExperimentConfig:
    name: str
    description: str
    test: TestConfig
    hardware: HardwareConfig
    vit_configs: List[ViTConfig]
    max_workers: int = 4
    resume: bool = True


@dataclass
class OutputConfig:
    base_dir: str = "outputs"
    nsys_reports_dir: str = "nsys_reports"
    logs_dir: str = "logs"
    results_file: str = "results.json"


@dataclass
class Config:
    experiment: ExperimentConfig
    output: OutputConfig
    defaults: List[Any] = None


class HydraExperimentRunner:
    """Hydra-powered experiment orchestration with advanced features"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        
        # Setup output directories (Hydra handles this automatically)
        self.output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.nsys_dir = self.output_dir / cfg.output.nsys_reports_dir
        self.logs_dir = self.output_dir / cfg.output.logs_dir
        
        # Create directories
        self.nsys_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Results tracking
        self.results = []
        self.results_file = self.output_dir / cfg.output.results_file
        
        self.logger.info(f"Experiment output directory: {self.output_dir}")
        self.logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    def generate_run_configurations(self) -> List[Dict[str, Any]]:
        """Generate all experiment run configurations"""
        runs = []
        
        exp_cfg = self.cfg.experiment
        
        for gpu_id in exp_cfg.hardware.gpu_ids:
            for numa_node in exp_cfg.hardware.numa_nodes:
                for vit_cfg in exp_cfg.vit_configs:
                    run_config = {
                        'gpu_id': gpu_id,
                        'numa_node': numa_node,
                        'vit_config': OmegaConf.to_container(vit_cfg),
                        'test_config': OmegaConf.to_container(exp_cfg.test),
                        'run_id': self._generate_run_id(gpu_id, numa_node, vit_cfg)
                    }
                    runs.append(run_config)
        
        return runs
    
    def _generate_run_id(self, gpu_id: int, numa_node: int, vit_cfg: ViTConfig) -> str:
        """Generate unique run identifier"""
        vit_name = vit_cfg.name if vit_cfg.name else f"p{vit_cfg.patch_size}_d{vit_cfg.depth}_dim{vit_cfg.dim}"
        run_id = f"gpu{gpu_id}_numa{numa_node}_{vit_name}"
        
        if self.cfg.experiment.test.compile_model:
            run_id += "_compiled"
        
        return run_id
    
    def execute_single_run(self, run_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single experimental run"""
        run_id = run_config['run_id']
        self.logger.info(f"Starting run: {run_id}")
        
        start_time = time.time()
        
        try:
            # Build command for gpu_numa_pipeline_test.py
            cmd = self._build_command(run_config)
            
            # Setup logging files
            stdout_log = self.logs_dir / f"{run_id}.stdout"
            stderr_log = self.logs_dir / f"{run_id}.stderr"
            nsys_output = self.nsys_dir / f"{run_id}"
            
            # Execute with nsys profiling and NUMA binding
            result = self._execute_command(cmd, stdout_log, stderr_log, nsys_output, run_config)
            
            duration = time.time() - start_time
            
            # Prepare result record
            result_record = {
                'run_id': run_id,
                'gpu_id': run_config['gpu_id'],
                'numa_node': run_config['numa_node'],
                'vit_config': run_config['vit_config'],
                'test_config': run_config['test_config'],
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'duration': duration,
                'success': result.returncode == 0,
                'exit_code': result.returncode,
                'stdout_log': str(stdout_log),
                'stderr_log': str(stderr_log),
                'nsys_report': str(nsys_output.with_suffix('.nsys-rep')) if result.returncode == 0 else None
            }
            
            if result.returncode == 0:
                self.logger.info(f"Completed run: {run_id} (duration: {duration:.1f}s)")
            else:
                self.logger.error(f"Failed run: {run_id} (exit code: {result.returncode})")
                # Read error message
                try:
                    with open(stderr_log, 'r') as f:
                        result_record['error'] = f.read()
                except:
                    result_record['error'] = "Unknown error"
            
            return result_record
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Exception in run {run_id}: {e}")
            
            return {
                'run_id': run_id,
                'gpu_id': run_config['gpu_id'],
                'numa_node': run_config['numa_node'],
                'vit_config': run_config['vit_config'],
                'test_config': run_config['test_config'],
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'duration': duration,
                'success': False,
                'exit_code': -1,
                'error': str(e)
            }
    
    def _build_command(self, run_config: Dict[str, Any]) -> List[str]:
        """Build command line for gpu_numa_pipeline_test.py"""
        vit_cfg = run_config['vit_config']
        test_cfg = run_config['test_config']
        
        cmd = [
            "python3", "gpu_numa_pipeline_test.py",
            "--gpu-id", str(run_config['gpu_id']),
            "--shape"] + [str(x) for x in test_cfg['tensor_shape']] + [
            "--num-samples", str(test_cfg['num_samples']),
            "--batch-size", str(test_cfg['batch_size']),
            "--warmup-samples", str(test_cfg['warmup_samples']),
            "--memory-size-mb", str(test_cfg['memory_size_mb']),
            "--vit-patch-size", str(vit_cfg['patch_size']),
            "--vit-depth", str(vit_cfg['depth']),
            "--vit-heads", str(vit_cfg['heads']),
            "--vit-dim", str(vit_cfg['dim']),
            "--vit-mlp-dim", str(vit_cfg['mlp_dim'])
        ]
        
        if test_cfg.get('compile_model', False):
            cmd.append("--compile-model")
        
        return cmd
    
    def _execute_command(self, cmd: List[str], stdout_log: Path, stderr_log: Path, 
                        nsys_output: Path, run_config: Dict[str, Any]) -> subprocess.CompletedProcess:
        """Execute command with NUMA binding and nsys profiling"""
        
        # NUMA binding
        numa_cmd = [
            "numactl", 
            f"--cpunodebind={run_config['numa_node']}", 
            f"--membind={run_config['numa_node']}"
        ]
        
        # nsys profiling
        timeout_s = run_config['test_config']['timeout_s']
        nsys_cmd = [
            "timeout", str(timeout_s),
            "nsys", "profile",
            "-o", str(nsys_output),
            "--trace=cuda,nvtx"
        ]
        
        full_cmd = nsys_cmd + numa_cmd + cmd
        
        # Execute
        with open(stdout_log, 'w') as stdout_f, open(stderr_log, 'w') as stderr_f:
            result = subprocess.run(
                full_cmd,
                stdout=stdout_f,
                stderr=stderr_f,
                cwd=Path.cwd()
            )
        
        return result
    
    def run_all_experiments(self):
        """Execute all experiments with parallel execution"""
        run_configs = self.generate_run_configurations()
        
        self.logger.info(f"Generated {len(run_configs)} experiment runs")
        self.logger.info(f"Using {self.cfg.experiment.max_workers} parallel workers")
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.cfg.experiment.max_workers) as executor:
            # Submit all jobs
            future_to_config = {
                executor.submit(self.execute_single_run, config): config 
                for config in run_configs
            }
            
            completed_count = 0
            failed_count = 0
            
            # Collect results as they complete
            for future in as_completed(future_to_config):
                result = future.result()
                self.results.append(result)
                
                if result['success']:
                    completed_count += 1
                else:
                    failed_count += 1
                
                # Progress update
                total_done = completed_count + failed_count
                self.logger.info(f"Progress: {total_done}/{len(run_configs)} "
                               f"(completed: {completed_count}, failed: {failed_count})")
                
                # Save intermediate results
                self._save_results()
        
        # Final summary
        self.logger.info(f"Experiment completed: {completed_count} successful, {failed_count} failed")
        self._generate_summary()
    
    def _save_results(self):
        """Save results to JSON file"""
        import json
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def _generate_summary(self):
        """Generate experiment summary"""
        if not self.results:
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.results)
        
        # Basic statistics
        success_rate = df['success'].mean() * 100
        avg_duration = df[df['success']]['duration'].mean() if df['success'].any() else 0
        
        summary = {
            'experiment_name': self.cfg.experiment.name,
            'total_runs': len(df),
            'successful_runs': df['success'].sum(),
            'failed_runs': (~df['success']).sum(),
            'success_rate': success_rate,
            'average_duration': avg_duration,
            'total_duration': df['duration'].sum(),
            'gpu_ids_tested': sorted(df['gpu_id'].unique().tolist()),
            'numa_nodes_tested': sorted(df['numa_node'].unique().tolist()),
            'vit_configs_tested': len(df['vit_config'].apply(str).unique())
        }
        
        # Save summary
        summary_file = self.output_dir / "experiment_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("=" * 60)
        for key, value in summary.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info(f"Results saved to: {self.results_file}")
        self.logger.info(f"Summary saved to: {summary_file}")


# Register configuration schemas with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="experiment", name="base", node=ExperimentConfig)
cs.store(group="output", name="base", node=OutputConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration management"""
    
    # Initialize experiment runner
    runner = HydraExperimentRunner(cfg)
    
    # Run all experiments
    runner.run_all_experiments()
    
    return runner.results  # Return for Hydra multi-run aggregation


if __name__ == "__main__":
    main()
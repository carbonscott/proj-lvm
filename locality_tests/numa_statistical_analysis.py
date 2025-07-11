#!/usr/bin/env python3
"""
Comprehensive NUMA Locality Statistical Analysis

Analyzes detailed nsys performance data to quantify NUMA locality effects
across different GPU-NUMA combinations and model complexities.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class NUMALocalityAnalyzer:
    def __init__(self, csv_dir: str):
        self.csv_dir = Path(csv_dir)
        self.data = {}
        self.results = {}
        
    def load_data(self):
        """Load all CSV performance data."""
        csv_files = {
            'memory': 'memory_transfer_performance.csv',
            'compute': 'compute_performance.csv', 
            'pipeline': 'pipeline_analysis.csv',
            'temporal': 'temporal_compression_analysis.csv'
        }
        
        for name, filename in csv_files.items():
            file_path = self.csv_dir / filename
            if file_path.exists():
                self.data[name] = pd.read_csv(file_path)
                print(f"Loaded {len(self.data[name])} records from {filename}")
            else:
                print(f"Warning: {filename} not found")
    
    def analyze_memory_bandwidth_numa_effects(self) -> Dict[str, Any]:
        """Analyze H2D and D2H bandwidth differences between NUMA nodes."""
        if 'memory' not in self.data:
            return {'error': 'Memory data not available'}
        
        df = self.data['memory'].copy()
        
        # Group by test_type and calculate NUMA effects
        results = {
            'by_model': {},
            'overall': {},
            'statistical_tests': {}
        }
        
        # Analyze by model type
        model_types = df['test_type'].unique()
        
        for model in model_types:
            model_data = df[df['test_type'] == model]
            
            # Calculate NUMA 0 vs NUMA 2 effects for H2D and D2H
            numa0_data = model_data[model_data['numa'] == 0]
            numa2_data = model_data[model_data['numa'] == 2]
            
            if len(numa0_data) > 0 and len(numa2_data) > 0:
                # H2D analysis
                h2d_numa0_mean = numa0_data['h2d_bandwidth_mbps'].mean()
                h2d_numa2_mean = numa2_data['h2d_bandwidth_mbps'].mean()
                h2d_effect = ((h2d_numa0_mean - h2d_numa2_mean) / h2d_numa2_mean) * 100
                
                # D2H analysis
                d2h_numa0_mean = numa0_data['d2h_bandwidth_mbps'].mean()
                d2h_numa2_mean = numa2_data['d2h_bandwidth_mbps'].mean()
                d2h_effect = ((d2h_numa0_mean - d2h_numa2_mean) / d2h_numa2_mean) * 100
                
                # Statistical tests
                h2d_ttest = stats.ttest_ind(numa0_data['h2d_bandwidth_mbps'], 
                                          numa2_data['h2d_bandwidth_mbps'])
                d2h_ttest = stats.ttest_ind(numa0_data['d2h_bandwidth_mbps'], 
                                          numa2_data['d2h_bandwidth_mbps'])
                
                results['by_model'][model] = {
                    'h2d_numa0_mean': h2d_numa0_mean,
                    'h2d_numa2_mean': h2d_numa2_mean,
                    'h2d_locality_effect_pct': h2d_effect,
                    'd2h_numa0_mean': d2h_numa0_mean,
                    'd2h_numa2_mean': d2h_numa2_mean,
                    'd2h_locality_effect_pct': d2h_effect,
                    'h2d_ttest_pvalue': h2d_ttest.pvalue,
                    'd2h_ttest_pvalue': d2h_ttest.pvalue,
                    'sample_size_numa0': len(numa0_data),
                    'sample_size_numa2': len(numa2_data)
                }
        
        # Overall analysis across all models
        numa0_all = df[df['numa'] == 0]
        numa2_all = df[df['numa'] == 2]
        
        if len(numa0_all) > 0 and len(numa2_all) > 0:
            h2d_overall_effect = ((numa0_all['h2d_bandwidth_mbps'].mean() - 
                                 numa2_all['h2d_bandwidth_mbps'].mean()) / 
                                numa2_all['h2d_bandwidth_mbps'].mean()) * 100
            
            d2h_overall_effect = ((numa0_all['d2h_bandwidth_mbps'].mean() - 
                                 numa2_all['d2h_bandwidth_mbps'].mean()) / 
                                numa2_all['d2h_bandwidth_mbps'].mean()) * 100
            
            results['overall'] = {
                'h2d_locality_effect_pct': h2d_overall_effect,
                'd2h_locality_effect_pct': d2h_overall_effect,
                'total_experiments': len(df)
            }
        
        return results
    
    def analyze_compute_performance_numa_effects(self) -> Dict[str, Any]:
        """Analyze compute performance differences between NUMA nodes."""
        if 'compute' not in self.data:
            return {'error': 'Compute data not available'}
        
        df = self.data['compute'].copy()
        
        # Filter out NOOP tests (no meaningful compute)
        df = df[df['test_type'] != 'NOOP (Memory Only)']
        
        results = {
            'by_model': {},
            'overall': {}
        }
        
        # Key compute metrics to analyze
        compute_metrics = [
            'total_kernels',
            'total_compute_ms', 
            'avg_kernel_ms',
            'compute_utilization'
        ]
        
        # Analyze by model type
        model_types = df['test_type'].unique()
        
        for model in model_types:
            model_data = df[df['test_type'] == model]
            numa0_data = model_data[model_data['numa'] == 0]
            numa2_data = model_data[model_data['numa'] == 2]
            
            if len(numa0_data) > 0 and len(numa2_data) > 0:
                model_results = {}
                
                for metric in compute_metrics:
                    if metric in model_data.columns:
                        numa0_mean = numa0_data[metric].mean()
                        numa2_mean = numa2_data[metric].mean()
                        
                        if numa2_mean != 0:
                            effect_pct = ((numa0_mean - numa2_mean) / numa2_mean) * 100
                        else:
                            effect_pct = 0
                        
                        model_results[metric] = {
                            'numa0_mean': numa0_mean,
                            'numa2_mean': numa2_mean,
                            'locality_effect_pct': effect_pct
                        }
                
                results['by_model'][model] = model_results
        
        return results
    
    def analyze_pipeline_efficiency_numa_effects(self) -> Dict[str, Any]:
        """Analyze pipeline efficiency differences between NUMA nodes."""
        if 'pipeline' not in self.data:
            return {'error': 'Pipeline data not available'}
        
        df = self.data['pipeline'].copy()
        
        results = {
            'by_model': {},
            'overall': {}
        }
        
        # Key pipeline metrics
        pipeline_metrics = [
            'avg_prep_us',
            'pipeline_efficiency', 
            'significant_gaps',
            'total_gap_ms'
        ]
        
        # Analyze by model type
        model_types = df['test_type'].unique()
        
        for model in model_types:
            model_data = df[df['test_type'] == model]
            numa0_data = model_data[model_data['numa'] == 0]
            numa2_data = model_data[model_data['numa'] == 2]
            
            if len(numa0_data) > 0 and len(numa2_data) > 0:
                model_results = {}
                
                for metric in pipeline_metrics:
                    if metric in model_data.columns:
                        numa0_mean = numa0_data[metric].mean()
                        numa2_mean = numa2_data[metric].mean()
                        
                        if numa2_mean != 0:
                            effect_pct = ((numa0_mean - numa2_mean) / numa2_mean) * 100
                        else:
                            effect_pct = 0
                        
                        model_results[metric] = {
                            'numa0_mean': numa0_mean,
                            'numa2_mean': numa2_mean,
                            'locality_effect_pct': effect_pct
                        }
                
                results['by_model'][model] = model_results
        
        return results
    
    def analyze_complexity_vs_numa_sensitivity(self) -> Dict[str, Any]:
        """Analyze how model complexity affects NUMA sensitivity."""
        if 'memory' not in self.data:
            return {'error': 'Memory data not available'}
        
        df = self.data['memory'].copy()
        
        # Define complexity ordering
        complexity_order = [
            'NOOP (Memory Only)',
            'Light Small ViT', 
            'Light Medium ViT',
            'Medium Balanced ViT',
            'Medium Compute ViT',
            'Heavy Large ViT'
        ]
        
        complexity_analysis = []
        
        for model in complexity_order:
            model_data = df[df['test_type'] == model]
            
            if len(model_data) > 0:
                numa0_data = model_data[model_data['numa'] == 0]
                numa2_data = model_data[model_data['numa'] == 2]
                
                if len(numa0_data) > 0 and len(numa2_data) > 0:
                    # Calculate average bandwidth difference
                    h2d_effect = ((numa0_data['h2d_bandwidth_mbps'].mean() - 
                                 numa2_data['h2d_bandwidth_mbps'].mean()) / 
                                numa2_data['h2d_bandwidth_mbps'].mean()) * 100
                    
                    d2h_effect = ((numa0_data['d2h_bandwidth_mbps'].mean() - 
                                 numa2_data['d2h_bandwidth_mbps'].mean()) / 
                                numa2_data['d2h_bandwidth_mbps'].mean()) * 100
                    
                    complexity_analysis.append({
                        'model': model,
                        'complexity_rank': complexity_order.index(model),
                        'h2d_numa_effect_pct': h2d_effect,
                        'd2h_numa_effect_pct': d2h_effect,
                        'avg_numa_effect_pct': (h2d_effect + d2h_effect) / 2
                    })
        
        return {
            'complexity_analysis': complexity_analysis,
            'correlation': self._calculate_complexity_correlation(complexity_analysis)
        }
    
    def _calculate_complexity_correlation(self, complexity_data: List[Dict]) -> Dict[str, float]:
        """Calculate correlation between complexity and NUMA sensitivity."""
        if len(complexity_data) < 3:
            return {'error': 'Insufficient data for correlation'}
        
        df = pd.DataFrame(complexity_data)
        
        correlations = {}
        for effect_col in ['h2d_numa_effect_pct', 'd2h_numa_effect_pct', 'avg_numa_effect_pct']:
            corr, p_value = stats.pearsonr(df['complexity_rank'], df[effect_col])
            correlations[effect_col] = {
                'correlation': corr,
                'p_value': p_value
            }
        
        return correlations
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        summary = {
            'data_overview': {},
            'key_findings': {},
            'statistical_significance': {}
        }
        
        # Data overview
        if 'memory' in self.data:
            mem_df = self.data['memory']
            summary['data_overview'] = {
                'total_experiments': len(mem_df),
                'gpus_tested': sorted(mem_df['gpu'].unique().tolist()),
                'numa_nodes_tested': sorted(mem_df['numa'].unique().tolist()),
                'model_types_tested': len(mem_df['test_type'].unique()),
                'unique_configurations': len(mem_df[['gpu', 'numa', 'test_type']].drop_duplicates())
            }
        
        return summary
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run all NUMA locality analyses."""
        print("Loading performance data...")
        self.load_data()
        
        print("Analyzing memory bandwidth NUMA effects...")
        memory_results = self.analyze_memory_bandwidth_numa_effects()
        
        print("Analyzing compute performance NUMA effects...")
        compute_results = self.analyze_compute_performance_numa_effects()
        
        print("Analyzing pipeline efficiency NUMA effects...")
        pipeline_results = self.analyze_pipeline_efficiency_numa_effects()
        
        print("Analyzing complexity vs NUMA sensitivity...")
        complexity_results = self.analyze_complexity_vs_numa_sensitivity()
        
        print("Generating summary statistics...")
        summary = self.generate_summary_statistics()
        
        return {
            'memory_analysis': memory_results,
            'compute_analysis': compute_results,
            'pipeline_analysis': pipeline_results,
            'complexity_analysis': complexity_results,
            'summary': summary
        }


def write_detailed_report(results: Dict[str, Any], output_file: Path):
    """Write comprehensive analysis report."""
    with open(output_file, 'w') as f:
        f.write("NUMA Locality Performance Analysis - Detailed Report\n")
        f.write("=" * 60 + "\n\n")
        
        # Summary overview
        if 'summary' in results and 'data_overview' in results['summary']:
            overview = results['summary']['data_overview']
            f.write("EXPERIMENT OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total experiments: {overview.get('total_experiments', 'N/A')}\n")
            f.write(f"GPUs tested: {overview.get('gpus_tested', 'N/A')}\n")
            f.write(f"NUMA nodes tested: {overview.get('numa_nodes_tested', 'N/A')}\n")
            f.write(f"Model types: {overview.get('model_types_tested', 'N/A')}\n")
            f.write(f"Unique configurations: {overview.get('unique_configurations', 'N/A')}\n\n")
        
        # Memory bandwidth analysis
        if 'memory_analysis' in results and 'by_model' in results['memory_analysis']:
            f.write("MEMORY BANDWIDTH NUMA LOCALITY EFFECTS\n")
            f.write("-" * 40 + "\n")
            
            memory_data = results['memory_analysis']['by_model']
            
            # Sort by locality effect (strongest first)
            sorted_models = sorted(memory_data.items(), 
                                 key=lambda x: abs(x[1].get('h2d_locality_effect_pct', 0)), 
                                 reverse=True)
            
            for model, data in sorted_models:
                f.write(f"\n{model}:\n")
                f.write(f"  H2D Bandwidth:\n")
                f.write(f"    NUMA 0 (local): {data['h2d_numa0_mean']:.1f} MB/s\n")
                f.write(f"    NUMA 2 (remote): {data['h2d_numa2_mean']:.1f} MB/s\n")
                f.write(f"    Locality Effect: {data['h2d_locality_effect_pct']:+.2f}%\n")
                f.write(f"    Statistical significance: p={data['h2d_ttest_pvalue']:.4f}\n")
                
                f.write(f"  D2H Bandwidth:\n")
                f.write(f"    NUMA 0 (local): {data['d2h_numa0_mean']:.1f} MB/s\n")
                f.write(f"    NUMA 2 (remote): {data['d2h_numa2_mean']:.1f} MB/s\n")
                f.write(f"    Locality Effect: {data['d2h_locality_effect_pct']:+.2f}%\n")
                f.write(f"    Statistical significance: p={data['d2h_ttest_pvalue']:.4f}\n")
            
            # Overall effect
            if 'overall' in results['memory_analysis']:
                overall = results['memory_analysis']['overall']
                f.write(f"\nOVERALL MEMORY BANDWIDTH EFFECTS:\n")
                f.write(f"  H2D Overall Locality Effect: {overall['h2d_locality_effect_pct']:+.2f}%\n")
                f.write(f"  D2H Overall Locality Effect: {overall['d2h_locality_effect_pct']:+.2f}%\n")
        
        # Complexity vs NUMA sensitivity analysis
        if 'complexity_analysis' in results and 'complexity_analysis' in results['complexity_analysis']:
            f.write(f"\n\nCOMPLEXITY vs NUMA SENSITIVITY ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            complexity_data = results['complexity_analysis']['complexity_analysis']
            
            f.write("Model Complexity → NUMA Sensitivity:\n")
            for item in complexity_data:
                f.write(f"  {item['model']}: {item['avg_numa_effect_pct']:+.2f}% average effect\n")
            
            # Correlation analysis
            if 'correlation' in results['complexity_analysis']:
                corr_data = results['complexity_analysis']['correlation']
                f.write(f"\nComplexity-Sensitivity Correlations:\n")
                for metric, stats in corr_data.items():
                    if 'correlation' in stats:
                        f.write(f"  {metric}: r={stats['correlation']:.3f}, p={stats['p_value']:.4f}\n")
        
        # Compute performance analysis
        if 'compute_analysis' in results and 'by_model' in results['compute_analysis']:
            f.write(f"\n\nCOMPUTE PERFORMANCE NUMA EFFECTS\n")
            f.write("-" * 35 + "\n")
            
            compute_data = results['compute_analysis']['by_model']
            
            for model, metrics in compute_data.items():
                f.write(f"\n{model}:\n")
                for metric, data in metrics.items():
                    f.write(f"  {metric}: {data['locality_effect_pct']:+.2f}% effect\n")
        
        # Pipeline efficiency analysis
        if 'pipeline_analysis' in results and 'by_model' in results['pipeline_analysis']:
            f.write(f"\n\nPIPELINE EFFICIENCY NUMA EFFECTS\n")
            f.write("-" * 35 + "\n")
            
            pipeline_data = results['pipeline_analysis']['by_model']
            
            for model, metrics in pipeline_data.items():
                f.write(f"\n{model}:\n")
                for metric, data in metrics.items():
                    f.write(f"  {metric}: {data['locality_effect_pct']:+.2f}% effect\n")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive NUMA locality statistical analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python numa_statistical_analysis.py analysis_reports/numa_nsys_csv_tables/
  python numa_statistical_analysis.py csv_data/ -o numa_analysis_results/
        """
    )
    
    parser.add_argument('csv_dir', help='Directory containing CSV performance data')
    parser.add_argument('-o', '--output-dir', default='numa_analysis_results',
                       help='Output directory for analysis results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    analyzer = NUMALocalityAnalyzer(args.csv_dir)
    results = analyzer.run_complete_analysis()
    
    # Write detailed report
    report_file = output_dir / 'numa_locality_detailed_analysis.txt'
    write_detailed_report(results, report_file)
    print(f"Detailed analysis report written to: {report_file}")
    
    # Save raw results as JSON for further processing
    import json
    json_file = output_dir / 'numa_analysis_raw_results.json'
    
    # Convert numpy types to native Python for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Recursively convert numpy types
    def clean_for_json(data):
        if isinstance(data, dict):
            return {k: clean_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [clean_for_json(v) for v in data]
        else:
            return convert_numpy(data)
    
    clean_results = clean_for_json(results)
    
    with open(json_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    print(f"Raw analysis results saved to: {json_file}")
    
    print(f"\nAnalysis completed. Results saved in: {output_dir}")


if __name__ == '__main__':
    main()
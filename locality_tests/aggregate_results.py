#!/usr/bin/env python3
"""
Convert performance analysis text files to CSV tables.

This script parses the output from analyze_latency.py and converts
the performance data into structured CSV tables for analysis.
"""

import argparse
import csv
import glob
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


class PerformanceAnalysisParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.config_name = Path(file_path).stem.replace('_performance_analysis', '')
        self.data = {}

    def parse_file(self) -> Dict[str, Any]:
        """Parse the performance analysis text file."""
        with open(self.file_path, 'r') as f:
            content = f.read()

        # Parse configuration
        self.data['config'] = self._parse_configuration(content)

        # Parse each performance section
        self.data['memory'] = self._parse_memory_transfers(content)
        self.data['compute'] = self._parse_compute_performance(content)
        self.data['pipeline'] = self._parse_pipeline_analysis(content)
        self.data['temporal'] = self._parse_temporal_compression(content)

        return self.data

    def _parse_configuration(self, content: str) -> Dict[str, Any]:
        """Parse the CONFIGURATION section."""
        config = {'config_name': self.config_name}

        config_section = self._extract_section(content, 'CONFIGURATION')
        if not config_section:
            return config

        # Parse key-value pairs
        for line in config_section.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                if key == 'GPU ID':
                    config['gpu'] = int(value) if value.isdigit() else value
                elif key == 'NUMA Node':
                    config['numa'] = int(value) if value.isdigit() else value
                elif key == 'Test Type':
                    config['test_type'] = value
                elif key == 'ViT Patch Size':
                    config['vit_patch_size'] = int(value) if value.isdigit() else value
                elif key == 'ViT Depth':
                    config['vit_depth'] = int(value) if value.isdigit() else value
                elif key == 'ViT Dimension':
                    config['vit_dim'] = int(value) if value.isdigit() else value

        # For H2D/D2H Only tests, set ViT config to N/A
        if config.get('test_type') == 'H2D/D2H Only':
            config['vit_patch_size'] = 'N/A'
            config['vit_depth'] = 'N/A'
            config['vit_dim'] = 'N/A'

        return config

    def _parse_memory_transfers(self, content: str) -> Dict[str, Any]:
        """Parse MEMORY TRANSFER PERFORMANCE section."""
        memory = {}

        section = self._extract_section(content, 'MEMORY TRANSFER PERFORMANCE')
        if not section:
            return memory

        # Split into H2D and D2H subsections
        h2d_section = self._extract_subsection(section, 'Host-to-Device (H2D) Transfers:')
        d2h_section = self._extract_subsection(section, 'Device-to-Host (D2H) Transfers:')

        # Parse H2D data
        if h2d_section:
            h2d_data = self._parse_transfer_data(h2d_section, 'h2d_')
            memory.update(h2d_data)

        # Parse D2H data
        if d2h_section:
            d2h_data = self._parse_transfer_data(d2h_section, 'd2h_')
            memory.update(d2h_data)

        return memory

    def _parse_transfer_data(self, section: str, prefix: str) -> Dict[str, Any]:
        """Parse transfer data from a subsection."""
        data = {}

        for line in section.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                if key == 'Count':
                    data[f'{prefix}count'] = int(value)
                elif key == 'Total bytes':
                    # Extract MB value
                    mb_value = re.search(r'([\d.]+) MB', value)
                    data[f'{prefix}total_mb'] = float(mb_value.group(1)) if mb_value else 0
                elif key == 'Average bandwidth':
                    # Extract MB/s value
                    bw_value = re.search(r'([\d.]+) MB/s', value)
                    data[f'{prefix}bandwidth_mbps'] = float(bw_value.group(1)) if bw_value else 0
                elif key == 'Average duration':
                    # Extract μs value
                    dur_value = re.search(r'([\d.]+) μs', value)
                    data[f'{prefix}avg_us'] = float(dur_value.group(1)) if dur_value else 0

        return data

    def _parse_compute_performance(self, content: str) -> Dict[str, Any]:
        """Parse COMPUTE PERFORMANCE section."""
        compute = {}

        section = self._extract_section(content, 'COMPUTE PERFORMANCE')
        if not section:
            return compute

        for line in section.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                if key == 'Total kernels':
                    compute['total_kernels'] = int(value)
                elif key == 'Total compute time':
                    ms_value = re.search(r'([\d.]+) ms', value)
                    compute['total_compute_ms'] = float(ms_value.group(1)) if ms_value else 0
                elif key == 'Average kernel duration':
                    ms_value = re.search(r'([\d.]+) ms', value)
                    compute['avg_kernel_ms'] = float(ms_value.group(1)) if ms_value else 0
                elif key == 'Compute timeline':
                    ms_value = re.search(r'([\d.]+) ms', value)
                    compute['compute_timeline_ms'] = float(ms_value.group(1)) if ms_value else 0
                elif key == 'Compute utilization':
                    # Extract decimal value before parentheses
                    util_value = re.search(r'([\d.]+)', value)
                    compute['compute_utilization'] = float(util_value.group(1)) if util_value else 0

        return compute

    def _parse_pipeline_analysis(self, content: str) -> Dict[str, Any]:
        """Parse PIPELINE ANALYSIS section."""
        pipeline = {}

        # Try both possible section names
        section = self._extract_section(content, 'PIPELINE ANALYSIS (Per-Stream)')
        if not section:
            section = self._extract_section(content, 'PIPELINE ANALYSIS')
        if not section:
            return pipeline

        for line in section.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                if key == 'Pipeline kernels':
                    pipeline['pipeline_kernels'] = int(value)
                elif key == 'Average preparation time':
                    us_value = re.search(r'([\d.]+) μs', value)
                    pipeline['avg_prep_us'] = float(us_value.group(1)) if us_value else 0
                elif key == 'Pipeline efficiency':
                    # Extract decimal value before parentheses
                    eff_value = re.search(r'([\d.]+)', value)
                    pipeline['pipeline_efficiency'] = float(eff_value.group(1)) if eff_value else 0
                elif key == 'Significant gaps (>1μs)':
                    pipeline['significant_gaps'] = int(value)
                elif key == 'Total significant gap time':
                    ms_value = re.search(r'([\d.]+) ms', value)
                    pipeline['total_gap_ms'] = float(ms_value.group(1)) if ms_value else 0
                elif key == 'Maximum gap':
                    us_value = re.search(r'([\d.]+) μs', value)
                    pipeline['max_gap_us'] = float(us_value.group(1)) if us_value else 0

        return pipeline

    def _parse_temporal_compression(self, content: str) -> Dict[str, Any]:
        """Parse TEMPORAL COMPRESSION ANALYSIS section."""
        temporal = {}

        section = self._extract_section(content, 'TEMPORAL COMPRESSION ANALYSIS')
        if not section:
            return temporal

        for line in section.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                if key == 'Timeline duration':
                    ms_value = re.search(r'([\d.]+) ms', value)
                    temporal['timeline_ms'] = float(ms_value.group(1)) if ms_value else 0
                elif key == 'Total operation time (sequential)':
                    ms_value = re.search(r'([\d.]+) ms', value)
                    temporal['total_operation_ms'] = float(ms_value.group(1)) if ms_value else 0
                elif key == 'Time saved through overlapping':
                    # Can be negative, so handle that
                    ms_value = re.search(r'(-?[\d.]+) ms', value)
                    temporal['time_saved_ms'] = float(ms_value.group(1)) if ms_value else 0
                elif key == 'Temporal compression ratio':
                    # Extract decimal value before parentheses
                    ratio_value = re.search(r'([\d.]+)', value)
                    temporal['compression_ratio'] = float(ratio_value.group(1)) if ratio_value else 0
                elif key == 'Active streams':
                    temporal['active_streams'] = int(value)
                elif key == 'Average stream utilization':
                    # Extract decimal value before parentheses
                    util_value = re.search(r'([\d.]+)', value)
                    temporal['avg_stream_util'] = float(util_value.group(1)) if util_value else 0

        return temporal

    def _extract_section(self, content: str, section_name: str) -> Optional[str]:
        """Extract a main section from the content."""
        pattern = rf'{re.escape(section_name)}\s*\n-+\n(.*?)(?=\n[A-Z][A-Z\s]+\n-+|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_subsection(self, section: str, subsection_name: str) -> Optional[str]:
        """Extract a subsection from a section."""
        lines = section.split('\n')
        start_idx = None

        for i, line in enumerate(lines):
            if subsection_name in line:
                start_idx = i + 1
                break

        if start_idx is None:
            return None

        # Find end of subsection (next subsection or empty line)
        end_idx = len(lines)
        for i in range(start_idx, len(lines)):
            if lines[i].strip() == '' or lines[i].endswith('Transfers:'):
                end_idx = i
                break

        return '\n'.join(lines[start_idx:end_idx])


def write_csv_tables(parsed_data: List[Dict[str, Any]], output_dir: Path):
    """Write CSV tables for each performance category."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Memory Transfer Performance Table
    memory_file = output_dir / 'memory_transfer_performance.csv'
    memory_headers = [
        'gpu', 'numa', 'test_type', 'vit_patch_size', 'vit_depth', 'vit_dim',
        'h2d_count', 'h2d_total_mb', 'h2d_bandwidth_mbps', 'h2d_avg_us',
        'd2h_count', 'd2h_total_mb', 'd2h_bandwidth_mbps', 'd2h_avg_us',
        'source_file'
    ]

    with open(memory_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=memory_headers)
        writer.writeheader()

        for data in parsed_data:
            row = {}
            row.update(data['config'])
            row.update(data['memory'])
            row['source_file'] = Path(data['source_file']).name
            # Fill missing fields with empty values
            for header in memory_headers:
                if header not in row:
                    row[header] = ''
            writer.writerow({k: row.get(k, '') for k in memory_headers})

    # Compute Performance Table
    compute_file = output_dir / 'compute_performance.csv'
    compute_headers = [
        'gpu', 'numa', 'test_type', 'vit_patch_size', 'vit_depth', 'vit_dim',
        'total_kernels', 'total_compute_ms', 'avg_kernel_ms',
        'compute_timeline_ms', 'compute_utilization',
        'source_file'
    ]

    with open(compute_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=compute_headers)
        writer.writeheader()

        for data in parsed_data:
            row = {}
            row.update(data['config'])
            row.update(data['compute'])
            row['source_file'] = Path(data['source_file']).name
            writer.writerow({k: row.get(k, '') for k in compute_headers})

    # Pipeline Analysis Table
    pipeline_file = output_dir / 'pipeline_analysis.csv'
    pipeline_headers = [
        'gpu', 'numa', 'test_type', 'vit_patch_size', 'vit_depth', 'vit_dim',
        'pipeline_kernels', 'avg_prep_us', 'pipeline_efficiency',
        'significant_gaps', 'total_gap_ms', 'max_gap_us',
        'source_file'
    ]

    with open(pipeline_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=pipeline_headers)
        writer.writeheader()

        for data in parsed_data:
            row = {}
            row.update(data['config'])
            row.update(data['pipeline'])
            row['source_file'] = Path(data['source_file']).name
            writer.writerow({k: row.get(k, '') for k in pipeline_headers})

    # Temporal Compression Analysis Table
    temporal_file = output_dir / 'temporal_compression_analysis.csv'
    temporal_headers = [
        'gpu', 'numa', 'test_type', 'vit_patch_size', 'vit_depth', 'vit_dim',
        'timeline_ms', 'total_operation_ms', 'time_saved_ms',
        'compression_ratio', 'active_streams', 'avg_stream_util',
        'source_file'
    ]

    with open(temporal_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=temporal_headers)
        writer.writeheader()

        for data in parsed_data:
            row = {}
            row.update(data['config'])
            row.update(data['temporal'])
            row['source_file'] = Path(data['source_file']).name
            writer.writerow({k: row.get(k, '') for k in temporal_headers})


def main():
    parser = argparse.ArgumentParser(
        description='Convert performance analysis text files to CSV tables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python results_to_table.py *.txt -o csv_tables/
  python results_to_table.py results/*_performance_analysis.txt -o analysis/
  python results_to_table.py "results/*.txt" -o tables/
        """
    )

    parser.add_argument('files', nargs='+',
                       help='Performance analysis text files to process')
    parser.add_argument('-o', '--output-dir', default='./csv_tables',
                       help='Output directory for CSV files (default: ./csv_tables)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Expand glob patterns
    all_files = []
    for pattern in args.files:
        expanded = glob.glob(pattern)
        if expanded:
            all_files.extend(expanded)
        else:
            # If no glob match, treat as literal filename
            all_files.append(pattern)

    if not all_files:
        print("No files found to process")
        return

    if args.verbose:
        print(f"Processing {len(all_files)} files...")

    # Parse all files
    parsed_data = []
    for file_path in all_files:
        if args.verbose:
            print(f"Parsing {file_path}...")

        try:
            parser = PerformanceAnalysisParser(file_path)
            data = parser.parse_file()
            data['source_file'] = file_path  # Add source file info
            parsed_data.append(data)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            continue

    if not parsed_data:
        print("No data was successfully parsed")
        return

    # Write CSV tables
    output_dir = Path(args.output_dir)
    if args.verbose:
        print(f"Writing CSV tables to {output_dir}...")

    write_csv_tables(parsed_data, output_dir)

    print(f"Generated CSV tables in {output_dir}:")
    for csv_file in output_dir.glob('*.csv'):
        print(f"  {csv_file.name}")


if __name__ == '__main__':
    main()

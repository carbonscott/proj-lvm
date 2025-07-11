#!/usr/bin/env python3
"""
Batch convert nsys reports to SQLite databases for analysis.

This script finds all nsys-rep files in the experiment outputs and converts
them to SQLite format using nsys export, enabling detailed performance analysis.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
import argparse


def find_nsys_reports(base_dir: str) -> List[Path]:
    """Find all nsys-rep files in the experiment output directory."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    
    # Find all .nsys-rep files recursively
    nsys_files = list(base_path.rglob("*.nsys-rep"))
    
    return sorted(nsys_files)


def convert_nsys_to_sqlite(nsys_file: Path, output_dir: Path = None) -> Tuple[bool, str]:
    """
    Convert a single nsys report to SQLite format.
    
    Args:
        nsys_file: Path to the .nsys-rep file
        output_dir: Optional output directory (default: same as nsys file)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        sqlite_file = output_dir / f"{nsys_file.stem}.sqlite"
    else:
        sqlite_file = nsys_file.with_suffix('.sqlite')
    
    # Skip if SQLite file already exists
    if sqlite_file.exists():
        return True, f"SQLite file already exists: {sqlite_file.name}"
    
    try:
        # Run nsys export command
        cmd = [
            'nsys', 'export',
            '--type', 'sqlite',
            '--output', str(sqlite_file),
            str(nsys_file)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per file
        )
        
        if result.returncode == 0:
            return True, f"Successfully converted: {nsys_file.name} → {sqlite_file.name}"
        else:
            return False, f"nsys export failed: {result.stderr.strip()}"
            
    except subprocess.TimeoutExpired:
        return False, f"Timeout converting {nsys_file.name}"
    except FileNotFoundError:
        return False, "nsys command not found - ensure NVIDIA Nsight Systems is installed"
    except Exception as e:
        return False, f"Error converting {nsys_file.name}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert nsys reports to SQLite for analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_nsys_to_sqlite.py outputs/numa_locality_study/
  python batch_nsys_to_sqlite.py outputs/ --output-dir sqlite_databases/
  python batch_nsys_to_sqlite.py outputs/ --dry-run
        """
    )
    
    parser.add_argument('input_dir', 
                       help='Directory containing nsys-rep files (searches recursively)')
    parser.add_argument('--output-dir', '-o',
                       help='Output directory for SQLite files (default: same as nsys files)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be converted without actually converting')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Find all nsys reports
    try:
        nsys_files = find_nsys_reports(args.input_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if not nsys_files:
        print(f"No nsys-rep files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"Found {len(nsys_files)} nsys report(s) to convert")
    
    if args.dry_run:
        print("\nDry run - would convert:")
        for nsys_file in nsys_files:
            output_dir = Path(args.output_dir) if args.output_dir else nsys_file.parent
            sqlite_file = output_dir / f"{nsys_file.stem}.sqlite"
            print(f"  {nsys_file.name} → {sqlite_file}")
        return
    
    # Convert each file
    output_dir = Path(args.output_dir) if args.output_dir else None
    successful = 0
    failed = 0
    
    print(f"\nConverting to SQLite...")
    print("=" * 60)
    
    for i, nsys_file in enumerate(nsys_files, 1):
        if args.verbose:
            print(f"[{i}/{len(nsys_files)}] Processing {nsys_file.name}...")
        else:
            print(f"[{i}/{len(nsys_files)}] {nsys_file.name}")
        
        success, message = convert_nsys_to_sqlite(nsys_file, output_dir)
        
        if success:
            successful += 1
            if args.verbose:
                print(f"  ✓ {message}")
        else:
            failed += 1
            print(f"  ✗ {message}")
    
    print("=" * 60)
    print(f"Conversion completed: {successful} successful, {failed} failed")
    
    if successful > 0:
        sqlite_dir = output_dir if output_dir else "same directories as nsys files"
        print(f"SQLite databases saved to: {sqlite_dir}")
        print("\nNext steps:")
        print("1. Run analyze_latency.py on the SQLite files")
        print("2. Use aggregate_results.py to convert to CSV tables")
    
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
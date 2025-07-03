import sqlite3
import argparse
from pathlib import Path
import numpy as np
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Tuple
import re

class SQLiteCUPTIAnalyzer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.config_name = Path(db_path).stem
        self.test_time_range: Optional[Tuple[int, int]] = None
        self.schema_info: Dict[str, Dict[str, bool]] = {}

    @contextmanager
    def get_connection(self):
        """Context manager for READ-ONLY database connections with safe temp indexes."""
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            conn.execute("PRAGMA optimize;")
            conn.execute("PRAGMA temp_store = MEMORY;")

            # Create temp indexes safely - only if tables exist
            cursor = conn.cursor()

            # Check which tables exist first
            existing_tables = cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN (
                    'CUPTI_ACTIVITY_KIND_KERNEL',
                    'CUPTI_ACTIVITY_KIND_RUNTIME',
                    'CUPTI_ACTIVITY_KIND_MEMCPY',
                    'CUPTI_ACTIVITY_KIND_MEMSET'
                )
            """).fetchall()

            existing_table_names = {row[0] for row in existing_tables}

            # Create indexes only for existing tables
            if 'CUPTI_ACTIVITY_KIND_KERNEL' in existing_table_names:
                try:
                    cursor.execute("CREATE TEMP INDEX k_corr_tmp ON CUPTI_ACTIVITY_KIND_KERNEL(correlationId)")
                    cursor.execute('CREATE TEMP INDEX k_time_tmp ON CUPTI_ACTIVITY_KIND_KERNEL(start, "end")')
                except sqlite3.OperationalError:
                    pass  # Index creation failed, continue without

            if 'CUPTI_ACTIVITY_KIND_RUNTIME' in existing_table_names:
                try:
                    cursor.execute("CREATE TEMP INDEX r_corr_tmp ON CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)")
                    cursor.execute('CREATE TEMP INDEX r_time_tmp ON CUPTI_ACTIVITY_KIND_RUNTIME(start, "end")')
                except sqlite3.OperationalError:
                    pass

            if 'CUPTI_ACTIVITY_KIND_MEMCPY' in existing_table_names:
                try:
                    cursor.execute('CREATE TEMP INDEX m_time_tmp ON CUPTI_ACTIVITY_KIND_MEMCPY(start, "end")')
                except sqlite3.OperationalError:
                    pass

            if 'CUPTI_ACTIVITY_KIND_MEMSET' in existing_table_names:
                try:
                    cursor.execute('CREATE TEMP INDEX s_time_tmp ON CUPTI_ACTIVITY_KIND_MEMSET(start, "end")')
                except sqlite3.OperationalError:
                    pass

            yield conn
        finally:
            conn.close()

    def probe_schema(self):
        """Probe database schema for column variations across CUPTI versions."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            tables_to_probe = {
                'CUPTI_ACTIVITY_KIND_MEMCPY': ['copyKind', 'memcpyKind', 'bytes'],
                'CUPTI_ACTIVITY_KIND_MEMSET': ['bytes'],
                'CUPTI_ACTIVITY_KIND_KERNEL': ['demangledName'],
                'CUPTI_ACTIVITY_KIND_RUNTIME': ['nameId']
            }

            for table, columns in tables_to_probe.items():
                self.schema_info[table] = {}
                try:
                    table_info = cursor.execute(f"PRAGMA table_info({table})").fetchall()
                    existing_columns = {col[1] for col in table_info}

                    for col in columns:
                        self.schema_info[table][col] = col in existing_columns

                except sqlite3.OperationalError as e:
                    if "no such table" in str(e).lower():
                        # Table doesn't exist
                        self.schema_info[table] = {col: False for col in columns}
                    else:
                        raise

    def parse_configuration(self) -> Dict[str, Any]:
        """Extract configuration details from filename."""
        config = {
            'gpu_id': 'unknown',
            'numa_node': 'unknown',
            'vit_patch_size': 'unknown',
            'vit_depth': 'unknown',
            'vit_dim': 'unknown',
            'test_type': 'unknown'
        }

        # Pattern: pipeline_gpu{N}_numa{N}_vit{patch}x{depth}x{dim}
        pipeline_pattern = r'pipeline_gpu(\d+)_numa(\d+)_vit(\d+)x(\d+)x(\d+)'
        match = re.search(pipeline_pattern, self.config_name)

        if match:
            config.update({
                'gpu_id': int(match.group(1)),
                'numa_node': int(match.group(2)),
                'vit_patch_size': int(match.group(3)),
                'vit_depth': int(match.group(4)),
                'vit_dim': int(match.group(5)),
                'test_type': 'ViT Pipeline' if int(match.group(4)) > 0 else 'H2D/D2H Only'
            })
        else:
            # Try h2d_d2h pattern
            h2d_pattern = r'h2d_d2h_gpu(\d+)_numa(\d+)'
            match = re.search(h2d_pattern, self.config_name)
            if match:
                config.update({
                    'gpu_id': int(match.group(1)),
                    'numa_node': int(match.group(2)),
                    'vit_depth': 0,
                    'test_type': 'H2D/D2H Only'
                })

        return config

    def get_time_conditions(self) -> Tuple[str, Tuple[int, ...]]:
        """Get time filter conditions and parameters for individual WHERE clauses."""
        if not self.test_time_range:
            return "", ()

        condition = "start >= ? AND \"end\" <= ?"
        params = self.test_time_range
        return condition, params

    def safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division that handles zero denominators."""
        return default if denominator == 0 else numerator / denominator

    def find_test_time_range(self) -> bool:
        """Find the time range for 'test_double_buffer' using MIN/MAX to handle multiple markers."""
        with self.get_connection() as conn:
            try:
                query = """
                SELECT MIN(start) as min_start, MAX("end") as max_end
                FROM NVTX_EVENTS
                WHERE text = ?
                """
                cursor = conn.cursor()
                result = cursor.execute(query, ('test_double_buffer',)).fetchone()

                if result and result[0] is not None and result[1] is not None:
                    start_time, end_time = result
                    self.test_time_range = (start_time, end_time)
                    print(f"Found test_double_buffer period: {start_time} - {end_time} ns")
                    print(f"Test duration: {(end_time - start_time) / 1e6:.2f} ms")
                    return True
                else:
                    print("Warning: 'test_double_buffer' event not found in NVTX_EVENTS")
                    return False

            except sqlite3.OperationalError as e:
                if "no such table" in str(e).lower():
                    print("No NVTX_EVENTS table found")
                else:
                    print(f"Database error finding test time range: {e}")
                return False

    def get_table_info(self) -> Dict[str, Dict[str, int]]:
        """Get table row counts."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            table_info = {}
            cupti_tables = ['CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY',
                          'CUPTI_ACTIVITY_KIND_MEMSET', 'CUPTI_ACTIVITY_KIND_RUNTIME']

            for table_name in cupti_tables:
                try:
                    total_count = cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]

                    if self.test_time_range:
                        time_condition, time_params = self.get_time_conditions()
                        filtered_count = cursor.execute(
                            f'SELECT COUNT(*) FROM "{table_name}" WHERE {time_condition}',
                            time_params
                        ).fetchone()[0]
                        table_info[table_name] = {
                            'total': total_count,
                            'filtered': filtered_count
                        }
                    else:
                        table_info[table_name] = {
                            'total': total_count,
                            'filtered': total_count
                        }
                except sqlite3.OperationalError as e:
                    if "no such table" in str(e).lower():
                        table_info[table_name] = {'total': 0, 'filtered': 0}
                    else:
                        raise

            return table_info

    def check_required_tables(self) -> bool:
        """Check if we have the minimum required tables for analysis."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                # Check for kernel table (minimum requirement)
                kernel_count = cursor.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
                return kernel_count > 0
            except sqlite3.OperationalError as e:
                if "no such table" in str(e).lower():
                    return False
                else:
                    raise

    def analyze_memory_transfers(self) -> Dict[str, Any]:
        """Analyze H2D and D2H transfer performance."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            time_condition, time_params = self.get_time_conditions()

            # Check if memcpy table exists
            try:
                cursor.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY LIMIT 1')
            except sqlite3.OperationalError as e:
                if "no such table" in str(e).lower():
                    return {'error': 'MEMCPY table not found'}
                else:
                    raise

            where_clause = f"WHERE {time_condition}" if time_condition else ""

            # Determine column name for copy kind (schema variations)
            copy_kind_col = 'copyKind' if self.schema_info.get('CUPTI_ACTIVITY_KIND_MEMCPY', {}).get('copyKind', False) else 'memcpyKind'

            # Check if bytes column exists - fail clearly if missing
            if not self.schema_info.get('CUPTI_ACTIVITY_KIND_MEMCPY', {}).get('bytes', False):
                return {'error': 'MEMCPY table missing bytes column - cannot calculate bandwidth'}
            bytes_col = 'bytes'

            query = f"""
            WITH transfer_stats AS (
                SELECT
                    {copy_kind_col} as copy_kind,
                    COUNT(*) as transfer_count,
                    SUM({bytes_col}) as total_bytes,
                    SUM("end" - start) as total_duration_ns,
                    AVG("end" - start) as avg_duration_ns,
                    MIN("end" - start) as min_duration_ns,
                    MAX("end" - start) as max_duration_ns,
                    MIN(start) as first_transfer,
                    MAX("end") as last_transfer
                FROM CUPTI_ACTIVITY_KIND_MEMCPY
                {where_clause}
                GROUP BY {copy_kind_col}
            )
            SELECT
                copy_kind,
                transfer_count,
                total_bytes,
                total_duration_ns,
                avg_duration_ns,
                min_duration_ns,
                max_duration_ns,
                first_transfer,
                last_transfer,
                CASE
                    WHEN total_duration_ns > 0 THEN (total_bytes / (total_duration_ns / 1e9)) / (1024.0 * 1024.0)
                    ELSE 0
                END as avg_bandwidth_mbps
            FROM transfer_stats
            ORDER BY copy_kind
            """

            try:
                results = cursor.execute(query, time_params).fetchall()
                columns = [desc[0] for desc in cursor.description]

                transfers = {}
                for row in results:
                    transfer_data = dict(zip(columns, row))
                    copy_kind = transfer_data['copy_kind']

                    # Map copy kinds to readable names
                    if copy_kind == 1:  # HtoD
                        transfers['h2d'] = transfer_data
                    elif copy_kind == 2:  # DtoH
                        transfers['d2h'] = transfer_data
                    else:
                        transfers[f'other_{copy_kind}'] = transfer_data

                return transfers
            except sqlite3.OperationalError as e:
                return {'error': f'Query failed: {e}'}

    def analyze_compute_performance(self) -> Dict[str, Any]:
        """Analyze kernel execution statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            time_condition, time_params = self.get_time_conditions()

            where_clause = f"WHERE {time_condition}" if time_condition else ""

            query = f"""
            SELECT
                COUNT(*) as total_kernels,
                SUM("end" - start) as total_compute_time_ns,
                AVG("end" - start) as avg_kernel_duration_ns,
                MIN("end" - start) as min_kernel_duration_ns,
                MAX("end" - start) as max_kernel_duration_ns,
                MIN(start) as first_kernel_start,
                MAX("end") as last_kernel_end
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            {where_clause}
            """

            try:
                result = cursor.execute(query, time_params).fetchone()
                columns = [desc[0] for desc in cursor.description]

                if result and result[0] is not None:
                    compute_stats = dict(zip(columns, result))

                    # Calculate additional metrics
                    if compute_stats['first_kernel_start'] and compute_stats['last_kernel_end']:
                        total_timeline = compute_stats['last_kernel_end'] - compute_stats['first_kernel_start']
                        compute_stats['total_timeline_ns'] = total_timeline
                        compute_stats['compute_utilization'] = self.safe_divide(
                            compute_stats['total_compute_time_ns'], total_timeline
                        )

                    return compute_stats
                return {}
            except sqlite3.OperationalError as e:
                return {'error': f'Query failed: {e}'}

    def analyze_pipeline_gaps_detailed(self) -> Dict[str, Any]:
        """Enhanced gap analysis with distribution info."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            time_condition, time_params = self.get_time_conditions()

            # Check if both required tables exist
            try:
                cursor.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME LIMIT 1')
                cursor.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 1')
            except sqlite3.OperationalError as e:
                if "no such table" in str(e).lower():
                    return {'error': 'Pipeline analysis requires both RUNTIME and KERNEL tables'}
                else:
                    raise

            if time_condition:
                # Use proper table aliases instead of string replacement
                r_condition = time_condition.replace('start', 'r.start').replace('"end"', 'r."end"')
                k_condition = time_condition.replace('start', 'k.start').replace('"end"', 'k."end"')
                where_clause = f"WHERE ({r_condition}) AND ({k_condition})"
                query_params = time_params + time_params
            else:
                where_clause = ""
                query_params = ()

            query = f"""
            WITH pipeline_flow AS (
                SELECT
                    r.start AS api_start,
                    r."end" AS api_end,
                    k.start AS gpu_start,
                    k."end" AS gpu_end,
                    k.streamId,
                    (k.start - r."end") AS preparation_time,
                    (k."end" - k.start) AS execution_time,
                    (r."end" - r.start) AS api_time,
                    ROW_NUMBER() OVER (ORDER BY k.start) as execution_order
                FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON r.correlationId = k.correlationId
                {where_clause}
                ORDER BY k.start
            ),
            pipeline_gaps AS (
                SELECT
                    execution_order,
                    gpu_start,
                    gpu_end,
                    streamId,
                    LAG(gpu_end) OVER (PARTITION BY streamId ORDER BY gpu_start) as prev_gpu_end,
                    CASE
                        WHEN LAG(gpu_end) OVER (PARTITION BY streamId ORDER BY gpu_start) IS NOT NULL
                        THEN gpu_start - LAG(gpu_end) OVER (PARTITION BY streamId ORDER BY gpu_start)
                        ELSE 0
                    END as gap_ns
                FROM pipeline_flow
            )
            SELECT
                COUNT(DISTINCT pf.execution_order) as total_kernels,
                AVG(pf.preparation_time) / 1e3 as avg_preparation_us,
                AVG(pf.execution_time) / 1e6 as avg_execution_ms,
                SUM(pf.execution_time) / 1e6 as total_compute_time_ms,
                (MAX(pf.gpu_end) - MIN(pf.gpu_start)) / 1e6 as total_timeline_ms,
                1.0 - (COALESCE(SUM(CASE WHEN pg.gap_ns > 1000 THEN pg.gap_ns ELSE 0 END), 0) /
                       CAST(MAX(pf.gpu_end) - MIN(pf.gpu_start) AS FLOAT)) as pipeline_efficiency,
                COUNT(pg.gap_ns) - 1 as total_gaps,
                COALESCE(SUM(CASE WHEN pg.gap_ns > 1000 THEN pg.gap_ns ELSE 0 END), 0) as total_significant_gap_time_ns,
                COALESCE(SUM(CASE WHEN pg.gap_ns > 1000 THEN 1 ELSE 0 END), 0) as significant_gaps_count,
                COALESCE(SUM(CASE WHEN pg.gap_ns > 10000 THEN 1 ELSE 0 END), 0) as large_gaps_count,
                COALESCE(SUM(CASE WHEN pg.gap_ns > 100000 THEN 1 ELSE 0 END), 0) as very_large_gaps_count,
                AVG(CASE WHEN pg.gap_ns > 0 THEN pg.gap_ns END) as avg_gap_ns,
                MAX(pg.gap_ns) as max_gap_ns,
                MIN(CASE WHEN pg.gap_ns > 0 THEN pg.gap_ns END) as min_positive_gap_ns
            FROM pipeline_flow pf
            LEFT JOIN pipeline_gaps pg ON pf.execution_order = pg.execution_order
            """

            try:
                result = cursor.execute(query, query_params).fetchone()
                columns = [desc[0] for desc in cursor.description]

                if result and result[0] is not None:
                    return dict(zip(columns, result))
                return {}
            except sqlite3.OperationalError as e:
                return {'error': f'Query failed: {e}'}

    def analyze_temporal_compression_ratio(self) -> Dict[str, Any]:
        """
        Analyze actual temporal compression ratio and idle time.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            time_condition, time_params = self.get_time_conditions()

            # Build the UNION query only for tables that exist
            union_parts = []
            all_params = []

            # Check each table and add to union if it exists
            tables_to_check = [
                ('CUPTI_ACTIVITY_KIND_KERNEL', 'kernel'),
                ('CUPTI_ACTIVITY_KIND_MEMCPY', 'memcpy'),
                ('CUPTI_ACTIVITY_KIND_MEMSET', 'memset')
            ]

            for table_name, op_type in tables_to_check:
                try:
                    # Test if table exists and has data
                    test_query = f'SELECT COUNT(*) FROM "{table_name}" LIMIT 1'
                    cursor.execute(test_query)

                    # Add to union
                    where_clause = f"WHERE {time_condition}" if time_condition else ""
                    union_parts.append(f"""
                        SELECT start, "end", '{op_type}' as op_type, streamId
                        FROM "{table_name}" {where_clause}
                    """)
                    if time_condition:
                        all_params.extend(time_params)

                except sqlite3.OperationalError:
                    # Table doesn't exist, skip it
                    continue

            if not union_parts:
                return {'error': 'No CUPTI activity tables found'}

            # Build the full query
            union_query = " UNION ALL ".join(union_parts)

            query = f"""
            WITH all_gpu_ops AS (
                {union_query}
            ),
            timeline AS (
                SELECT
                    MIN(start) as first_op_start,
                    MAX("end") as last_op_end,
                    SUM("end" - start) as total_active_time,
                    COUNT(*) as total_operations
                FROM all_gpu_ops
            ),
            stream_stats AS (
                SELECT
                    streamId,
                    MIN(start) as stream_start,
                    MAX("end") as stream_end,
                    SUM("end" - start) as stream_active_time,
                    COUNT(*) as stream_ops,
                    (MAX("end") - MIN(start)) - SUM("end" - start) as stream_idle_time
                FROM all_gpu_ops
                GROUP BY streamId
            )
            SELECT
                t.first_op_start,
                t.last_op_end,
                (t.last_op_end - t.first_op_start) / 1e6 as total_timeline_ms,
                t.total_active_time / 1e6 as total_active_ms,
                ((t.last_op_end - t.first_op_start) - t.total_active_time) / 1e6 as total_idle_ms,
                t.total_active_time / CAST(t.last_op_end - t.first_op_start AS FLOAT) as temporal_compression_ratio,
                t.total_operations,
                (SELECT COUNT(*) FROM stream_stats) as active_streams,
                (SELECT AVG(stream_active_time / CAST(stream_end - stream_start AS FLOAT)) FROM stream_stats) as avg_stream_utilization,
                (SELECT SUM(stream_idle_time) FROM stream_stats) / 1e6 as total_stream_idle_ms
            FROM timeline t
            """

            try:
                result = cursor.execute(query, all_params).fetchone()

                if result and result[0] is not None:
                    return {
                        'timeline_start': result[0],
                        'timeline_end': result[1],
                        'total_timeline_ms': result[2],
                        'total_active_ms': result[3],
                        'total_idle_ms': result[4],
                        'temporal_compression_ratio': result[5],
                        'total_operations': result[6],
                        'active_streams': result[7],
                        'avg_stream_utilization': result[8] or 0,
                        'total_stream_idle_ms': result[9] or 0
                    }
                return {}
            except sqlite3.OperationalError as e:
                return {'error': f'Query failed: {e}'}



    def run_full_analysis(self, analyze_full_trace: bool = False) -> Dict[str, Any]:
        """Run comprehensive analysis with objective metrics."""
        print(f"Analyzing configuration: {self.config_name}")

        # Check if we have minimum required tables
        if not self.check_required_tables():
            print("Error: No CUPTI_ACTIVITY_KIND_KERNEL table found or it's empty")
            return {
                'config_name': self.config_name,
                'error': 'Missing required CUPTI tables'
            }

        self.probe_schema()
        config = self.parse_configuration()

        if not analyze_full_trace:
            found_test_range = self.find_test_time_range()
        else:
            self.test_time_range = None
            found_test_range = False
            print("Analyzing entire trace (including warmup/cleanup)")

        table_info = self.get_table_info()
        print(f"Tables found: {table_info}")

        # Run all analysis components
        print("Analyzing memory transfers...")
        memory_transfers = self.analyze_memory_transfers()

        print("Analyzing compute performance...")
        compute_performance = self.analyze_compute_performance()

        print("Analyzing pipeline gaps...")
        pipeline_analysis = self.analyze_pipeline_gaps_detailed()

        print("Analyzing temporal compression ratio...")
        temporal_compression = self.analyze_temporal_compression_ratio()

        results = {
            'config_name': self.config_name,
            'configuration': config,
            'test_time_range': self.test_time_range,
            'filtered_analysis': found_test_range,
            'table_info': table_info,
            'memory_transfers': memory_transfers,
            'compute_performance': compute_performance,
            'pipeline_analysis': pipeline_analysis,
            'temporal_compression': temporal_compression
        }

        return results

def write_analysis_results(results: Dict[str, Any], output_file):
    """Write objective analysis results."""
    config_name = results['config_name']

    if 'error' in results:
        with open(output_file, 'w') as f:
            f.write(f"Analysis Error - Configuration: {config_name}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Error: {results['error']}\n")
        return

    with open(output_file, 'w') as f:
        f.write(f"CUPTI Performance Analysis - Configuration: {config_name}\n")
        f.write("=" * 80 + "\n")

        # Configuration section
        if results.get('configuration'):
            f.write("CONFIGURATION\n")
            f.write("-" * 20 + "\n")
            config = results['configuration']
            f.write(f"GPU ID: {config.get('gpu_id', 'unknown')}\n")
            f.write(f"NUMA Node: {config.get('numa_node', 'unknown')}\n")
            f.write(f"Test Type: {config.get('test_type', 'unknown')}\n")
            if config.get('vit_depth', 0) > 0:
                f.write(f"ViT Patch Size: {config.get('vit_patch_size', 'unknown')}\n")
                f.write(f"ViT Depth: {config.get('vit_depth', 'unknown')}\n")
                f.write(f"ViT Dimension: {config.get('vit_dim', 'unknown')}\n")
            f.write("\n")

        # Analysis scope
        if results['filtered_analysis']:
            f.write("ANALYSIS SCOPE\n")
            f.write("-" * 20 + "\n")
            f.write("Scope: test_double_buffer period only\n")
            if results['test_time_range']:
                start_ns, end_ns = results['test_time_range']
                duration_ms = (end_ns - start_ns) / 1e6
                f.write(f"Time range: {start_ns} - {end_ns} ns\n")
                f.write(f"Duration: {duration_ms:.2f} ms\n")
        else:
            f.write("ANALYSIS SCOPE\n")
            f.write("-" * 20 + "\n")
            f.write("Scope: Entire trace\n")
        f.write("\n")

        # Memory transfer performance
        if results.get('memory_transfers') and 'error' not in results['memory_transfers']:
            f.write("MEMORY TRANSFER PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            transfers = results['memory_transfers']

            for transfer_type, data in sorted(transfers.items()):
                if transfer_type == 'h2d':
                    f.write("Host-to-Device (H2D) Transfers:\n")
                elif transfer_type == 'd2h':
                    f.write("Device-to-Host (D2H) Transfers:\n")
                else:
                    f.write(f"{transfer_type.replace('_', ' ').title()} Transfers:\n")

                f.write(f"  Count: {data.get('transfer_count', 0) or 0}\n")
                f.write(f"  Total bytes: {(data.get('total_bytes', 0) or 0) / (1024*1024):.2f} MB\n")
                f.write(f"  Total time: {(data.get('total_duration_ns', 0) or 0) / 1e6:.2f} ms\n")
                f.write(f"  Average bandwidth: {data.get('avg_bandwidth_mbps', 0) or 0:.2f} MB/s\n")
                f.write(f"  Average duration: {(data.get('avg_duration_ns', 0) or 0) / 1e3:.2f} μs\n")
                f.write(f"  Duration range: {(data.get('min_duration_ns', 0) or 0) / 1e3:.2f} - {(data.get('max_duration_ns', 0) or 0) / 1e3:.2f} μs\n")
                f.write("\n")
        elif results.get('memory_transfers', {}).get('error'):
            f.write("MEMORY TRANSFER PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            f.write(f"Error: {results['memory_transfers']['error']}\n\n")

        # Compute performance
        if results.get('compute_performance') and 'error' not in results['compute_performance']:
            f.write("COMPUTE PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            compute = results['compute_performance']
            f.write(f"Total kernels: {compute.get('total_kernels', 0) or 0}\n")
            f.write(f"Total compute time: {(compute.get('total_compute_time_ns', 0) or 0) / 1e6:.2f} ms\n")
            f.write(f"Average kernel duration: {(compute.get('avg_kernel_duration_ns', 0) or 0) / 1e6:.2f} ms\n")
            f.write(f"Kernel duration range: {(compute.get('min_kernel_duration_ns', 0) or 0) / 1e6:.2f} - {(compute.get('max_kernel_duration_ns', 0) or 0) / 1e6:.2f} ms\n")
            if compute.get('total_timeline_ns'):
                f.write(f"Compute timeline: {(compute.get('total_timeline_ns', 0) or 0) / 1e6:.2f} ms\n")
                f.write(f"Compute utilization: {compute.get('compute_utilization', 0) or 0:.3f} ({(compute.get('compute_utilization', 0) or 0)*100:.1f}%)\n")
            f.write("\n")

        # Pipeline analysis
        if results.get('pipeline_analysis') and 'error' not in results['pipeline_analysis']:
            f.write("PIPELINE ANALYSIS (Per-Stream)\n")
            f.write("-" * 20 + "\n")
            pipeline = results['pipeline_analysis']
            f.write(f"Pipeline kernels: {pipeline.get('total_kernels', 0)}\n")
            f.write(f"Average preparation time: {pipeline.get('avg_preparation_us', 0):.2f} μs\n")
            f.write(f"Average execution time: {pipeline.get('avg_execution_ms', 0):.2f} ms\n")
            f.write(f"Total compute time: {pipeline.get('total_compute_time_ms', 0):.2f} ms\n")
            f.write(f"Total timeline: {pipeline.get('total_timeline_ms', 0):.2f} ms\n")
            f.write(f"Pipeline efficiency: {pipeline.get('pipeline_efficiency', 0):.3f} ({pipeline.get('pipeline_efficiency', 0)*100:.1f}%)\n")
            f.write("\n")

            # Gap statistics
            f.write("Gap Statistics:\n")
            f.write(f"  Total gaps: {pipeline.get('total_gaps', 0)}\n")
            f.write(f"  Significant gaps (>1μs): {pipeline.get('significant_gaps_count', 0)}\n")
            f.write(f"  Large gaps (>10μs): {pipeline.get('large_gaps_count', 0)}\n")
            f.write(f"  Very large gaps (>100μs): {pipeline.get('very_large_gaps_count', 0)}\n")
            f.write(f"  Total significant gap time: {pipeline.get('total_significant_gap_time_ns', 0) / 1e6:.2f} ms\n")
            if pipeline.get('avg_gap_ns'):
                f.write(f"  Average gap: {pipeline.get('avg_gap_ns', 0) / 1e3:.2f} μs\n")
            if pipeline.get('max_gap_ns'):
                f.write(f"  Maximum gap: {pipeline.get('max_gap_ns', 0) / 1e3:.2f} μs\n")

            # Check for suspicious preparation time
            prep_time_us = pipeline.get('avg_preparation_us', 0)
            if prep_time_us > 50000:  # > 50ms seems suspicious
                f.write(f"\nWarning: High preparation time ({prep_time_us:.0f} μs) may indicate timing issues.\n")
            f.write("\n")

        # Temporal compression ratio
        if results.get('temporal_compression') and 'error' not in results['temporal_compression']:
            f.write("TEMPORAL COMPRESSION ANALYSIS\n")
            f.write("-" * 30 + "\n")
            tcr = results['temporal_compression']
            f.write(f"Timeline duration: {tcr.get('total_timeline_ms', 0):.2f} ms\n")
            f.write(f"Total operation time (sequential): {tcr.get('total_active_ms', 0):.2f} ms\n")
            f.write(f"Time saved through overlapping: {-tcr.get('total_idle_ms', 0):.2f} ms\n")
            f.write(f"Temporal compression ratio: {tcr.get('temporal_compression_ratio', 0):.3f} ({tcr.get('temporal_compression_ratio', 0)*100:.1f}%)\n")
            f.write(f"Total operations: {tcr.get('total_operations', 0)}\n")
            f.write(f"Active streams: {tcr.get('active_streams', 0)}\n")
            f.write(f"Average stream utilization: {tcr.get('avg_stream_utilization', 0):.3f} ({tcr.get('avg_stream_utilization', 0)*100:.1f}%)\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(
        description="SQLite CUPTI performance analyzer with objective metrics.",
        epilog="Provides comprehensive performance analysis without subjective interpretations."
    )

    parser.add_argument('sqlite_files', nargs='+', help='SQLite database files to analyze')
    parser.add_argument('-o', '--output-dir', default='./performance_analysis_results',
                        help='Output directory for analysis results')
    parser.add_argument('--console-summary', action='store_true',
                        help='Print summary to console for each configuration')
    parser.add_argument('--include-warmup', action='store_true',
                        help='Include warmup/cleanup periods (analyze entire trace)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    print("Analyzing performance with objective metrics.")

    for i, sqlite_file in enumerate(args.sqlite_files, 1):
        sqlite_path = Path(sqlite_file)

        if not sqlite_path.exists():
            print(f"Warning: File not found: {sqlite_file}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(args.sqlite_files)}: {sqlite_path.name}")
        print(f"{'='*60}")

        try:
            analyzer = SQLiteCUPTIAnalyzer(str(sqlite_path))
            results = analyzer.run_full_analysis(analyze_full_trace=args.include_warmup)

            output_file = output_dir / f"{results['config_name']}_performance_analysis.txt"
            write_analysis_results(results, output_file)
            print(f"Results written to: {output_file}")

            if args.console_summary:
                if 'error' in results:
                    print(f"  Error: {results['error']}")
                else:
                    # Print key metrics
                    if results.get('configuration'):
                        config = results['configuration']
                        print(f"  GPU: {config.get('gpu_id', 'unknown')}, NUMA: {config.get('numa_node', 'unknown')}, Type: {config.get('test_type', 'unknown')}")

                    if results.get('temporal_compression'):
                        tcr = results['temporal_compression']
                        ratio = tcr.get('temporal_compression_ratio', 0)
                        print(f"  Temporal Compression Ratio: {ratio:.3f} ({ratio*100:.1f}%)")

                    if results.get('memory_transfers'):
                        transfers = results['memory_transfers']
                        if 'h2d' in transfers:
                            h2d_bw = transfers['h2d'].get('avg_bandwidth_mbps', 0)
                            print(f"  H2D Bandwidth: {h2d_bw:.1f} MB/s")
                        if 'd2h' in transfers:
                            d2h_bw = transfers['d2h'].get('avg_bandwidth_mbps', 0)
                            print(f"  D2H Bandwidth: {d2h_bw:.1f} MB/s")

                    if results.get('compute_performance'):
                        compute = results['compute_performance']
                        kernels = compute.get('total_kernels', 0)
                        util = compute.get('compute_utilization', 0)
                        print(f"  Kernels: {kernels}, Compute Utilization: {util:.3f}")

        except Exception as e:
            print(f"Error processing {sqlite_file}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nCompleted performance analysis. Results saved in: {output_dir}")

if __name__ == '__main__':
    main()

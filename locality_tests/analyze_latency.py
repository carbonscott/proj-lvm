import sqlite3
import argparse
from pathlib import Path
import numpy as np
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Tuple

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

                except sqlite3.OperationalError:
                    # Table doesn't exist
                    self.schema_info[table] = {col: False for col in columns}

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
                except sqlite3.OperationalError:
                    table_info[table_name] = {'total': 0, 'filtered': 0}

            return table_info

    def check_required_tables(self) -> bool:
        """Check if we have the minimum required tables for analysis."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                # Check for kernel table (minimum requirement)
                kernel_count = cursor.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
                return kernel_count > 0
            except sqlite3.OperationalError:
                return False

    def analyze_gpu_utilization(self) -> Dict[str, Any]:
        """
        Analyze actual GPU utilization and idle time.
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
                        FROM {table_name} {where_clause}
                    """)
                    if time_condition:
                        all_params.extend(time_params)

                except sqlite3.OperationalError:
                    # Table doesn't exist, skip it
                    continue

            if not union_parts:
                print("No CUPTI activity tables found")
                return {}

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
                t.total_active_time / CAST(t.last_op_end - t.first_op_start AS FLOAT) as gpu_utilization,
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
                        'gpu_utilization': result[5],
                        'total_operations': result[6],
                        'active_streams': result[7],
                        'avg_stream_utilization': result[8] or 0,
                        'total_stream_idle_ms': result[9] or 0,
                        'latency_hiding_effectiveness': result[5]
                    }
                return {}
            except sqlite3.OperationalError as e:
                print(f"Database error in GPU utilization analysis: {e}")
                return {}

    def analyze_stream_overlap(self) -> List[Dict[str, Any]]:
        """
        Analyze per-stream efficiency with safe table handling.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            time_condition, time_params = self.get_time_conditions()

            # Build union only for existing tables
            union_parts = []
            all_params = []

            tables_to_check = [
                ('CUPTI_ACTIVITY_KIND_KERNEL', 'kernel'),
                ('CUPTI_ACTIVITY_KIND_MEMCPY', 'memcpy'),
                ('CUPTI_ACTIVITY_KIND_MEMSET', 'memset')
            ]

            for table_name, op_type in tables_to_check:
                try:
                    cursor.execute(f'SELECT COUNT(*) FROM "{table_name}" LIMIT 1')
                    where_clause = f"WHERE {time_condition}" if time_condition else ""
                    union_parts.append(f"""
                        SELECT streamId, start, "end", '{op_type}' as op_type,
                               ("end" - start) as duration
                        FROM {table_name} {where_clause}
                    """)
                    if time_condition:
                        all_params.extend(time_params)
                except sqlite3.OperationalError:
                    continue

            if not union_parts:
                return []

            union_query = " UNION ALL ".join(union_parts)

            query = f"""
            WITH ops AS (
                {union_query}
            ),
            stream_stats AS (
                SELECT streamId,
                       COUNT(CASE WHEN op_type='kernel' THEN 1 END) AS kernel_count,
                       COUNT(CASE WHEN op_type IN ('memcpy', 'memset') THEN 1 END) AS memory_count,
                       SUM(CASE WHEN op_type='kernel' THEN duration END) / 1e6 AS compute_ms,
                       SUM(CASE WHEN op_type IN ('memcpy', 'memset') THEN duration END) / 1e6 AS memory_ms,
                       (MAX("end") - MIN(start)) / 1e6 AS span_ms,
                       SUM(duration) / 1e6 AS total_ops_ms,
                       ((MAX("end") - MIN(start)) - SUM(duration)) / 1e6 AS idle_time_ms
                FROM ops
                GROUP BY streamId
            )
            SELECT *,
                   CASE WHEN span_ms > 0 THEN total_ops_ms / span_ms ELSE 0 END as stream_efficiency,
                   CASE WHEN span_ms > 0 THEN idle_time_ms / span_ms ELSE 0 END as idle_percentage,
                   CASE WHEN span_ms > 0 THEN (kernel_count + memory_count) / span_ms ELSE 0 END as ops_per_ms,
                   CASE
                       WHEN kernel_count > 0 AND memory_count > 0 THEN 'mixed'
                       WHEN kernel_count > 0 THEN 'compute_only'
                       WHEN memory_count > 0 THEN 'memory_only'
                       ELSE 'empty'
                   END as stream_type
            FROM stream_stats
            ORDER BY streamId
            """

            try:
                results = cursor.execute(query, all_params).fetchall()
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in results]
            except sqlite3.OperationalError as e:
                print(f"Database error in stream analysis: {e}")
                return []

    def analyze_pipeline_efficiency(self) -> Optional[Dict[str, Any]]:
        """
        Analyze pipeline efficiency with safe table handling.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            time_condition, time_params = self.get_time_conditions()

            # Check if both required tables exist
            try:
                cursor.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME LIMIT 1')
                cursor.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 1')
            except sqlite3.OperationalError:
                print("Pipeline analysis requires both RUNTIME and KERNEL tables")
                return None

            if time_condition:
                k_condition = time_condition.replace('start', 'k.start').replace('"end"', 'k."end"')
                r_condition = time_condition.replace('start', 'r.start').replace('"end"', 'r."end"')
                where_clause = f"WHERE ({k_condition}) AND ({r_condition})"
                query_params = time_params + time_params
            else:
                where_clause = ""
                query_params = ()

            query = f"""
            WITH pipeline_flow AS (
                SELECT
                    r.start  AS api_start,
                    r."end"    AS api_end,
                    k.start  AS gpu_start,
                    k."end"    AS gpu_end,
                    k.streamId,
                    (k.start - r."end") AS preparation_time,
                    (k."end" - k.start) AS execution_time,
                    (r."end" - r.start) AS api_time,
                    ROW_NUMBER() OVER (ORDER BY k.start) as execution_order
                FROM    CUPTI_ACTIVITY_KIND_RUNTIME r
                JOIN    CUPTI_ACTIVITY_KIND_KERNEL  k ON r.correlationId = k.correlationId
                {where_clause}
                ORDER BY k.start
            ),
            pipeline_gaps AS (
                SELECT
                    execution_order,
                    gpu_start,
                    gpu_end,
                    streamId,
                    LAG(gpu_end) OVER (ORDER BY gpu_start) as prev_gpu_end,
                    CASE
                        WHEN LAG(gpu_end) OVER (ORDER BY gpu_start) IS NOT NULL
                        THEN gpu_start - LAG(gpu_end) OVER (ORDER BY gpu_start)
                        ELSE 0
                    END as actual_gap_ns
                FROM pipeline_flow
            )
            SELECT
                COUNT(*) as total_kernels,
                AVG(preparation_time) / 1e3 as avg_preparation_us,
                AVG(execution_time) / 1e6 as avg_execution_ms,
                SUM(CASE WHEN actual_gap_ns > 1000 THEN actual_gap_ns ELSE 0 END) / 1e6 as total_gap_time_ms,
                SUM(CASE WHEN actual_gap_ns > 1000 THEN 1 ELSE 0 END) as significant_gaps,
                AVG(CASE WHEN actual_gap_ns > 0 THEN actual_gap_ns END) / 1e3 as avg_gap_us,
                1.0 - (SUM(CASE WHEN actual_gap_ns > 1000 THEN actual_gap_ns ELSE 0 END) /
                       CAST(MAX(gpu_end) - MIN(gpu_start) AS FLOAT)) as pipeline_efficiency
            FROM pipeline_flow pf
            LEFT JOIN pipeline_gaps pg ON pf.execution_order = pg.execution_order
            """

            try:
                result = cursor.execute(query, query_params).fetchone()

                if result and result[0] is not None:
                    return {
                        'total_kernels': result[0],
                        'avg_preparation_us': result[1] or 0,
                        'avg_execution_ms': result[2] or 0,
                        'total_gap_time_ms': result[3] or 0,
                        'significant_gaps': result[4] or 0,
                        'avg_gap_us': result[5] or 0,
                        'pipeline_efficiency': result[6] or 0
                    }
                return None
            except sqlite3.OperationalError as e:
                print(f"Database error in pipeline efficiency analysis: {e}")
                return None

    def run_full_analysis(self, analyze_full_trace: bool = False) -> Dict[str, Any]:
        """Run corrected latency hiding analysis with safe error handling."""
        print(f"Analyzing configuration: {self.config_name}")

        # Check if we have minimum required tables
        if not self.check_required_tables():
            print("Error: No CUPTI_ACTIVITY_KIND_KERNEL table found or it's empty")
            return {
                'config_name': self.config_name,
                'error': 'Missing required CUPTI tables'
            }

        self.probe_schema()

        if not analyze_full_trace:
            found_test_range = self.find_test_time_range()
        else:
            self.test_time_range = None
            found_test_range = False
            print("Analyzing entire trace (including warmup/cleanup)")

        table_info = self.get_table_info()
        print(f"Tables found: {table_info}")

        print("Analyzing GPU utilization (primary latency hiding metric)...")
        gpu_utilization = self.analyze_gpu_utilization()

        print("Analyzing stream efficiency...")
        stream_analysis = self.analyze_stream_overlap()

        print("Analyzing pipeline efficiency...")
        pipeline_efficiency = self.analyze_pipeline_efficiency()

        results = {
            'config_name': self.config_name,
            'test_time_range': self.test_time_range,
            'filtered_analysis': found_test_range,
            'table_info': table_info,
            'gpu_utilization': gpu_utilization,
            'stream_analysis': stream_analysis,
            'pipeline_efficiency': pipeline_efficiency
        }

        return results

def write_analysis_results(results: Dict[str, Any], output_file):
    """Write corrected analysis results."""
    config_name = results['config_name']

    if 'error' in results:
        with open(output_file, 'w') as f:
            f.write(f"Analysis Error - Configuration: {config_name}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Error: {results['error']}\n")
        return

    with open(output_file, 'w') as f:
        f.write(f"CUPTI Latency Analysis - Configuration: {config_name}\n")
        f.write("=" * 80 + "\n")

        if results['filtered_analysis']:
            f.write("ANALYSIS SCOPE: test_double_buffer period only\n")
            if results['test_time_range']:
                start_ns, end_ns = results['test_time_range']
                duration_ms = (end_ns - start_ns) / 1e6
                f.write(f"Test period: {start_ns} - {end_ns} ns ({duration_ms:.2f} ms)\n")
        else:
            f.write("ANALYSIS SCOPE: Entire trace\n")
        f.write("\n")

        # Primary latency hiding metrics
        if results['gpu_utilization']:
            f.write("GPU UTILIZATION (PRIMARY LATENCY HIDING METRIC)\n")
            f.write("-" * 50 + "\n")
            gu = results['gpu_utilization']
            f.write(f"Timeline duration: {gu.get('total_timeline_ms', 0):.2f} ms\n")
            f.write(f"GPU active time: {gu.get('total_active_ms', 0):.2f} ms\n")
            f.write(f"GPU idle time: {gu.get('total_idle_ms', 0):.2f} ms\n")
            f.write(f"GPU UTILIZATION: {gu.get('gpu_utilization', 0):.3f} ({gu.get('gpu_utilization', 0)*100:.1f}%)\n")
            f.write(f"LATENCY HIDING EFFECTIVENESS: {gu.get('latency_hiding_effectiveness', 0):.3f}\n")
            f.write(f"Active streams: {gu.get('active_streams', 0)}\n")
            f.write(f"Average stream utilization: {gu.get('avg_stream_utilization', 0):.3f}\n")
            f.write("\n")

            # Interpretation
            util = gu.get('gpu_utilization', 0)
            if util > 0.95:
                f.write("INTERPRETATION: Excellent latency hiding! GPU is >95% utilized.\n")
            elif util > 0.85:
                f.write("INTERPRETATION: Good latency hiding. GPU is >85% utilized.\n")
            elif util > 0.70:
                f.write("INTERPRETATION: Moderate latency hiding. Some optimization possible.\n")
            else:
                f.write("INTERPRETATION: Poor latency hiding. Significant idle time detected.\n")
            f.write("\n")

        # Pipeline efficiency
        if results['pipeline_efficiency']:
            f.write("PIPELINE EFFICIENCY\n")
            f.write("-" * 20 + "\n")
            pe = results['pipeline_efficiency']
            f.write(f"Total kernels executed: {pe.get('total_kernels', 0)}\n")
            f.write(f"Average preparation time: {pe.get('avg_preparation_us', 0):.2f} μs (normal)\n")
            f.write(f"Average execution time: {pe.get('avg_execution_ms', 0):.2f} ms\n")
            f.write(f"Total gap time: {pe.get('total_gap_time_ms', 0):.2f} ms\n")
            f.write(f"Significant gaps (>1μs): {pe.get('significant_gaps', 0)}\n")
            f.write(f"PIPELINE EFFICIENCY: {pe.get('pipeline_efficiency', 0):.3f}\n")
            f.write("\n")

        # Stream analysis
        if results['stream_analysis']:
            f.write("STREAM EFFICIENCY ANALYSIS\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Stream':<8} {'Type':<12} {'Ops':<6} {'Active':<10} {'Idle':<10} {'Efficiency':<10} {'Idle %':<8}\n")
            f.write(f"{'ID':<8} {'':<12} {'Count':<6} {'(ms)':<10} {'(ms)':<10} {'':<10} {'':<8}\n")
            f.write("-" * 70 + "\n")

            for stream in results['stream_analysis']:
                stream_id = stream.get('streamId', 'N/A')
                stream_type = stream.get('stream_type', 'unknown')
                total_ops = (stream.get('kernel_count', 0) or 0) + (stream.get('memory_count', 0) or 0)
                active_ms = stream.get('total_ops_ms', 0) or 0
                idle_ms = stream.get('idle_time_ms', 0) or 0
                efficiency = stream.get('stream_efficiency', 0) or 0
                idle_pct = stream.get('idle_percentage', 0) or 0

                f.write(f"{str(stream_id):<8} "
                       f"{stream_type:<12} "
                       f"{total_ops:<6} "
                       f"{active_ms:<10.2f} "
                       f"{idle_ms:<10.2f} "
                       f"{efficiency:<10.3f} "
                       f"{idle_pct*100:<8.1f}\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(
        description="SQLite CUPTI latency analyzer with safe error handling.",
        epilog="This version correctly measures GPU utilization and handles missing tables safely."
    )

    parser.add_argument('sqlite_files', nargs='+', help='SQLite database files to analyze')
    parser.add_argument('-o', '--output-dir', default='./corrected_analysis_results',
                        help='Output directory for analysis results')
    parser.add_argument('--console-summary', action='store_true',
                        help='Print summary to console for each configuration')
    parser.add_argument('--include-warmup', action='store_true',
                        help='Include warmup/cleanup periods (analyze entire trace)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    print("Using latency hiding metrics with safe error handling.")

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

            output_file = output_dir / f"{results['config_name']}_corrected_analysis.txt"
            write_analysis_results(results, output_file)
            print(f"Results written to: {output_file}")

            if args.console_summary:
                if 'error' in results:
                    print(f"  Error: {results['error']}")
                elif results['gpu_utilization']:
                    gu = results['gpu_utilization']
                    util = gu.get('gpu_utilization', 0)
                    print(f"  GPU Utilization: {util:.3f} ({util*100:.1f}%)")
                    print(f"  Latency Hiding: {'EXCELLENT' if util > 0.95 else 'GOOD' if util > 0.85 else 'MODERATE' if util > 0.70 else 'POOR'}")

                if results.get('pipeline_efficiency'):
                    pe = results['pipeline_efficiency']
                    efficiency = pe.get('pipeline_efficiency', 0)
                    print(f"  Pipeline Efficiency: {efficiency:.3f}")

        except Exception as e:
            print(f"Error processing {sqlite_file}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nCompleted corrected analysis. Results saved in: {output_dir}")

if __name__ == '__main__':
    main()


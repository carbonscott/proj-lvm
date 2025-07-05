#!/usr/bin/env python3
"""
Extract runs that are likely collecting data from real samples
by analyzing logbook entries and questionnaire data.
"""

import sqlite3
import csv
import re
from datetime import datetime
import sys
import os

# Connect to the database - look for it in current directory or parent
db_path = None
for path in ["2025_0704_1629.db", "elog.db", "../elog.db", "../../elog.db"]:
    if os.path.exists(path):
        db_path = path
        break

if not db_path:
    print("Error: Cannot find database file")
    sys.exit(1)

print(f"Using database: {db_path}")
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Keywords that indicate real sample data collection
SAMPLE_KEYWORDS = [
    # Core sample terms
    'sample', 'crystal', 'protein', 'specimen', 'substrate', 'target',
    
    # Biological samples
    'lysozyme', 'ferredoxin', 'psii', 'photosystem', 'enzyme', 'virus', 
    'bacteria', 'dna', 'rna', 'membrane', 'molecule',
    
    # Sample delivery methods
    'jet', 'capillary', 'droplet', 'injection', 'gdvn', 'electrospray',
    'nozzle', 'aerosol', 'vesicle', 'micelle', 'sheet jet', 'even-lavie', 
    'parker valve', 'gas needle', 'mesh',
    
    # Sample types & materials
    'nanoparticle', 'quantum dot', 'colloid', 'emulsion', 'suspension',
    'buffer', 'solution', 'concentration', 'batch', 'prep', 'solvent',
    
    # Chemical/material names
    'rhodopsin', 'impdh', 'cco', 'cbds', 'fakp', 'agtrif', 'triflic acid',
    'silver particles', 'gold palladium', 'xe atoms', 'adenine', 'indole',
    'cobalt chloride',
    
    # Sample states & conditions
    'flow rate', 'pressure', 'temperature', 'ph', 'dilution', 'molarity',
    'crystallization', 'crystallography', 'diffraction'
]

# Keywords that indicate calibration/diagnostics (not real samples)
CALIBRATION_KEYWORDS = [
    # Detector calibration
    'dark', 'pedestal', 'calibration', 'flat field', 'gain map', 'bad pixel', 
    'hot pixel', 'geometry', 'detector distance', 'detector calibration',
    
    # System alignment & setup
    'alignment', 'focus', 'optimization', 'setup', 'test', 'tuning',
    'beam profile', 'beam position', 'beam alignment', 'pointing',
    'motor scan', 'energy scan', 'delay scan', 'timing calibration',
    
    # Background & reference
    'background', 'reference', 'empty', 'air', 'vacuum', 'without sample',
    'no sample', 'x-ray only', 'laser only', 'fel only', 'visar reference',
    
    # Diagnostic operations
    'diagnostic', 'troubleshooting', 'maintenance', 'repair',
    'baseline', 'rocking curve', 'transmission scan', 'attenuator scan',
    
    # Timing & synchronization
    'time tool', 'timing drift', 'txt calibration', 'lxt calibration',
    'overlap scan', 'delay calibration', 't0 calibration', 'timing reference',
    
    # Beam characterization
    'beam drop', 'beam loss', 'pulse energy', 'transmission', 'attenuation',
    'spectral characterization', 'pulse duration', 'bandwidth',
    
    # Instrument status
    'daq restart', 'daq problem', 'shutter problem', 'pulse picker',
    'klystron trip', 'abort', 'failed', 'crashed', 'error', 'problem', 'issue'
]

# Exclusion patterns for explicit non-sample phrases
EXCLUSION_PATTERNS = [
    'remove sample', 'sample out', 'move away from sample', 'finished with sample',
    'sample change', 'sample loading', 'sample preparation', 'sample delivery problem',
    'injector problem', 'nozzle clogged', 'jet unstable'
]

def extract_experiment_info():
    """Extract basic experiment information including techniques and sample delivery methods."""
    
    print("Extracting experiment information...")
    
    # Get all experiments with their basic info
    query = """
    SELECT 
        e.experiment_id,
        e.name,
        e.instrument,
        e.pi,
        e.description,
        e.start_time,
        e.end_time
    FROM Experiment e
    ORDER BY e.start_time DESC
    """
    
    experiments = {}
    for row in cursor.execute(query):
        experiments[row['experiment_id']] = {
            'name': row['name'],
            'instrument': row['instrument'],
            'pi': row['pi'],
            'description': row['description'],
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'techniques': [],
            'sample_delivery': []
        }
    
    # Extract techniques from questionnaire
    tech_query = """
    SELECT experiment_id, field_name, field_value
    FROM Questionnaire
    WHERE category = 'xraytech' 
    AND field_name LIKE 'tech-%'
    AND field_value IS NOT NULL 
    AND field_value != ''
    """
    
    for row in cursor.execute(tech_query):
        exp_id = row['experiment_id']
        if exp_id in experiments:
            technique = row['field_value']
            if technique not in experiments[exp_id]['techniques']:
                experiments[exp_id]['techniques'].append(technique)
    
    # Extract sample delivery methods from questionnaire
    delivery_query = """
    SELECT experiment_id, field_name, field_value
    FROM Questionnaire
    WHERE category = 'sample' 
    AND (field_name LIKE 'deliverymethod-%' OR field_name LIKE 'nozzle%')
    AND field_value IS NOT NULL 
    AND field_value != ''
    AND field_value NOT LIKE '%prio%'
    """
    
    for row in cursor.execute(delivery_query):
        exp_id = row['experiment_id']
        if exp_id in experiments:
            method = row['field_value']
            if method not in experiments[exp_id]['sample_delivery'] and method != 'Other - specify':
                experiments[exp_id]['sample_delivery'].append(method)
    
    # Also check for delivery methods in the other fields
    other_delivery_query = """
    SELECT experiment_id, field_value
    FROM Questionnaire
    WHERE category = 'sample' 
    AND field_name = 'deliverymethod-other'
    AND field_value IS NOT NULL 
    AND field_value != ''
    """
    
    for row in cursor.execute(other_delivery_query):
        exp_id = row['experiment_id']
        if exp_id in experiments:
            experiments[exp_id]['sample_delivery'].append(row['field_value'])
    
    return experiments

def is_sample_run(logbook_content):
    """Determine if a run is likely collecting real sample data based on logbook content."""
    
    if not logbook_content:
        return False, "No logbook content"
    
    content_lower = logbook_content.lower()
    
    # Check for exclusion patterns first (explicit non-sample phrases)
    for pattern in EXCLUSION_PATTERNS:
        if pattern in content_lower:
            return False, f"Contains exclusion pattern: {pattern}"
    
    # Check for calibration/diagnostic keywords
    calibration_matches = []
    for keyword in CALIBRATION_KEYWORDS:
        if keyword in content_lower:
            calibration_matches.append(keyword)
    
    # Check for sample-related keywords
    sample_matches = []
    for keyword in SAMPLE_KEYWORDS:
        if keyword in content_lower:
            sample_matches.append(keyword)
    
    # Decision logic: prioritize strong calibration indicators
    if calibration_matches and not sample_matches:
        return False, f"Contains calibration keywords: {', '.join(calibration_matches[:3])}"
    
    # If we have sample keywords but also calibration keywords, be more careful
    if sample_matches and calibration_matches:
        # Count how many of each type
        if len(calibration_matches) > len(sample_matches):
            return False, f"More calibration ({len(calibration_matches)}) than sample ({len(sample_matches)}) keywords"
        # If equal or more sample keywords, check for strong calibration indicators
        strong_calibration = ['dark', 'pedestal', 'calibration', 'alignment', 'background', 'reference']
        strong_cal_matches = [k for k in calibration_matches if k in strong_calibration]
        if strong_cal_matches:
            return False, f"Contains strong calibration keywords: {', '.join(strong_cal_matches[:2])}"
    
    if sample_matches:
        return True, f"Sample keywords found: {', '.join(sample_matches[:3])}"
    
    return False, "No sample keywords found"

def extract_sample_runs(experiments):
    """Extract runs that are likely collecting data from real samples."""
    
    print("\nAnalyzing runs for sample data collection...")
    
    results = []
    
    # Query to get runs with their logbook entries
    query = """
    WITH FirstLogEntry AS (
        SELECT 
            l.experiment_id,
            l.run_id,
            l.content,
            l.timestamp,
            ROW_NUMBER() OVER (PARTITION BY l.experiment_id, l.run_id ORDER BY l.timestamp) as rn
        FROM Logbook l
        WHERE l.run_id IS NOT NULL
        AND l.content IS NOT NULL
        AND l.content != ''
    )
    SELECT 
        r.experiment_id,
        r.run_number,
        r.start_time,
        fle.content as logbook_content,
        fle.timestamp as first_timestamp
    FROM Run r
    LEFT JOIN FirstLogEntry fle ON r.run_id = fle.run_id AND fle.rn = 1
    WHERE r.experiment_id IN (SELECT experiment_id FROM Experiment)
    ORDER BY r.experiment_id, r.run_number
    """
    
    total_runs = 0
    sample_runs = 0
    
    for row in cursor.execute(query):
        total_runs += 1
        
        exp_id = row['experiment_id']
        if exp_id not in experiments:
            continue
        
        exp_info = experiments[exp_id]
        
        # Check if this is a sample run
        is_sample, reason = is_sample_run(row['logbook_content'])
        
        if is_sample:
            sample_runs += 1
            
            # Prepare techniques string
            techniques = ', '.join(exp_info['techniques']) if exp_info['techniques'] else 'Not specified'
            
            # Prepare sample delivery string
            delivery = ', '.join(exp_info['sample_delivery']) if exp_info['sample_delivery'] else 'Not specified'
            
            # Clean up logbook content for CSV
            logbook_desc = row['logbook_content'][:500] if row['logbook_content'] else ''
            logbook_desc = re.sub(r'\s+', ' ', logbook_desc).strip()
            
            results.append({
                'instrument': exp_info['instrument'],
                'experiment_id': exp_id,
                'run_number': row['run_number'],
                'first_timestamp': row['first_timestamp'] or row['start_time'] or '',
                'pi': exp_info['pi'],
                'experiment_technique': techniques,
                'sample_delivery_method': delivery,
                'logbook_entry_description': logbook_desc
            })
    
    print(f"\nAnalyzed {total_runs} runs, found {sample_runs} likely sample runs")
    
    return results

def main():
    """Main function to extract and save sample run data."""
    
    try:
        # Extract experiment information
        experiments = extract_experiment_info()
        print(f"Found {len(experiments)} experiments")
        
        # Extract sample runs
        sample_runs = extract_sample_runs(experiments)
        
        # Save to CSV
        output_file = 'sample_runs_analysis.csv'
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'instrument', 'experiment_id', 'run_number', 'first_timestamp',
                'pi', 'experiment_technique', 'sample_delivery_method', 
                'logbook_entry_description'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for run in sample_runs:
                writer.writerow(run)
        
        print(f"\nResults saved to {output_file}")
        print(f"Total sample runs found: {len(sample_runs)}")
        
        # Print summary statistics
        instruments = {}
        techniques = {}
        
        for run in sample_runs:
            # Count by instrument
            inst = run['instrument']
            instruments[inst] = instruments.get(inst, 0) + 1
            
            # Count by technique
            techs = run['experiment_technique'].split(', ')
            for tech in techs:
                if tech and tech != 'Not specified':
                    techniques[tech] = techniques.get(tech, 0) + 1
        
        print("\n--- Summary Statistics ---")
        print("\nSample runs by instrument:")
        for inst, count in sorted(instruments.items()):
            print(f"  {inst}: {count}")
        
        print("\nMost common experiment techniques:")
        for tech, count in sorted(techniques.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {tech}: {count}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    main()
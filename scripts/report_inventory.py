#!/usr/bin/env python3
"""
Inventory Report Generator - Produce ingestion reports

Scans the state directory for ingestion logs and produces a comprehensive
inventory report showing documents, chunks, and skipped files.

Usage:
    python scripts/report_inventory.py --state /state

Output:
    /state/reports/inventory.json
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================================================
# REPORT AGGREGATION
# ============================================================================

def find_ingestion_logs(state_dir: str) -> List[Path]:
    """
    Find all ingestion log files in state directory
    
    Args:
        state_dir: State directory path
        
    Returns:
        List of log file paths
    """
    state_path = Path(state_dir)
    
    if not state_path.exists():
        logger.warning(f"State directory not found: {state_dir}")
        return []
    
    # Find all ingestion_*.json files
    log_files = list(state_path.glob("ingestion_*.json"))
    
    logger.info(f"Found {len(log_files)} ingestion log files")
    return log_files


def parse_ingestion_log(log_file: Path) -> Dict[str, Any]:
    """
    Parse a single ingestion log file
    
    Args:
        log_file: Path to log file
        
    Returns:
        Parsed log data
    """
    try:
        with open(log_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to parse {log_file}: {e}")
        return None


def aggregate_inventory(log_files: List[Path]) -> Dict[str, Any]:
    """
    Aggregate inventory from multiple ingestion logs
    
    Args:
        log_files: List of log file paths
        
    Returns:
        Aggregated inventory report
    """
    # Collections tracking
    collections = defaultdict(lambda: {
        'total_files': 0,
        'total_chunks': 0,
        'files_success': [],
        'files_failed': [],
        'files_skipped': [],
        'last_ingestion': None
    })
    
    # Overall stats
    overall = {
        'total_collections': 0,
        'total_files': 0,
        'total_chunks': 0,
        'files_success': 0,
        'files_failed': 0,
        'files_skipped': 0
    }
    
    # Process each log
    for log_file in log_files:
        log_data = parse_ingestion_log(log_file)
        if not log_data:
            continue
        
        collection_name = log_data.get('collection', 'unknown')
        timestamp = log_data.get('timestamp')
        
        # Update collection data
        coll_data = collections[collection_name]
        
        # Track latest ingestion
        if not coll_data['last_ingestion'] or timestamp > coll_data['last_ingestion']:
            coll_data['last_ingestion'] = timestamp
        
        # Process results
        for result in log_data.get('results', []):
            file_name = result.get('file_name', 'unknown')
            status = result.get('status', 'unknown')
            chunks = result.get('chunks_upserted', 0)
            
            if status == 'success':
                coll_data['files_success'].append({
                    'file_name': file_name,
                    'chunks': chunks,
                    'timestamp': timestamp
                })
                coll_data['total_chunks'] += chunks
                overall['files_success'] += 1
                overall['total_chunks'] += chunks
                
            elif status == 'failed':
                coll_data['files_failed'].append({
                    'file_name': file_name,
                    'error': result.get('error', 'Unknown error'),
                    'timestamp': timestamp
                })
                overall['files_failed'] += 1
                
            elif status == 'skipped':
                coll_data['files_skipped'].append({
                    'file_name': file_name,
                    'reason': result.get('error', result.get('reason', 'Unknown')),
                    'ingest_flag': result.get('ingest_flag'),
                    'timestamp': timestamp
                })
                overall['files_skipped'] += 1
        
        # Update totals
        coll_data['total_files'] = (
            len(coll_data['files_success']) +
            len(coll_data['files_failed']) +
            len(coll_data['files_skipped'])
        )
    
    # Finalize overall stats
    overall['total_collections'] = len(collections)
    overall['total_files'] = (
        overall['files_success'] +
        overall['files_failed'] +
        overall['files_skipped']
    )
    
    return {
        'overall': overall,
        'collections': dict(collections)
    }


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_inventory_report(state_dir: str) -> Dict[str, Any]:
    """
    Generate comprehensive inventory report
    
    Args:
        state_dir: State directory path
        
    Returns:
        Complete inventory report
    """
    logger.info("Generating inventory report...")
    
    # Find and parse logs
    log_files = find_ingestion_logs(state_dir)
    
    if not log_files:
        logger.warning("No ingestion logs found, generating empty report")
        return {
            'generated_at': datetime.utcnow().isoformat(),
            'state_directory': state_dir,
            'overall': {
                'total_collections': 0,
                'total_files': 0,
                'total_chunks': 0,
                'files_success': 0,
                'files_failed': 0,
                'files_skipped': 0
            },
            'collections': {}
        }
    
    # Aggregate data
    inventory = aggregate_inventory(log_files)
    
    # Add metadata
    report = {
        'generated_at': datetime.utcnow().isoformat(),
        'state_directory': state_dir,
        'logs_processed': len(log_files),
        **inventory
    }
    
    return report


def format_report_summary(report: Dict[str, Any]) -> str:
    """
    Format report as human-readable summary
    
    Args:
        report: Inventory report
        
    Returns:
        Formatted summary string
    """
    lines = []
    lines.append("\n" + "="*80)
    lines.append("INVENTORY REPORT SUMMARY")
    lines.append("="*80)
    
    overall = report['overall']
    lines.append(f"\nGenerated: {report['generated_at']}")
    lines.append(f"State Directory: {report['state_directory']}")
    lines.append(f"Logs Processed: {report.get('logs_processed', 0)}")
    
    lines.append(f"\nOVERALL STATISTICS:")
    lines.append(f"  Total Collections: {overall['total_collections']}")
    lines.append(f"  Total Files: {overall['total_files']}")
    lines.append(f"  Total Chunks: {overall['total_chunks']}")
    lines.append(f"  ✅ Success: {overall['files_success']}")
    lines.append(f"  ❌ Failed: {overall['files_failed']}")
    lines.append(f"  ⏭️  Skipped: {overall['files_skipped']}")
    
    # Collection details
    collections = report.get('collections', {})
    if collections:
        lines.append(f"\nCOLLECTIONS ({len(collections)}):")
        
        for coll_name, coll_data in sorted(collections.items()):
            lines.append(f"\n  📁 {coll_name}")
            lines.append(f"     Total Files: {coll_data['total_files']}")
            lines.append(f"     Total Chunks: {coll_data['total_chunks']}")
            lines.append(f"     Success: {len(coll_data['files_success'])}")
            lines.append(f"     Failed: {len(coll_data['files_failed'])}")
            lines.append(f"     Skipped: {len(coll_data['files_skipped'])}")
            lines.append(f"     Last Ingestion: {coll_data.get('last_ingestion', 'N/A')}")
            
            # Show failed files if any
            if coll_data['files_failed']:
                lines.append(f"     Failed Files:")
                for failed in coll_data['files_failed'][:5]:  # Show first 5
                    lines.append(f"       - {failed['file_name']}: {failed.get('error', 'Unknown')}")
                if len(coll_data['files_failed']) > 5:
                    lines.append(f"       ... and {len(coll_data['files_failed']) - 5} more")
            
            # Show skipped files if any
            if coll_data['files_skipped']:
                lines.append(f"     Skipped Files:")
                for skipped in coll_data['files_skipped'][:5]:  # Show first 5
                    lines.append(f"       - {skipped['file_name']}: {skipped.get('reason', 'Unknown')}")
                if len(coll_data['files_skipped']) > 5:
                    lines.append(f"       ... and {len(coll_data['files_skipped']) - 5} more")
    
    lines.append("\n" + "="*80)
    
    return "\n".join(lines)


def save_inventory_report(report: Dict[str, Any], state_dir: str) -> str:
    """
    Save inventory report to JSON file
    
    Args:
        report: Inventory report
        state_dir: State directory path
        
    Returns:
        Path to saved report
    """
    # Create reports subdirectory
    reports_dir = Path(state_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as inventory.json (always overwrite latest)
    report_file = reports_dir / "inventory.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, sort_keys=False)
    
    logger.info(f"Report saved: {report_file}")
    
    # Also save timestamped version for history
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    historical_file = reports_dir / f"inventory_{timestamp}.json"
    
    with open(historical_file, 'w') as f:
        json.dump(report, f, indent=2, sort_keys=False)
    
    logger.info(f"Historical report saved: {historical_file}")
    
    return str(report_file)


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Generate inventory report from ingestion logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/report_inventory.py --state /state

Output:
  /state/reports/inventory.json
  /state/reports/inventory_YYYYMMDD_HHMMSS.json (historical)
        """
    )
    
    parser.add_argument(
        '--state',
        type=str,
        required=True,
        help='State directory containing ingestion logs'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed report summary to console'
    )
    
    args = parser.parse_args()
    
    # Validate state directory
    state_path = Path(args.state)
    if not state_path.exists():
        logger.error(f"State directory not found: {args.state}")
        sys.exit(1)
    
    try:
        # Generate report
        report = generate_inventory_report(args.state)
        
        # Save report
        report_file = save_inventory_report(report, args.state)
        
        # Print summary
        summary = format_report_summary(report)
        print(summary)
        
        if args.verbose:
            print("\n" + "="*80)
            print("DETAILED REPORT")
            print("="*80)
            print(json.dumps(report, indent=2))
        
        logger.info("\n✅ Inventory report generated successfully!")
        logger.info(f"   Output: {report_file}")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
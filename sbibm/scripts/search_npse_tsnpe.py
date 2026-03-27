#!/usr/bin/env python3
import os
import json
import sys
import logging
from typing import Dict, List, Any
from utils import setup_logger, measure_time, save_json_metadata

def search_pattern_in_file(file_path: str, pattern: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if pattern.lower() in content.lower():
                lines = content.split('\n')
                matching_lines = []
                for i, line in enumerate(lines, 1):
                    if pattern.lower() in line.lower():
                        matching_lines.append({
                            'line_number': i,
                            'line_content': line.strip()
                        })
                return {
                    'found': True,
                    'file_path': file_path,
                    'matching_lines': matching_lines,
                    'total_matches': len(matching_lines)
                }
            else:
                return {
                    'found': False,
                    'file_path': file_path,
                    'matching_lines': [],
                    'total_matches': 0
                }
    except Exception as e:
        return {
            'found': False,
            'file_path': file_path,
            'error': str(e),
            'matching_lines': [],
            'total_matches': 0
        }

def search_pattern_in_directory(directory: str, pattern: str, logger) -> tuple[List[Dict[str, Any]], int]:
    results = []
    total_files_searched = 0
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    total_files_searched += 1
                    logger.debug(f"Searching in {file_path}")
                    result = search_pattern_in_file(file_path, pattern)
                    if result['found'] or 'error' in result:
                        results.append(result)
    except Exception as e:
        logger.error(f"Error searching directory {directory}: {e}")
    return results, total_files_searched

def main():
    logger = setup_logger('search_npse_tsnpe', level=logging.INFO)
    
    search_dir = '/home/vrlab/qinwch/dingo-mmdit/sbibm/sbibm'
    output_file = '/home/vrlab/qinwch/dingo-mmdit/sbibm/scripts/npse_tsnpe_search_results.json'
    
    if not os.path.exists(search_dir):
        logger.error(f"Search directory does not exist: {search_dir}")
        sys.exit(1)
    
    with measure_time(logger, "NPSE and TSNPE search"):
        npse_results, npse_total_files = search_pattern_in_directory(search_dir, 'NPSE', logger)
        tsnpe_results, tsnpe_total_files = search_pattern_in_directory(search_dir, 'TSNPE', logger)
        
        total_npse_found = sum(1 for r in npse_results if r['found'])
        total_tsnpe_found = sum(1 for r in tsnpe_results if r['found'])
        
        search_results = {
            'search_timestamp': __import__('datetime').datetime.now().isoformat(),
            'search_directory': search_dir,
            'npse_search': {
                'pattern': 'NPSE',
                'total_files_searched': npse_total_files,
                'total_files_with_matches': total_npse_found,
                'results': npse_results
            },
            'tsnpe_search': {
                'pattern': 'TSNPE',
                'total_files_searched': tsnpe_total_files,
                'total_files_with_matches': total_tsnpe_found,
                'results': tsnpe_results
            }
        }
        
        logger.info(f"NPSE search: {total_npse_found} files found with matches")
        logger.info(f"TSNPE search: {total_tsnpe_found} files found with matches")
        
        try:
            save_json_metadata(search_results, output_file, overwrite=True)
            logger.info(f"Search results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save search results: {e}")
            sys.exit(1)

if __name__ == '__main__':
    main()

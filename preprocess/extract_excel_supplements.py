#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excel supplementary material extractor
Extracts compounds from Excel files and saves to specified output directory

Usage:
  python extract_excel_supplements.py <excel_file> [--output_dir <output_dir>]
  
  If --output_dir is not specified, defaults to supplement_extracted/PMC###/
"""

import sys
import argparse
from pathlib import Path

# Import the module
from admet_extract_llama import run as run_excel

def main():
    parser = argparse.ArgumentParser(description="Extract compounds from Excel supplementary files")
    parser.add_argument("excel_file", help="Path to Excel file (.xlsx or .xls)")
    parser.add_argument("--output_dir", "-o", help="Output directory (default: supplement_extracted/PMC###/)")
    parser.add_argument("--llama-normalize", action="store_true", help="Use Llama for attribute normalization")
    parser.add_argument("--model", default="llama4:latest", help="Ollama model name")
    
    args = parser.parse_args()
    
    excel_path = Path(args.excel_file)
    
    if not excel_path.exists():
        print(f"[ERR] Excel file not found: {excel_path}")
        sys.exit(1)
    
    # Auto-detect PMC ID
    pmc_id = None
    for part in excel_path.parts:
        if part.startswith("PMC"):
            pmc_id = part
            break
    
    if not pmc_id:
        print("[ERR] PMC ID not found in path")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        # Default: supplement_extracted/PMC###/
        out_dir = Path("supplement_extracted") / pmc_id
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run with default settings (llm_find_headers=True for robustness)
    run_excel(
        excel_path, 
        out_dir=out_dir,
        llama_normalize_flag=args.llama_normalize,
        llm_find_headers=True, 
        model=args.model
    )

if __name__ == "__main__":
    main()



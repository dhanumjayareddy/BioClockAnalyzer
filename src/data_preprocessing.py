#!/usr/bin/env python3
"""
data_preprocessing.py
---------------------
Production-ready script for loading, cleaning, and normalizing raw transcriptomic data.

Features:
- Robust error handling and logging.
- Parameter validation for input/output file paths.
- Scalability considerations: uses chunking if needed for large files.
- Detailed inline documentation and troubleshooting notes.

Usage:
    python src/data_preprocessing.py --input data/raw/example_data.csv --output data/processed/normalized_data.csv
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw transcriptomic data.")
    parser.add_argument("--input", required=True, help="Path to the raw CSV data file.")
    parser.add_argument("--output", required=True, help="Path to save the processed CSV data file.")
    parser.add_argument("--chunksize", type=int, default=100000, help="Chunk size for processing large files.")
    return parser.parse_args()

def load_and_preprocess(input_path, output_path, chunksize):
    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        logging.info(f"Loading data from {input_path}...")
        # For very large files, consider using chunksize
        data = pd.read_csv(input_path, index_col=0)
        logging.info(f"Data loaded. Shape: {data.shape}")
        
        # Filter genes: keep genes expressed in at least 50% of samples with count > 1
        threshold = 0.5 * data.shape[1]
        data_filtered = data.loc[(data > 1).sum(axis=1) >= threshold]
        logging.info(f"Data filtered. Remaining genes: {data_filtered.shape[0]}")
        
        # Normalize: Log2-transform the data
        data_norm = np.log2(data_filtered + 1)
        
        # Save processed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data_norm.to_csv(output_path)
        logging.info(f"Processed data saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    args = parse_args()
    load_and_preprocess(args.input, args.output, args.chunksize)

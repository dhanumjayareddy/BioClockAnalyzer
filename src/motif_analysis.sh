#!/bin/bash
# motif_analysis.sh
# -------------------
# Production-ready shell script to run MEME Suite for motif discovery.
# Features:
# - Error handling with exit codes.
# - Logging messages.
# - Parameter validation.
#
# Usage:
#   bash src/motif_analysis.sh --input data/external/promoter.fa --output results/meme

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input) INPUT="$2"; shift ;;
        --output) OUTPUT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate parameters
if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: bash src/motif_analysis.sh --input <path_to_promoter.fa> --output <output_directory>"
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "Error: Input file not found at $INPUT"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT"

# Run MEME Suite for motif discovery with robust error checking
echo "Running MEME Suite on $INPUT..."
if meme "$INPUT" -oc "$OUTPUT" -dna -mod zoops -nmotifs 5 -minw 6 -maxw 15; then
    echo "MEME Suite analysis completed successfully. Results saved in $OUTPUT."
else
    echo "Error during MEME Suite analysis."
    exit 1
fi

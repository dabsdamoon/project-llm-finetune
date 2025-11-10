#!/bin/bash

# Convert KorQuAD dataset with 4 parallel workers
# Usage: ./run_convert_korquad.sh

# Configuration
INPUT_DIR="/mnt/d/datasets/KorQuAD"  # Change this to your KorQuAD data directory
OUTPUT_DIR="./data/korquad"
FORMAT="instruction"  # Options: instruction, qa, chat
NUM_WORKERS=4
MAX_CONTEXT_LENGTH=2000
VAL_RATIO=0.1
OUTPUT_FORMAT="jsonl"  # Options: json, jsonl

echo "=========================================="
echo "KorQuAD Dataset Conversion"
echo "=========================================="
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Format: $FORMAT"
echo "Workers: $NUM_WORKERS"
echo "=========================================="
echo ""

# Run the conversion script
python utils/convert_korquad.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --format "$FORMAT" \
    --num_workers $NUM_WORKERS \
    --max_context_length $MAX_CONTEXT_LENGTH \
    --val_ratio $VAL_RATIO \
    --output_format "$OUTPUT_FORMAT" \
    --clean_html

echo ""
echo "=========================================="
echo "Conversion complete!"
echo "=========================================="

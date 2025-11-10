#!/bin/bash

# Train TinyLlama with KorQuAD dataset
# Usage: ./run_train.sh

CONFIG_FILE="configs/korquad_tinyllama.yaml"

echo "=========================================="
echo "LLM Finetuning with QLoRA"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "=========================================="
echo ""

# Check if data files exist
if [ ! -f "./data/korquad/train.jsonl" ]; then
    echo "❌ Error: Training data not found at ./data/korquad/train.jsonl"
    echo "Please run ./run_convert_korquad.sh first to convert the dataset."
    exit 1
fi

if [ ! -f "./data/korquad/val.jsonl" ]; then
    echo "❌ Error: Validation data not found at ./data/korquad/val.jsonl"
    echo "Please run ./run_convert_korquad.sh first to convert the dataset."
    exit 1
fi

echo "✓ Data files found"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

echo "✓ Config file found"
echo ""

# Run training
echo "Starting training..."
echo ""

python train.py --config "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "Training script completed!"
echo "=========================================="

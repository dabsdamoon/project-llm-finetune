#!/bin/bash
# Quick start script for LLM finetuning on RTX 2080

echo "========================================"
echo "LLM Finetuning Quick Start"
echo "========================================"
echo ""

# Check GPU
echo "Step 1: Checking GPU..."
python check_gpu.py

if [ $? -ne 0 ]; then
    echo "❌ GPU check failed. Please resolve issues before continuing."
    exit 1
fi

echo ""
echo "Step 2: Creating sample dataset..."
python utils/create_sample_dataset.py --num_samples 100

echo ""
echo "Step 3: Preparing training data..."
python utils/prepare_data.py \
    --input ./data/samples/instruction_dataset.json \
    --output_dir ./data \
    --format instruction \
    --val_ratio 0.1

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Choose a model configuration:"
echo "   - configs/tinyllama_1b.yaml (Recommended for testing)"
echo "   - configs/phi2_2.7b.yaml (Good balance)"
echo "   - configs/mistral_7b.yaml (Best quality, tight fit on 8GB)"
echo ""
echo "2. Update the config file to point to your data:"
echo "   train_file: \"./data/train.json\""
echo "   validation_file: \"./data/val.json\""
echo ""
echo "3. Start training:"
echo "   python train.py --config configs/tinyllama_1b.yaml"
echo ""
echo "4. Test your model:"
echo "   python inference.py --model_path ./results/tinyllama-1b-finetuned --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0"

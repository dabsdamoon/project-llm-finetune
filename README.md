# LLM Finetuning for NVIDIA RTX 2080 (8GB VRAM)

A comprehensive, memory-optimized setup for finetuning Large Language Models on consumer GPUs using **QLoRA** (Quantized Low-Rank Adaptation).

## 🚀 Features

- **Memory-Efficient Training**: Uses QLoRA (4-bit quantization + LoRA) to fit larger models in limited VRAM
- **Multiple Model Support**: Pre-configured for TinyLlama 1.1B, Phi-2 2.7B, Gemma 2B, and Mistral 7B
- **Optimized for RTX 2080**: Specifically tuned for 8GB VRAM constraints
- **Easy Data Preparation**: Utilities for converting various data formats
- **Complete Workflow**: From data prep to training to inference

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA RTX 2080 (8GB VRAM) or similar
- **RAM**: 16GB+ system RAM recommended
- **Storage**: 10-40 GB free space (see [MODEL_STORAGE.md](MODEL_STORAGE.md) for details)
  - Single model: 10-15 GB
  - Multiple models: 30-40 GB
  - **Note**: Models are downloaded once and cached locally

### Software
- Python 3.8+
- CUDA 11.8+ or 12.1+
- NVIDIA drivers (latest recommended)

## 🔧 Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd project-llm-finetune
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify GPU setup
```bash
python check_gpu.py
```

This will show your GPU information and recommend suitable models.

## 🎯 Quick Start

### Option 1: Automated Setup
```bash
chmod +x quick_start.sh
./quick_start.sh
```

This script will:
1. Check your GPU
2. Create sample datasets
3. Prepare data for training
4. Show next steps

### Option 2: Manual Setup

#### Step 1: Prepare Your Data

**Create sample data** (for testing):
```bash
python utils/create_sample_dataset.py --num_samples 100
```

**Format your own data**:
```bash
python utils/prepare_data.py \
    --input your_data.json \
    --output_dir ./data \
    --format instruction \
    --val_ratio 0.1
```

Supported formats:
- `instruction`: Alpaca-style (instruction-input-output)
- `qa`: Question-answer pairs
- `conversation`: Multi-turn conversations

#### Step 2: Choose a Model Configuration

Review `GPU_COMPATIBILITY.md` for detailed model recommendations.

**For RTX 2080 (8GB), we recommend:**

| Model | Download | VRAM (Training) | Speed | Quality | Config File |
|-------|----------|----------------|-------|---------|-------------|
| **TinyLlama 1.1B** | 2.2 GB | 4-5 GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | `configs/tinyllama_1b.yaml` |
| **Phi-2 2.7B** | 5.4 GB | 5-6 GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | `configs/phi2_2.7b.yaml` |
| **Gemma 2B** | 4.0 GB | 4-5 GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | `configs/gemma_2b.yaml` |
| **Mistral 7B** | 14 GB | 7-8 GB | ⭐⭐ | ⭐⭐⭐⭐⭐ | `configs/mistral_7b.yaml` |

**Note**: Models are downloaded once and cached in `~/.cache/huggingface/`. See [MODEL_STORAGE.md](MODEL_STORAGE.md) for detailed storage info.

#### Step 3: Update Configuration

Edit your chosen config file to point to your data:

```yaml
train_file: "./data/train.json"
validation_file: "./data/val.json"
```

You can also adjust:
- `num_train_epochs`: Number of training epochs (default: 3)
- `learning_rate`: Learning rate (default: 2e-4)
- `per_device_train_batch_size`: Batch size (1-4 for 8GB GPU)
- `max_seq_length`: Maximum sequence length (512-1024)

#### Step 4: Start Training

```bash
python train.py --config configs/tinyllama_1b.yaml
```

**Monitor GPU usage** (in another terminal):
```bash
watch -n 1 nvidia-smi
```

Training progress will be displayed in the console. Models are saved to the `output_dir` specified in your config.

#### Step 5: Test Your Model

**Interactive mode**:
```bash
python inference.py \
    --model_path ./results/tinyllama-1b-finetuned \
    --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

**Single prompt**:
```bash
python inference.py \
    --model_path ./results/tinyllama-1b-finetuned \
    --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --prompt "Explain quantum computing in simple terms"
```

## 📁 Project Structure

```
project-llm-finetune/
├── README.md                    # This file
├── GPU_COMPATIBILITY.md         # Detailed GPU and model compatibility guide
├── MODEL_STORAGE.md            # Model download sizes and storage requirements
├── requirements.txt             # Python dependencies
├── check_gpu.py                # GPU verification script
├── train.py                    # Main training script
├── inference.py                # Inference script for testing models
├── quick_start.sh              # Automated setup script
├── configs/                    # Model configurations
│   ├── tinyllama_1b.yaml
│   ├── phi2_2.7b.yaml
│   ├── gemma_2b.yaml
│   └── mistral_7b.yaml
├── utils/                      # Data processing utilities
│   ├── prepare_data.py
│   └── create_sample_dataset.py
└── data/                       # Your datasets (created during setup)
```

## 🎓 Understanding QLoRA

QLoRA makes it possible to finetune large models on consumer GPUs:

1. **4-bit Quantization**: Reduces model memory by ~75%
2. **LoRA Adapters**: Only trains small adapter layers (~1-2% of parameters)
3. **Gradient Checkpointing**: Trades compute for memory during backprop
4. **Paged Optimizers**: Efficient memory management

**Result**: Mistral 7B (normally requires ~28GB) fits in 8GB with QLoRA!

## 💡 Tips and Tricks

### Maximizing Performance on 8GB GPU

1. **Start Small**: Begin with TinyLlama to verify your setup
2. **Monitor Memory**: Use `nvidia-smi` to watch GPU usage
3. **Adjust Batch Size**: If OOM errors occur, reduce `per_device_train_batch_size`
4. **Sequence Length**: Reduce `max_seq_length` if memory is tight
5. **Close Other Apps**: Close browser tabs and other GPU-using applications

### Troubleshooting OOM (Out of Memory) Errors

If you get OOM errors with Mistral 7B:

```yaml
# In your config file:
per_device_train_batch_size: 1        # Must be 1
gradient_accumulation_steps: 8        # Increase for same effective batch size
max_seq_length: 384                   # Reduce from 512
lora_r: 4                             # Reduce from 8
```

### Improving Model Quality

1. **More Data**: More diverse, high-quality training data
2. **Longer Training**: Increase `num_train_epochs`
3. **Hyperparameter Tuning**: Experiment with learning rate and LoRA rank
4. **Larger Models**: If memory allows, use Phi-2 or Mistral instead of TinyLlama

### Using Weights & Biases for Tracking

Enable in your config:
```yaml
use_wandb: true
wandb_project: "my-llm-project"
wandb_run_name: "experiment-1"
```

Then login:
```bash
wandb login
```

## 📊 Expected Training Times (RTX 2080)

Approximate times for 1000 samples, 3 epochs:

- **TinyLlama 1.1B**: ~30-45 minutes
- **Phi-2 2.7B**: ~1-2 hours
- **Gemma 2B**: ~1-1.5 hours
- **Mistral 7B**: ~3-4 hours

*Times vary based on sequence length and batch size*

## 🔍 Data Format Examples

### Instruction Format (Alpaca-style)
```json
[
  {
    "instruction": "Write a poem about AI",
    "input": "",
    "output": "Silicon dreams in electric night..."
  }
]
```

### Q&A Format
```json
[
  {
    "question": "What is machine learning?",
    "answer": "Machine learning is..."
  }
]
```

### Conversation Format
```json
[
  [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
]
```

## ⚠️ Common Issues

### CUDA Out of Memory
- Reduce batch size to 1
- Reduce `max_seq_length`
- Reduce `lora_r`
- Use a smaller model

### Slow Training
- Normal for larger models on 8GB GPU
- Consider using a smaller model for experimentation
- Ensure no other processes are using the GPU

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check CUDA compatibility with PyTorch

### Model Download Issues
- Some models require accepting license terms on Hugging Face
- You may need to login: `huggingface-cli login`

## 📚 Additional Resources

- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)
- [PEFT (LoRA) Documentation](https://huggingface.co/docs/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- **QLoRA**: Tim Dettmers et al.
- **LoRA**: Edward Hu et al.
- **Hugging Face**: For the amazing transformers and PEFT libraries
- **Model creators**: TinyLlama, Microsoft (Phi-2), Google (Gemma), Mistral AI

## 📞 Support

If you encounter issues:
1. Check `GPU_COMPATIBILITY.md` for your GPU
2. Review the troubleshooting section above
3. Ensure all dependencies are correctly installed
4. Check GPU drivers and CUDA installation

---

**Happy Finetuning! 🚀**

Built with ❤️ for the AI community

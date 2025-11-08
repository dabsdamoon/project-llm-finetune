# Model Storage Requirements

## Understanding Model Downloads

When you run training or inference, the models are automatically downloaded from Hugging Face and cached in:
- **Linux/Mac**: `~/.cache/huggingface/hub/`
- **Windows**: `C:\Users\<username>\.cache\huggingface\hub\`

## Model Sizes Breakdown

### Full Model Sizes (FP16/BF16 - Normal Precision)

These are the sizes if you download the full model without quantization:

| Model | Parameters | Storage Size | RAM/VRAM (Inference) |
|-------|-----------|--------------|---------------------|
| **TinyLlama 1.1B** | 1.1B | ~2.2 GB | ~2.5 GB |
| **Phi-2 2.7B** | 2.7B | ~5.4 GB | ~6 GB |
| **Gemma 2B** | 2B | ~4 GB | ~4.5 GB |
| **Mistral 7B** | 7B | ~14 GB | ~16 GB |

### 4-bit Quantized Sizes (What QLoRA Uses)

When using 4-bit quantization (NF4), the downloaded model size is the same, but **loaded memory usage** is:

| Model | Downloaded Size | Loaded in VRAM (4-bit) | + LoRA Adapters | Training Total |
|-------|----------------|----------------------|----------------|----------------|
| **TinyLlama 1.1B** | ~2.2 GB | ~0.6 GB | ~0.1 GB | ~4-5 GB* |
| **Phi-2 2.7B** | ~5.4 GB | ~1.4 GB | ~0.2 GB | ~5-6 GB* |
| **Gemma 2B** | ~4 GB | ~1 GB | ~0.1 GB | ~4-5 GB* |
| **Mistral 7B** | ~14 GB | ~3.5 GB | ~0.2 GB | ~7-8 GB* |

*Total includes model + adapters + optimizer states + gradients + activations during training

## What Gets Downloaded?

When you first run the training script, it downloads:

1. **Model weights** (main storage cost)
2. **Tokenizer files** (~1-5 MB)
3. **Configuration files** (~KB)

### Example for Mistral 7B:
```bash
~/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/
├── snapshots/
│   └── <hash>/
│       ├── model.safetensors  # ~14 GB (main model file)
│       ├── tokenizer.json     # ~1.8 MB
│       ├── config.json        # ~600 bytes
│       └── ...
```

## Inference-Only Sizes

If you're ONLY doing inference (not training), here's what you need:

### Option 1: Full Precision Inference (Best Quality)
```python
# This loads the full model
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
```

**Requirements:**
- **TinyLlama 1.1B**: ~2.5 GB VRAM
- **Phi-2 2.7B**: ~6 GB VRAM
- **Gemma 2B**: ~4.5 GB VRAM
- **Mistral 7B**: ~16 GB VRAM ❌ (Won't fit on RTX 2080)

### Option 2: 4-bit Quantized Inference (Memory Efficient)
```python
# This loads with 4-bit quantization (what our inference.py uses)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    load_in_4bit=True  # Uses BitsAndBytes
)
```

**Requirements:**
- **TinyLlama 1.1B**: ~1 GB VRAM
- **Phi-2 2.7B**: ~2 GB VRAM
- **Gemma 2B**: ~1.5 GB VRAM
- **Mistral 7B**: ~4.5 GB VRAM ✅ (Fits on RTX 2080!)

### Option 3: 8-bit Quantized Inference (Good Balance)
```python
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    load_in_8bit=True
)
```

**Requirements:**
- **TinyLlama 1.1B**: ~1.5 GB VRAM
- **Phi-2 2.7B**: ~3.5 GB VRAM
- **Gemma 2B**: ~2.5 GB VRAM
- **Mistral 7B**: ~8 GB VRAM ⚠️ (Very tight on RTX 2080)

## Total Disk Space Required

For a complete setup with multiple models:

```
Download all models:
TinyLlama 1.1B:     2.2 GB
Phi-2 2.7B:         5.4 GB
Gemma 2B:           4.0 GB
Mistral 7B:        14.0 GB
                   -------
Total:             25.6 GB

Add your finetuned adapters:
LoRA adapters:      ~50-200 MB each
Training checkpoints: ~50-200 MB per checkpoint
Datasets:           ~100 MB - 1 GB

Recommended free space: 30-40 GB
```

## Storage Optimization Tips

### 1. Download Only What You Need
```python
# Don't download all models at once
# Start with one model
python train.py --config configs/phi2_2.7b.yaml  # Only downloads Phi-2
```

### 2. Clear Hugging Face Cache
```bash
# Remove unused models
rm -rf ~/.cache/huggingface/hub/models--<model-name>

# Or use huggingface-cli
huggingface-cli delete-cache
```

### 3. Use Smaller Models for Development
```bash
# Test your data pipeline with TinyLlama first
# Then switch to larger models for final training
```

### 4. Symbolic Links for Shared Storage
```bash
# If disk space is limited, use external storage
ln -s /path/to/external/drive ~/.cache/huggingface
```

## Pre-quantized Models (GGUF/GPTQ)

Some models are available in pre-quantized formats that are smaller to download:

### GGUF (for llama.cpp)
- Mistral 7B (4-bit GGUF): ~3.8 GB download
- Compatible with llama.cpp, not directly with our training script

### GPTQ (for transformers)
- Mistral 7B (4-bit GPTQ): ~3.5-4 GB download
- Compatible with transformers library with AutoGPTQ

**Note:** These are inference-only and cannot be used for finetuning.

## Recommendations for RTX 2080 Users

### Disk Space Priority:
1. **Start with Phi-2 2.7B** (~5.4 GB download)
   - Good quality, reasonable size
   - Fast to download and test

2. **Add TinyLlama 1.1B** (~2.2 GB) if you want:
   - Fast experimentation
   - Quick iteration on datasets

3. **Add Mistral 7B** (~14 GB) only if:
   - You have 20+ GB free disk space
   - You need the highest quality outputs
   - You're patient with training times

### Minimum Disk Space:
- **Single model setup**: 10-15 GB free
- **Multi-model setup**: 30-40 GB free
- **With datasets & checkpoints**: 40-50 GB free

## Inference-Only Recommendation

If you're ONLY doing inference (no training), you can:

1. **Download smaller quantized versions** (GGUF/GPTQ)
2. **Use online APIs** (no downloads):
   - Hugging Face Inference API
   - Groq, Together AI, etc.
3. **Load with 4-bit quantization** (our inference.py does this by default)

## Summary Table: RTX 2080 (8GB VRAM)

| Use Case | Model | Download Size | VRAM Usage | Recommended? |
|----------|-------|---------------|------------|--------------|
| **Learning** | TinyLlama 1.1B | 2.2 GB | ~4-5 GB | ✅ Yes |
| **Production** | Phi-2 2.7B | 5.4 GB | ~5-6 GB | ✅ Yes |
| **High Quality** | Mistral 7B | 14 GB | ~7-8 GB | ⚠️ Tight fit |
| **Inference Only** | Mistral 7B (4-bit) | 14 GB | ~4.5 GB | ✅ Yes |

## Key Takeaway

**The model files are downloaded once and cached. The download size equals the full model size (~2-14 GB), but with 4-bit quantization, they only use 1/4 of that VRAM when loaded.**

For your RTX 2080:
- ✅ **Training**: Phi-2 2.7B or TinyLlama (best options)
- ✅ **Inference**: All models work with 4-bit quantization
- ⚠️ **Disk space**: Plan for 10-40 GB depending on how many models you want

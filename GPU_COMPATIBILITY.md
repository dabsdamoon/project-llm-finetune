# GPU Compatibility Guide: RTX 2080 (8GB VRAM)

## Overview
Your NVIDIA GeForce RTX 2080 has **8GB VRAM**, which requires careful model selection and memory optimization techniques for LLM finetuning.

## Recommended Models for RTX 2080

### 1. **TinyLlama 1.1B** ⭐ RECOMMENDED
- **Parameters:** 1.1 billion
- **Memory Required:** ~4-5GB with QLoRA
- **Training Speed:** Fast
- **Use Case:** Best for learning, prototyping, and resource-constrained finetuning
- **Model ID:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

### 2. **Phi-2 2.7B** ⭐ RECOMMENDED
- **Parameters:** 2.7 billion
- **Memory Required:** ~5-6GB with QLoRA
- **Training Speed:** Moderate
- **Use Case:** Excellent performance-to-size ratio, good for instruction tuning
- **Model ID:** `microsoft/phi-2`

### 3. **Gemma 2B**
- **Parameters:** 2 billion
- **Memory Required:** ~4-5GB with QLoRA
- **Training Speed:** Fast-Moderate
- **Use Case:** High-quality outputs, efficient training
- **Model ID:** `google/gemma-2b`

### 4. **Mistral 7B** ⚠️ ADVANCED
- **Parameters:** 7 billion
- **Memory Required:** ~7-8GB with aggressive QLoRA optimization
- **Training Speed:** Slow, requires careful tuning
- **Use Case:** Best quality, but needs maximum optimization
- **Model ID:** `mistralai/Mistral-7B-v0.1`

## Memory Optimization Techniques

### QLoRA (Quantized LoRA) - **ESSENTIAL for RTX 2080**
- Uses 4-bit quantization to reduce model memory footprint by ~75%
- Adds trainable LoRA adapters (~1-2% of parameters)
- Maintains ~95-99% of full finetuning quality

### Key Optimizations
1. **4-bit Quantization:** Reduces base model memory by 4x
2. **Gradient Checkpointing:** Trades compute for memory
3. **Small Batch Sizes:** Use batch_size=1-4 with gradient accumulation
4. **LoRA Rank:** Use r=8-16 for LoRA adapters
5. **Mixed Precision (bf16/fp16):** Further reduces memory usage

## Memory Breakdown Example (Mistral 7B with QLoRA)

```
Base Model (4-bit):        ~3.5 GB
LoRA Adapters:             ~0.1 GB
Optimizer States:          ~1.5 GB
Gradients:                 ~1.0 GB
Activations (batch=1):     ~1.5 GB
--------------------------------------
Total Estimated:           ~7.6 GB ✅ Fits in 8GB!
```

## What WON'T Fit on RTX 2080

❌ **LLaMA-2 7B without quantization** (~28GB)
❌ **LLaMA-2 13B** (even with QLoRA, >10GB)
❌ **Mistral 7B with full finetuning** (~28GB)
❌ **Any model >7B parameters** (even with QLoRA)

## Recommendations

### For Beginners
Start with **TinyLlama 1.1B** or **Phi-2 2.7B** using the provided configs.

### For Best Quality on 8GB
Use **Mistral 7B** with the aggressive optimization config, but be prepared for:
- Slower training
- Need to monitor GPU memory carefully
- May require further tuning of batch size

### For Fastest Iteration
Use **TinyLlama 1.1B** - you can experiment quickly and iterate on your dataset and hyperparameters.

## Checking Your GPU

Run this to verify your GPU:
```bash
python check_gpu.py
```

This will show available VRAM and recommend the best model size for your hardware.

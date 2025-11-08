"""
GPU Compatibility Checker for LLM Finetuning
Checks available GPU memory and recommends suitable models
"""

import torch
import sys


def check_gpu():
    """Check GPU availability and memory"""

    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        print("PyTorch is not detecting any NVIDIA GPU.")
        print("\nPlease ensure:")
        print("1. NVIDIA drivers are installed")
        print("2. CUDA toolkit is installed")
        print("3. PyTorch with CUDA support is installed")
        return False

    # Get GPU information
    gpu_count = torch.cuda.device_count()
    print(f"✅ CUDA is available!")
    print(f"Number of GPUs: {gpu_count}\n")

    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB

        print(f"GPU {i}: {gpu_name}")
        print(f"Total Memory: {gpu_memory:.2f} GB")

        # Get current memory usage
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        free = gpu_memory - reserved

        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved: {reserved:.2f} GB")
        print(f"Free: {free:.2f} GB\n")

        # Recommend models based on available memory
        print("=" * 60)
        print("RECOMMENDED MODELS FOR YOUR GPU")
        print("=" * 60)

        if gpu_memory >= 24:
            print("🚀 Your GPU has plenty of memory!")
            print("✅ Mistral 7B (full finetuning possible)")
            print("✅ LLaMA-2 7B (full finetuning possible)")
            print("✅ LLaMA-2 13B (with QLoRA)")
            print("✅ Any model below 7B parameters")
        elif gpu_memory >= 16:
            print("💪 Your GPU has good memory capacity!")
            print("✅ Mistral 7B (QLoRA recommended)")
            print("✅ LLaMA-2 7B (QLoRA recommended)")
            print("✅ Phi-2 2.7B (full finetuning possible)")
            print("✅ TinyLlama 1.1B (full finetuning possible)")
        elif gpu_memory >= 10:
            print("👍 Your GPU can handle medium-sized models!")
            print("✅ Mistral 7B (QLoRA with optimization)")
            print("✅ Phi-2 2.7B (QLoRA recommended)")
            print("✅ Gemma 2B (QLoRA or full finetuning)")
            print("✅ TinyLlama 1.1B (full finetuning)")
        elif gpu_memory >= 6:
            print("⚡ Your GPU requires careful optimization!")
            print("✅ Phi-2 2.7B (QLoRA, batch_size=1-2)")
            print("✅ Gemma 2B (QLoRA)")
            print("✅ TinyLlama 1.1B (QLoRA or full finetuning)")
            print("⚠️  Mistral 7B (QLoRA, very tight fit, may need tuning)")
        else:
            print("⚠️  Your GPU has limited memory!")
            print("✅ TinyLlama 1.1B (QLoRA)")
            print("✅ GPT-2 Small (full finetuning)")
            print("❌ Larger models will be difficult to train")

        print("\n" + "=" * 60)
        print("OPTIMIZATION TIPS")
        print("=" * 60)
        print("• Use QLoRA (4-bit quantization) for larger models")
        print("• Enable gradient checkpointing")
        print("• Use small batch sizes (1-4) with gradient accumulation")
        print("• Set max_length to reasonable values (512-2048)")
        print("• Monitor GPU memory during training")
        print("• Consider using DeepSpeed ZeRO for further optimization")

    return True


def check_cuda_version():
    """Check CUDA and PyTorch versions"""
    print("\n" + "=" * 60)
    print("SOFTWARE VERSIONS")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GPU COMPATIBILITY CHECKER FOR LLM FINETUNING")
    print("=" * 60 + "\n")

    check_cuda_version()
    print()

    if check_gpu():
        print("\n✅ Your system is ready for LLM finetuning!")
        print("\nNext steps:")
        print("1. Review GPU_COMPATIBILITY.md for detailed model recommendations")
        print("2. Choose a config file from configs/ directory")
        print("3. Prepare your training data")
        print("4. Run: python train.py --config configs/your_config.yaml")
    else:
        print("\n❌ GPU setup needs attention. Please resolve the issues above.")
        sys.exit(1)

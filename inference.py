"""
Inference script for finetuned models

Use this script to test your finetuned LLM models.
Supports both base models and LoRA-finetuned models.
"""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(model_path: str, base_model: str = None, device: str = "cuda"):
    """
    Load model for inference.

    Args:
        model_path: Path to finetuned model (if LoRA) or full model
        base_model: Base model name (required if loading LoRA adapters)
        device: Device to load model on
    """

    print(f"Loading model from {model_path}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model if base_model else model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Check if this is a LoRA model
    if base_model:
        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"Loading LoRA adapters from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # Merge LoRA weights with base model
    else:
        print(f"Loading full model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    print("✅ Model loaded successfully!")

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    device: str = "cuda",
):
    """
    Generate response from model.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repeating tokens
        device: Device to run generation on
    """

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


def interactive_mode(model, tokenizer, device: str = "cuda"):
    """Run interactive chat mode"""

    print("\n" + "="*60)
    print("Interactive Mode - Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")

    while True:
        # Get user input
        prompt = input("\nYou: ").strip()

        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not prompt:
            continue

        # Format prompt (adjust based on your training format)
        formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""

        # Generate response
        print("\nAssistant: ", end="", flush=True)

        response = generate_response(
            model,
            tokenizer,
            formatted_prompt,
            max_new_tokens=256,
            temperature=0.7,
            device=device,
        )

        # Extract only the response part (remove prompt)
        response_only = response[len(formatted_prompt):].strip()
        print(response_only)


def main():
    parser = argparse.ArgumentParser(description="Run inference with finetuned model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to finetuned model or LoRA adapters"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name (required if using LoRA adapters)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for generation (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )

    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load model
    model, tokenizer = load_model(
        args.model_path,
        args.base_model,
        args.device
    )

    # Run inference
    if args.prompt:
        # Single prompt mode
        print(f"\nPrompt: {args.prompt}\n")

        response = generate_response(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )

        print(f"Response:\n{response}")
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, args.device)


if __name__ == "__main__":
    main()

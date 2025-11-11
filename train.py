"""
LLM Finetuning Script with QLoRA for RTX 2080 (8GB VRAM)

This script implements memory-efficient finetuning using:
- QLoRA (4-bit quantization + LoRA adapters)
- Gradient checkpointing
- Memory-optimized training settings
"""

import os
import sys
import torch
import argparse
from dataclasses import dataclass, field
from typing import Optional
import yaml

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
import wandb


@dataclass
class ModelConfig:
    """Configuration for model and training"""
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    use_4bit: bool = True
    use_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    # Training parameters
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # Memory optimization
    gradient_checkpointing: bool = True
    max_seq_length: int = 512

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3

    # Data
    dataset_name: Optional[str] = None
    dataset_text_field: str = "text"
    train_file: Optional[str] = None
    validation_file: Optional[str] = None

    # Misc
    seed: int = 42
    use_wandb: bool = False
    wandb_project: str = "llm-finetuning"
    wandb_run_name: Optional[str] = None


def load_config(config_path: str) -> ModelConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return ModelConfig(**config_dict)


def print_gpu_utilization():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def create_bnb_config(config: ModelConfig):
    """Create BitsAndBytes configuration for quantization"""
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        load_in_8bit=config.use_8bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )

    return bnb_config


def load_model_and_tokenizer(config: ModelConfig):
    """Load model and tokenizer with quantization"""
    print(f"\n{'='*60}")
    print(f"Loading model: {config.model_name}")
    print(f"{'='*60}\n")

    # Create quantization config
    bnb_config = create_bnb_config(config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    print_gpu_utilization()

    return model, tokenizer


def create_peft_config(config: ModelConfig):
    """Create PEFT (LoRA) configuration"""
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return peft_config


def load_and_prepare_dataset(config: ModelConfig, tokenizer):
    """Load and prepare dataset for training"""
    print(f"\n{'='*60}")
    print("Loading dataset")
    print(f"{'='*60}\n")

    # Load dataset
    if config.dataset_name:
        dataset = load_dataset(config.dataset_name)
    elif config.train_file:
        data_files = {"train": config.train_file}
        if config.validation_file:
            data_files["validation"] = config.validation_file

        # Determine file type
        file_extension = config.train_file.split('.')[-1]
        if file_extension == "json" or file_extension == "jsonl":
            dataset = load_dataset("json", data_files=data_files)
        elif file_extension == "csv":
            dataset = load_dataset("csv", data_files=data_files)
        elif file_extension == "txt":
            dataset = load_dataset("text", data_files=data_files)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    else:
        raise ValueError("Either dataset_name or train_file must be provided")

    # Tokenize dataset
    def tokenize_function(examples):
        import numpy as np

        outputs = tokenizer(
            examples[config.dataset_text_field],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
        )
        # Mask padding tokens in labels (set them to -100 so loss ignores them)
        # Using vectorized numpy operations for efficiency
        labels = np.array(outputs["input_ids"])
        attention_mask = np.array(outputs["attention_mask"])
        labels[attention_mask == 0] = -100
        outputs["labels"] = labels.tolist()
        return outputs

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
    )

    print(f"Training samples: {len(tokenized_dataset['train'])}")
    if "validation" in tokenized_dataset:
        print(f"Validation samples: {len(tokenized_dataset['validation'])}")

    return tokenized_dataset


def train(config: ModelConfig):
    """Main training function"""

    # Set seed for reproducibility
    torch.manual_seed(config.seed)

    # Initialize wandb if requested
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.__dict__,
        )

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Create PEFT model
    peft_config = create_peft_config(config)
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    # Load and prepare dataset
    tokenized_dataset = load_and_prepare_dataset(config, tokenizer)

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if "validation" in tokenized_dataset else None,
        eval_strategy="steps" if "validation" in tokenized_dataset else "no",
        save_total_limit=config.save_total_limit,
        fp16=False,
        bf16=True if config.bnb_4bit_compute_dtype == "bfloat16" else False,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to="wandb" if config.use_wandb else "none",
        seed=config.seed,
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),
    )

    # Print initial GPU usage
    print(f"\n{'='*60}")
    print("Starting training")
    print(f"{'='*60}\n")
    print_gpu_utilization()

    # Train
    trainer.train()

    # Save final model
    print(f"\n{'='*60}")
    print("Saving final model")
    print(f"{'='*60}\n")

    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    print(f"\n✅ Training complete! Model saved to {config.output_dir}")
    print_gpu_utilization()

    if config.use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Finetune LLM with QLoRA")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Please check your GPU setup.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("GPU INFORMATION")
    print(f"{'='*60}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"{'='*60}\n")

    # Start training
    train(config)


if __name__ == "__main__":
    main()

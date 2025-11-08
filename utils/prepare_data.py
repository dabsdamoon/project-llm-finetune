"""
Data Preparation Utilities for LLM Finetuning

This module provides utilities to convert various data formats into
training-ready formats for LLM finetuning.
"""

import json
import argparse
from typing import List, Dict
from pathlib import Path


def format_instruction_dataset(
    instructions: List[Dict[str, str]],
    instruction_key: str = "instruction",
    input_key: str = "input",
    output_key: str = "output",
    template: str = None,
) -> List[Dict[str, str]]:
    """
    Format instruction-following dataset into text format.

    Args:
        instructions: List of instruction-input-output dictionaries
        instruction_key: Key for instruction field
        input_key: Key for input field (optional)
        output_key: Key for output/response field
        template: Custom template string (uses {instruction}, {input}, {output})

    Returns:
        List of dictionaries with 'text' field
    """

    if template is None:
        # Default Alpaca-style template
        template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

    formatted_data = []

    for item in instructions:
        instruction = item.get(instruction_key, "")
        input_text = item.get(input_key, "")
        output = item.get(output_key, "")

        # Format using template
        text = template.format(
            instruction=instruction,
            input=input_text if input_text else "N/A",
            output=output
        )

        formatted_data.append({"text": text})

    return formatted_data


def format_conversation_dataset(
    conversations: List[List[Dict[str, str]]],
    role_key: str = "role",
    content_key: str = "content",
) -> List[Dict[str, str]]:
    """
    Format conversational dataset (like ChatML format).

    Args:
        conversations: List of conversation histories
        role_key: Key for role field (user/assistant)
        content_key: Key for message content

    Returns:
        List of dictionaries with 'text' field
    """

    formatted_data = []

    for conversation in conversations:
        text_parts = []

        for message in conversation:
            role = message.get(role_key, "")
            content = message.get(content_key, "")

            if role.lower() in ["user", "human"]:
                text_parts.append(f"### User:\n{content}")
            elif role.lower() in ["assistant", "bot", "ai"]:
                text_parts.append(f"### Assistant:\n{content}")
            elif role == "system":
                text_parts.append(f"### System:\n{content}")

        text = "\n\n".join(text_parts)
        formatted_data.append({"text": text})

    return formatted_data


def format_qa_dataset(
    qa_pairs: List[Dict[str, str]],
    question_key: str = "question",
    answer_key: str = "answer",
) -> List[Dict[str, str]]:
    """
    Format Q&A dataset.

    Args:
        qa_pairs: List of question-answer dictionaries
        question_key: Key for question field
        answer_key: Key for answer field

    Returns:
        List of dictionaries with 'text' field
    """

    formatted_data = []

    for item in qa_pairs:
        question = item.get(question_key, "")
        answer = item.get(answer_key, "")

        text = f"""Question: {question}

Answer: {answer}"""

        formatted_data.append({"text": text})

    return formatted_data


def create_train_val_split(
    data: List[Dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    Split data into train and validation sets.

    Args:
        data: List of data samples
        val_ratio: Ratio of validation data (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data)
    """

    import random
    random.seed(seed)

    # Shuffle data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # Calculate split index
    val_size = int(len(shuffled_data) * val_ratio)
    train_size = len(shuffled_data) - val_size

    # Split
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:]

    return train_data, val_data


def save_jsonl(data: List[Dict], filepath: str):
    """Save data as JSONL format"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_json(data: List[Dict], filepath: str):
    """Save data as JSON format"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare data for LLM finetuning")
    parser.add_argument("--input", type=str, required=True, help="Input JSON/JSONL file")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--format", type=str, choices=["instruction", "conversation", "qa"],
                        default="instruction", help="Data format type")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--output_format", type=str, choices=["json", "jsonl"],
                        default="json", help="Output file format")

    args = parser.parse_args()

    # Load input data
    print(f"Loading data from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        if args.input.endswith('.jsonl'):
            raw_data = [json.loads(line) for line in f]
        else:
            raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} samples")

    # Format data based on type
    print(f"Formatting data as {args.format} format")
    if args.format == "instruction":
        formatted_data = format_instruction_dataset(raw_data)
    elif args.format == "conversation":
        formatted_data = format_conversation_dataset(raw_data)
    elif args.format == "qa":
        formatted_data = format_qa_dataset(raw_data)

    # Create train/val split
    print(f"Splitting data with validation ratio {args.val_ratio}")
    train_data, val_data = create_train_val_split(formatted_data, args.val_ratio)

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save data
    save_func = save_jsonl if args.output_format == "jsonl" else save_json
    extension = "jsonl" if args.output_format == "jsonl" else "json"

    train_path = output_dir / f"train.{extension}"
    val_path = output_dir / f"val.{extension}"

    print(f"Saving training data to {train_path}")
    save_func(train_data, str(train_path))

    print(f"Saving validation data to {val_path}")
    save_func(val_data, str(val_path))

    print("\n✅ Data preparation complete!")
    print(f"\nTo use this data for training, update your config file:")
    print(f"  train_file: \"{train_path}\"")
    print(f"  validation_file: \"{val_path}\"")


if __name__ == "__main__":
    main()

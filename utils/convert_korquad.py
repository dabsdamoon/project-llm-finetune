"""
Convert KorQuAD dataset to training-ready format.

This script converts KorQuAD 2.1 dataset (SQuAD format) into a format
suitable for LLM finetuning with instruction-following or Q&A tasks.

Supports multiprocessing for faster conversion of large datasets.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


def extract_text_from_html(html_text: str) -> str:
    """
    Simple HTML tag removal (you may want to use BeautifulSoup for better cleaning)
    """
    from html import unescape
    import re

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', html_text)
    # Unescape HTML entities
    text = unescape(text)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def format_korquad_qa(
    context: str,
    question: str,
    answer: str,
    format_type: str = "instruction"
) -> str:
    """
    Format KorQuAD Q&A into training text.

    Args:
        context: The context/passage text
        question: The question
        answer: The answer text
        format_type: "instruction", "qa", or "chat"

    Returns:
        Formatted text string
    """

    if format_type == "instruction":
        # Alpaca-style instruction format
        text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
다음 문맥을 읽고 질문에 답하세요.

### Context:
{context}

### Question:
{question}

### Response:
{answer}"""

    elif format_type == "qa":
        # Simple Q&A format
        text = f"""Context: {context}

Question: {question}

Answer: {answer}"""

    elif format_type == "chat":
        # Chat format
        text = f"""### User:
다음 문맥을 읽고 질문에 답하세요.

{context}

{question}

### Assistant:
{answer}"""

    else:
        raise ValueError(f"Unknown format_type: {format_type}")

    return text


def convert_korquad_file(
    file_path: str,
    format_type: str = "instruction",
    clean_html: bool = True,
    max_context_length: int = 2000,
    show_progress: bool = False,
) -> List[Dict[str, str]]:
    """
    Convert a single KorQuAD JSON file to training format.

    Args:
        file_path: Path to KorQuAD JSON file
        format_type: Format type for Q&A pairs
        clean_html: Whether to clean HTML from context
        max_context_length: Maximum context length (in characters)
        show_progress: Whether to show progress bar (disable for multiprocessing)

    Returns:
        List of training examples with 'text' field
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    training_examples = []

    # Use tqdm only if show_progress is True
    data_iter = tqdm(data['data'], desc=f"Converting {Path(file_path).name}", disable=not show_progress)

    for doc in data_iter:
        context = doc.get('context', '')

        # Clean HTML if requested
        if clean_html and context:
            context = extract_text_from_html(context)

        # Truncate context if too long
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."

        # Process each Q&A pair
        for qa in doc.get('qas', []):
            question = qa.get('question', '')
            answer_text = qa.get('answer', {}).get('text', '')

            if not question or not answer_text:
                continue

            # Format the Q&A
            formatted_text = format_korquad_qa(
                context=context,
                question=question,
                answer=answer_text,
                format_type=format_type
            )

            training_examples.append({
                "text": formatted_text,
                "id": qa.get('id', ''),
                "title": doc.get('title', '')
            })

    return training_examples


def process_file_worker(
    file_path: str,
    format_type: str,
    clean_html: bool,
    max_context_length: int,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Worker function for multiprocessing.

    Args:
        file_path: Path to JSON file to process
        format_type: Format type for Q&A pairs
        clean_html: Whether to clean HTML
        max_context_length: Maximum context length

    Returns:
        Tuple of (file_path, examples) for tracking progress
    """
    try:
        examples = convert_korquad_file(
            file_path=file_path,
            format_type=format_type,
            clean_html=clean_html,
            max_context_length=max_context_length,
            show_progress=False  # Disable progress bar in workers
        )
        return (file_path, examples)
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return (file_path, [])


def chunk_list(lst: List, n: int) -> List[List]:
    """
    Split a list into n roughly equal chunks.

    Args:
        lst: List to chunk
        n: Number of chunks

    Returns:
        List of chunks
    """
    chunk_size = len(lst) // n + (1 if len(lst) % n else 0)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def main():
    parser = argparse.ArgumentParser(description="Convert KorQuAD dataset to training format")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing KorQuAD JSON files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/korquad",
        help="Output directory for converted data"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["instruction", "qa", "chat"],
        default="instruction",
        help="Format type for Q&A pairs"
    )
    parser.add_argument(
        "--clean_html",
        action="store_true",
        default=True,
        help="Clean HTML tags from context"
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=2000,
        help="Maximum context length in characters"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["json", "jsonl"],
        default="jsonl",
        help="Output file format"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of input files to process (for testing)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help=f"Number of parallel workers (default: CPU count = {cpu_count()})"
    )

    args = parser.parse_args()

    # Set default number of workers
    if args.num_workers is None:
        args.num_workers = cpu_count()

    # Ensure at least 1 worker
    args.num_workers = max(1, args.num_workers)

    # Find all JSON files
    input_path = Path(args.input_dir)
    json_files = []

    # Search in subdirectories
    for pattern in ["**/*.json", "*.json"]:
        json_files.extend(input_path.glob(pattern))

    json_files = sorted(set(str(f) for f in json_files))

    if args.max_files:
        json_files = json_files[:args.max_files]

    print(f"\nFound {len(json_files)} JSON files")
    print(f"Using {args.num_workers} worker(s) for parallel processing")

    if not json_files:
        print("❌ No JSON files found!")
        return

    # Convert all files
    all_examples = []

    if args.num_workers == 1:
        # Single-threaded processing with progress bar
        print("\nProcessing files sequentially...")
        for json_file in json_files:
            print(f"Processing {json_file}...")
            examples = convert_korquad_file(
                file_path=json_file,
                format_type=args.format,
                clean_html=args.clean_html,
                max_context_length=args.max_context_length,
                show_progress=True
            )
            all_examples.extend(examples)
    else:
        # Multiprocessing
        print(f"\nProcessing files in parallel with {args.num_workers} workers...")

        # Create partial function with fixed parameters
        worker_func = partial(
            process_file_worker,
            format_type=args.format,
            clean_html=args.clean_html,
            max_context_length=args.max_context_length
        )

        # Process files in parallel with progress bar
        with Pool(processes=args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(worker_func, json_files),
                total=len(json_files),
                desc="Processing files"
            ))

        # Accumulate results
        for file_path, examples in results:
            if examples:
                all_examples.extend(examples)
                print(f"✓ {Path(file_path).name}: {len(examples)} examples")
            else:
                print(f"✗ {Path(file_path).name}: No examples")

    print(f"\n✅ Total examples: {len(all_examples)}")

    # Create train/val split
    import random
    random.seed(42)
    random.shuffle(all_examples)

    val_size = int(len(all_examples) * args.val_ratio)
    train_data = all_examples[val_size:]
    val_data = all_examples[:val_size]

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save data
    extension = args.output_format
    train_path = output_dir / f"train.{extension}"
    val_path = output_dir / f"val.{extension}"

    print(f"\nSaving training data to {train_path}")
    if args.output_format == "jsonl":
        with open(train_path, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

    print(f"Saving validation data to {val_path}")
    if args.output_format == "jsonl":
        with open(val_path, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

    # Print sample
    print("\n" + "="*60)
    print("Sample training example:")
    print("="*60)
    print(train_data[0]['text'][:500] + "...")

    print("\n✅ Conversion complete!")
    print(f"\nTo use this data for training, update your config file:")
    print(f'  train_file: "{train_path}"')
    print(f'  validation_file: "{val_path}"')
    print(f'  dataset_text_field: "text"')


if __name__ == "__main__":
    main()

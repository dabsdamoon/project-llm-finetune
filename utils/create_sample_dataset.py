"""
Create sample datasets for testing LLM finetuning

This script generates example datasets in various formats
to help you get started with finetuning.
"""

import json
import argparse
from pathlib import Path


def create_instruction_dataset(num_samples: int = 50) -> list:
    """Create sample instruction-following dataset (Alpaca-style)"""

    samples = [
        {
            "instruction": "Write a short poem about artificial intelligence.",
            "input": "",
            "output": "Silicon minds awake at dawn,\nThinking thoughts where none were drawn,\nLearning patterns, finding ways,\nTo illuminate our modern days."
        },
        {
            "instruction": "Explain the concept in simple terms.",
            "input": "Quantum Computing",
            "output": "Quantum computing uses the principles of quantum mechanics to process information. Unlike regular computers that use bits (0 or 1), quantum computers use qubits that can be both 0 and 1 at the same time, allowing them to solve certain problems much faster."
        },
        {
            "instruction": "Translate the following sentence to French.",
            "input": "Hello, how are you today?",
            "output": "Bonjour, comment allez-vous aujourd'hui?"
        },
        {
            "instruction": "Write a Python function to calculate factorial.",
            "input": "",
            "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)"
        },
        {
            "instruction": "Summarize the main idea.",
            "input": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
            "output": "Machine learning is an AI subset that allows systems to automatically learn and improve from data without explicit programming."
        },
        {
            "instruction": "What are the health benefits of regular exercise?",
            "input": "",
            "output": "Regular exercise provides numerous health benefits including: improved cardiovascular health, stronger muscles and bones, better mental health and mood, weight management, reduced risk of chronic diseases, enhanced immune system, and improved sleep quality."
        },
        {
            "instruction": "Convert the temperature from Celsius to Fahrenheit.",
            "input": "25°C",
            "output": "25°C is equal to 77°F. The formula is: F = (C × 9/5) + 32"
        },
        {
            "instruction": "Suggest three ways to reduce carbon footprint.",
            "input": "",
            "output": "1. Use public transportation or carpool instead of driving alone\n2. Reduce meat consumption and eat more plant-based meals\n3. Use energy-efficient appliances and switch to renewable energy sources"
        },
        {
            "instruction": "Debug this code and explain the error.",
            "input": "for i in range(10)\n    print(i)",
            "output": "The code is missing a colon after the for statement. Corrected version:\nfor i in range(10):\n    print(i)\n\nIn Python, for loops require a colon (:) at the end of the for statement."
        },
        {
            "instruction": "Write a professional email subject line.",
            "input": "Requesting a meeting with the team to discuss Q4 project goals",
            "output": "Subject: Meeting Request - Q4 Project Goals Discussion"
        }
    ]

    # Repeat samples to reach desired number
    result = []
    while len(result) < num_samples:
        result.extend(samples)

    return result[:num_samples]


def create_qa_dataset(num_samples: int = 50) -> list:
    """Create sample Q&A dataset"""

    samples = [
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris."
        },
        {
            "question": "How does photosynthesis work?",
            "answer": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar."
        },
        {
            "question": "Who wrote 'Romeo and Juliet'?",
            "answer": "William Shakespeare wrote 'Romeo and Juliet'."
        },
        {
            "question": "What is the speed of light?",
            "answer": "The speed of light in a vacuum is approximately 299,792,458 meters per second (or about 186,282 miles per second)."
        },
        {
            "question": "What causes seasons on Earth?",
            "answer": "Seasons are caused by the tilt of Earth's axis (23.5 degrees) as it orbits around the Sun, resulting in different amounts of sunlight reaching different parts of Earth throughout the year."
        }
    ]

    # Repeat samples to reach desired number
    result = []
    while len(result) < num_samples:
        result.extend(samples)

    return result[:num_samples]


def create_conversation_dataset(num_samples: int = 50) -> list:
    """Create sample conversational dataset"""

    samples = [
        [
            {"role": "user", "content": "Can you help me learn Python?"},
            {"role": "assistant", "content": "Of course! I'd be happy to help you learn Python. Python is a great programming language for beginners. What aspect would you like to start with - basic syntax, data types, or something specific?"},
            {"role": "user", "content": "Let's start with basic syntax."},
            {"role": "assistant", "content": "Great choice! Python syntax is known for being clean and readable. Here are some basics:\n\n1. Print statement: print('Hello, World!')\n2. Variables: x = 5\n3. Comments: # This is a comment\n4. Indentation is important in Python (use 4 spaces)\n\nWould you like to try writing a simple program?"}
        ],
        [
            {"role": "user", "content": "What's the difference between a list and a tuple in Python?"},
            {"role": "assistant", "content": "The main differences are:\n\n1. Mutability: Lists are mutable (can be changed), tuples are immutable (cannot be changed)\n2. Syntax: Lists use [], tuples use ()\n3. Performance: Tuples are slightly faster\n4. Use case: Use lists for collections that might change, tuples for fixed collections\n\nExample:\nmy_list = [1, 2, 3]  # Can modify\nmy_tuple = (1, 2, 3)  # Cannot modify"}
        ],
        [
            {"role": "user", "content": "How do I install a Python package?"},
            {"role": "assistant", "content": "You can install Python packages using pip:\n\npip install package_name\n\nFor example:\npip install numpy\n\nOr to install a specific version:\npip install numpy==1.21.0\n\nTo see installed packages:\npip list"}
        ]
    ]

    # Repeat samples to reach desired number
    result = []
    while len(result) < num_samples:
        result.extend(samples)

    return result[:num_samples]


def main():
    parser = argparse.ArgumentParser(description="Create sample datasets for testing")
    parser.add_argument("--output_dir", type=str, default="./data/samples",
                        help="Output directory for sample datasets")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples per dataset")
    parser.add_argument("--format", type=str,
                        choices=["instruction", "qa", "conversation", "all"],
                        default="all", help="Dataset format to create")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formats_to_create = ["instruction", "qa", "conversation"] if args.format == "all" else [args.format]

    for fmt in formats_to_create:
        print(f"Creating {fmt} dataset with {args.num_samples} samples...")

        if fmt == "instruction":
            data = create_instruction_dataset(args.num_samples)
            output_file = output_dir / "instruction_dataset.json"
        elif fmt == "qa":
            data = create_qa_dataset(args.num_samples)
            output_file = output_dir / "qa_dataset.json"
        elif fmt == "conversation":
            data = create_conversation_dataset(args.num_samples)
            output_file = output_dir / "conversation_dataset.json"

        # Save dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved to {output_file}")

    print(f"\n✅ Sample dataset creation complete!")
    print(f"\nTo prepare these for training, run:")
    print(f"  python utils/prepare_data.py --input {output_dir}/instruction_dataset.json --format instruction")


if __name__ == "__main__":
    main()

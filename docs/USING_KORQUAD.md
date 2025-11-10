# Using Local KorQuAD Dataset for Finetuning

This guide explains how to use your local KorQuAD dataset located in `/mnt/d/datasets/KorQuAD` for LLM finetuning.

## Overview

KorQuAD (Korean Question Answering Dataset) is a question-answering dataset based on Korean Wikipedia articles. It follows a similar structure to SQuAD (Stanford Question Answering Dataset).

## Dataset Structure

Your KorQuAD dataset has the following structure:
```
/mnt/d/datasets/KorQuAD/
├── KorQuAD_2.1_train_00/
│   ├── korquad2.1_train_00.json
│   ├── korquad2.1_train_01.json
│   └── korquad2.1_train_02.json
├── KorQuAD_2.1_train_01/
├── KorQuAD_2.1_train_02/
└── ... (more subdirectories)
```

Each JSON file contains:
- `version`: Dataset version
- `data`: List of documents with:
  - `title`: Document title
  - `context`: HTML context text
  - `qas`: Question-answer pairs

## Step 1: Convert KorQuAD to Training Format

The training script expects data with a simple `text` field. Use the conversion script to transform KorQuAD format:

### Basic Usage

```bash
# Convert all KorQuAD files
python utils/convert_korquad.py \
  --input_dir /mnt/d/datasets/KorQuAD \
  --output_dir ./data/korquad \
  --format instruction \
  --val_ratio 0.1
```

### Options

- `--input_dir`: Directory containing KorQuAD JSON files (default: `/mnt/d/datasets/KorQuAD`)
- `--output_dir`: Output directory for converted files (default: `./data/korquad`)
- `--format`: Format type - `instruction`, `qa`, or `chat` (default: `instruction`)
- `--clean_html`: Clean HTML tags from context (default: `true`)
- `--max_context_length`: Maximum context length in characters (default: `2000`)
- `--val_ratio`: Validation split ratio (default: `0.1`)
- `--output_format`: Output format - `json` or `jsonl` (default: `jsonl`)
- `--max_files`: Maximum number of files to process (useful for testing)

### Format Types

1. **instruction** (Recommended for general finetuning):
   ```
   Below is an instruction that describes a task...

   ### Instruction:
   다음 문맥을 읽고 질문에 답하세요.

   ### Context:
   [context text]

   ### Question:
   [question]

   ### Response:
   [answer]
   ```

2. **qa** (Simple Q&A format):
   ```
   Context: [context text]

   Question: [question]

   Answer: [answer]
   ```

3. **chat** (Chat-style format):
   ```
   ### User:
   다음 문맥을 읽고 질문에 답하세요.
   [context]
   [question]

   ### Assistant:
   [answer]
   ```

### Test with Subset

To test the conversion with a small subset first:

```bash
python utils/convert_korquad.py \
  --input_dir /mnt/d/datasets/KorQuAD \
  --output_dir ./data/korquad_test \
  --max_files 2 \
  --val_ratio 0.1
```

This will process only 2 files (~2000 examples) for quick testing.

### Full Dataset Conversion

For the complete dataset:

```bash
python utils/convert_korquad.py \
  --input_dir /mnt/d/datasets/KorQuAD \
  --output_dir ./data/korquad_full \
  --format instruction \
  --val_ratio 0.05
```

**Note**: The full KorQuAD 2.1 dataset is large. Conversion may take several minutes.

## Step 2: Update Configuration

After conversion, update your config file to use the converted dataset:

### Option A: Modify Existing Config

Edit `configs/tinyllama_1b.yaml`:

```yaml
# Dataset configuration
dataset_name: null
dataset_text_field: "text"
train_file: "./data/korquad/train.jsonl"
validation_file: "./data/korquad/val.jsonl"
```

### Option B: Use Provided KorQuAD Config

A pre-configured file is available at `configs/korquad_tinyllama.yaml`:

```yaml
# Already configured for KorQuAD
train_file: "./data/korquad/train.jsonl"
validation_file: "./data/korquad/val.jsonl"
dataset_text_field: "text"
```

Just update the paths if you used a different output directory.

## Step 3: Start Training

```bash
python train.py --config configs/korquad_tinyllama.yaml
```

## Memory Considerations

KorQuAD examples can be long due to the context. Adjust these parameters based on your GPU:

### For RTX 2080 (8GB VRAM):

```yaml
max_seq_length: 1024           # Reduce to 512 if OOM
per_device_train_batch_size: 2  # Reduce to 1 if OOM
gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

### If Out of Memory:

1. Reduce `max_seq_length` to 512 or 768
2. Reduce `per_device_train_batch_size` to 1
3. Increase `gradient_accumulation_steps` to 16
4. Reduce `max_context_length` during conversion (e.g., `--max_context_length 1500`)

## Alternative: Direct Loading (Advanced)

If you prefer to modify the training script to load KorQuAD directly without conversion:

### 1. Modify `train.py`

Add a KorQuAD-specific loading function in `train.py` around line 158:

```python
def load_korquad_dataset(data_path: str, tokenizer, max_seq_length: int):
    """Load KorQuAD dataset directly"""
    import json
    import glob
    from datasets import Dataset

    # Find all JSON files
    json_files = glob.glob(f"{data_path}/**/*.json", recursive=True)

    examples = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for doc in data['data']:
            context = doc.get('context', '')
            for qa in doc.get('qas', []):
                question = qa.get('question', '')
                answer = qa.get('answer', {}).get('text', '')

                text = f"Question: {question}\nContext: {context}\nAnswer: {answer}"
                examples.append({"text": text})

    return Dataset.from_list(examples)
```

### 2. Update Config

```yaml
dataset_name: null
korquad_data_path: "/mnt/d/datasets/KorQuAD"
dataset_text_field: "text"
```

**Note**: The conversion approach (Option 1) is recommended as it's cleaner and more flexible.

## Validation

After conversion, verify the data:

```bash
# Check number of samples
wc -l data/korquad/train.jsonl data/korquad/val.jsonl

# View a sample
head -n 1 data/korquad/train.jsonl | python -m json.tool

# Check file sizes
du -h data/korquad/
```

## Tips

1. **Start Small**: Test with `--max_files 2` first to verify everything works
2. **Monitor Memory**: Use `nvidia-smi` to watch GPU memory during training
3. **Context Length**: Korean text can be verbose; adjust `max_context_length` and `max_seq_length` accordingly
4. **Format Choice**: Try different formats (`instruction`, `qa`, `chat`) to see what works best for your use case
5. **Validation Ratio**: For large datasets, a smaller validation ratio (0.05) is sufficient

## Troubleshooting

### "No JSON files found"
- Verify the input directory path
- Check that JSON files exist in the directory or subdirectories

### Out of Memory during training
- Reduce `max_seq_length` in config
- Reduce `per_device_train_batch_size`
- Reduce `max_context_length` during conversion

### Conversion too slow
- The script processes ~1000-1500 examples/second
- For full dataset, expect 10-20 minutes
- Use `--max_files` for testing

### Poor training results
- Try different format types (`instruction` vs `qa` vs `chat`)
- Adjust learning rate
- Increase training epochs
- Consider using a Korean-specific base model

## Example Workflow

Complete workflow from start to finish:

```bash
# 1. Test conversion with small subset
python utils/convert_korquad.py \
  --input_dir /mnt/d/datasets/KorQuAD \
  --output_dir ./data/korquad_test \
  --max_files 2

# 2. Verify output
head -n 1 data/korquad_test/train.jsonl | python -m json.tool

# 3. Test training with small dataset
python train.py --config configs/korquad_tinyllama.yaml

# 4. If successful, convert full dataset
python utils/convert_korquad.py \
  --input_dir /mnt/d/datasets/KorQuAD \
  --output_dir ./data/korquad_full \
  --val_ratio 0.05

# 5. Update config to use full dataset
# Edit configs/korquad_tinyllama.yaml:
#   train_file: "./data/korquad_full/train.jsonl"
#   validation_file: "./data/korquad_full/val.jsonl"

# 6. Start full training
python train.py --config configs/korquad_tinyllama.yaml
```

## Next Steps

- Try different base models (e.g., models pre-trained on Korean text)
- Experiment with different LoRA configurations
- Use Weights & Biases for experiment tracking (`use_wandb: true`)
- Fine-tune hyperparameters based on validation performance

## References

### Official Resources

- **Official Website**: [https://korquad.github.io/](https://korquad.github.io/)
  - Dataset downloads, leaderboards, and evaluation scripts

- **Official GitHub Repository**: [https://github.com/korquad/korquad.github.io](https://github.com/korquad/korquad.github.io)
  - Korean wiki QA dataset for Machine Reading Comprehension (MRC)

### Research Papers

- **KorQuAD 1.0 Paper** (arXiv):
  - Lim, Seungyoung; Kim, Myungji; Lee, Jooyoul (2019)
  - "KorQuAD1.0: Korean QA Dataset for Machine Reading Comprehension"
  - [https://arxiv.org/abs/1909.07005](https://arxiv.org/abs/1909.07005)

- **KorQuAD 2.0 Paper** (Korean Journal):
  - Kim, Youngmin; Lim, Seungyoung; Lee, Hyunjeong; Park, Soyoon; Kim, Myungji (2020)
  - "KorQuAD 2.0: Korean QA Dataset for Web Document Machine Comprehension"
  - Journal of KIISE, Vol. 47, No. 6, pp. 577-586
  - [Paper PDF](https://korquad.github.io/dataset/KorQuAD_2.0/KorQuAD_2.0_paper.pdf)

### Dataset Information

**KorQuAD 2.1 Details:**
- 102,960 question-answer pairs
- 47,957 Wikipedia articles
- 83,486 training pairs
- 10,165 validation pairs
- Corrected version of KorQuAD 2.0 with improved HTML tag handling

**Key Differences from KorQuAD 1.0:**
1. Documents are whole Wikipedia pages (not just paragraphs)
2. Contains tables and lists requiring HTML tag understanding
3. Answers can be long text covering paragraphs, tables, and lists

### HuggingFace Datasets

- **KorQuAD 1.0**: [KorQuAD/squad_kor_v1](https://huggingface.co/datasets/KorQuAD/squad_kor_v1)
  - 60,407 training examples
  - 5,774 validation examples

- **KorQuAD 2.0**: [KorQuAD/squad_kor_v2](https://huggingface.co/datasets/KorQuAD/squad_kor_v2)
  - 100,000+ question-answer pairs
  - Whole Wikipedia pages as context

- **KorQuAD Chat**: [heegyu/korquad-chat-v1](https://huggingface.co/datasets/heegyu/korquad-chat-v1)
  - 9,619 conversational datasets generated from KorQuAD 1.0

### License

KorQuAD datasets are released under **CC BY-ND 2.0 KR** license
- Attribution required
- No derivatives (original dataset)
- For this project: You're creating training data transformations which is a permitted use case

### Related Resources

**Korean Language Models:**
- [KoBERT](https://github.com/SKTBrain/KoBERT) - Korean BERT pre-trained model
- [KoGPT2](https://github.com/SKT-AI/KoGPT2) - Korean GPT-2 model
- [KoAlpaca](https://github.com/Beomi/KoAlpaca) - Korean Alpaca model

**Implementations:**
- [KorQuAD with BERT](https://github.com/lyeoni/KorQuAD) - PyTorch implementation
- [KoBERT-KorQuAD](https://github.com/monologg/KoBERT-KorQuAD) - Korean MRC with KoBERT

**Evaluation:**
- [Papers with Code - KorQuAD](https://paperswithcode.com/dataset/korquad)
- State-of-the-art models and benchmarks

### Project Documentation

- [Training Documentation](../README.md)
- [Model Configuration Guide](../MODEL_STORAGE.md)
- [Data Preparation Utilities](../utils/prepare_data.py)

### Citations

If you use KorQuAD in your research, please cite:

```bibtex
@article{lim2019korquad1,
  title={KorQuAD1.0: Korean QA Dataset for Machine Reading Comprehension},
  author={Lim, Seungyoung and Kim, Myungji and Lee, Jooyoul},
  journal={arXiv preprint arXiv:1909.07005},
  year={2019}
}

@article{kim2020korquad,
  title={KorQuAD 2.0: Korean QA Dataset for Web Document Machine Comprehension},
  author={Kim, Youngmin and Lim, Seungyoung and Lee, Hyunjeong and Park, Soyoon and Kim, Myungji},
  journal={Journal of KIISE},
  volume={47},
  number={6},
  pages={577--586},
  year={2020}
}
```

### Additional Notes

**Dataset Source**: Your local dataset at `/mnt/d/datasets/KorQuAD` appears to be KorQuAD 2.1, which is the corrected version of KorQuAD 2.0 with improved HTML tag removal.

**Community**: Join the discussion and see latest benchmarks at the [official website](https://korquad.github.io/).

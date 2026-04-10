#!/bin/bash
# SEVA v2 SFT Training
# Trains Qwen2.5-3B-Instruct on structured reasoning chains
#
# Usage:
#   bash scripts/run_seva_sft.sh
#   bash scripts/run_seva_sft.sh --num_train_epochs 5

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate verl

set -x

# ============================================================
# Paths
# ============================================================
BASE_MODEL=${BASE_MODEL:-/home/yinian/models/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1}
DATA_DIR=${DATA_DIR:-/home/yinian/verifiable_agent/data/attribution}
OUTPUT_DIR=${OUTPUT_DIR:-/home/yinian/verifiable_agent/checkpoints/seva_sft}

TRAIN_FILE="${DATA_DIR}/seva_sft_train.jsonl"

echo "============================================"
echo "SEVA v2 SFT Training"
echo "============================================"
echo "Base model: ${BASE_MODEL}"
echo "Train data: ${TRAIN_FILE}"
echo "Output:     ${OUTPUT_DIR}"
echo "============================================"

# Check data exists
if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: Training data not found: $TRAIN_FILE"
    echo "Run: python scripts/generate_seva_sft_data.py first"
    exit 1
fi

# Count training samples
N_SAMPLES=$(wc -l < "$TRAIN_FILE")
echo "Training samples: $N_SAMPLES"

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    -m transformers.trainer_seq2seq \
    2>/dev/null || \
python -c "
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import json
import torch

# Load model and tokenizer
model_path = '${BASE_MODEL}'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load data
dataset = load_dataset('json', data_files='${TRAIN_FILE}', split='train')

def tokenize_fn(examples):
    texts = []
    for messages in examples['messages']:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=1280,  # longer for structured output
        padding='max_length',
        return_tensors='pt',
    )
    tokenized['labels'] = tokenized['input_ids'].clone()

    # Mask system + user tokens (only train on assistant response)
    for i, messages in enumerate(examples['messages']):
        # Find where assistant response starts
        prefix = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
        prefix_ids = tokenizer(prefix, truncation=True, max_length=1280)['input_ids']
        prefix_len = len(prefix_ids)
        tokenized['labels'][i, :prefix_len] = -100

    return tokenized

dataset = dataset.map(tokenize_fn, batched=True, batch_size=32, remove_columns=dataset.column_names)

# Training args
args = TrainingArguments(
    output_dir='${OUTPUT_DIR}',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    save_strategy='epoch',
    save_total_limit=2,
    report_to='none',
    dataloader_num_workers=4,
    gradient_checkpointing=True,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Save final
trainer.save_model('${OUTPUT_DIR}/final')
tokenizer.save_pretrained('${OUTPUT_DIR}/final')
print('SFT training complete. Model saved to ${OUTPUT_DIR}/final')
" "$@"

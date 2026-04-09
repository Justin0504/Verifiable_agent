"""SFT warm-up for SEVA attribution verification.

Trains Qwen2.5-3B on ANLI-based attribution data (same starting point as ClearCheck).

Usage:
    python scripts/run_sft_attribution.py
    python scripts/run_sft_attribution.py --epochs 3 --lr 2e-5
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)


MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/home/yinian/models/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
)
DATA_PATH = os.environ.get(
    "DATA_PATH",
    "/home/yinian/verifiable_agent/data/attribution/sft_train.jsonl"
)
VAL_PATH = os.environ.get(
    "VAL_PATH",
    "/home/yinian/verifiable_agent/data/attribution/sft_val.jsonl"
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "/home/yinian/verifiable_agent/checkpoints/sft_attribution"
)


class AttributionSFTDataset(Dataset):
    """Chat-format SFT dataset for attribution verification."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path) as f:
            for line in f:
                d = json.loads(line)
                self.samples.append(d["messages"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]

        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        full_ids = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            return_tensors="pt", add_special_tokens=False
        )
        input_ids = full_ids["input_ids"].squeeze(0)
        attention_mask = full_ids["attention_mask"].squeeze(0)

        # Create labels: only compute loss on assistant response
        labels = input_ids.clone()

        # Find assistant turn start — mask everything before it
        assistant_text = self.tokenizer.apply_chat_template(
            messages[:2], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(
            assistant_text, truncation=True, max_length=self.max_length,
            return_tensors="pt", add_special_tokens=False
        )
        prompt_len = prompt_ids["input_ids"].shape[1]

        labels[:prompt_len] = -100  # Mask prompt tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--val-data", default=VAL_PATH)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    print("=" * 60)
    print("SEVA Attribution SFT Training")
    print("=" * 60)
    print(f"Model:    {args.model}")
    print(f"Data:     {args.data}")
    print(f"Output:   {args.output}")
    print(f"Epochs:   {args.epochs}")
    print(f"LR:       {args.lr}")
    print("=" * 60)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load datasets
    train_dataset = AttributionSFTDataset(args.data, tokenizer, args.max_length)
    print(f"Train samples: {len(train_dataset)}")

    val_dataset = None
    if os.path.exists(args.val_data):
        val_dataset = AttributionSFTDataset(args.val_data, tokenizer, args.max_length)
        print(f"Val samples:   {len(val_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=4,
    )

    # Data collator
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # Train
    trainer.train()

    # Save final checkpoint
    final_dir = os.path.join(args.output, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nSaved final model to: {final_dir}")


if __name__ == "__main__":
    main()

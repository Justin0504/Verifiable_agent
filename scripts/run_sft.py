"""SFT warm-up: Fine-tune Qwen2.5-3B-Instruct on GPT-4o verification trajectories.

Usage:
    python scripts/run_sft.py
    python scripts/run_sft.py --epochs 2 --lr 2e-5 --batch-size 4
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


MODEL_PATH = "/home/yinian/models/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
DATA_PATH = "/home/yinian/verifiable_agent/data/sft_train.jsonl"
OUTPUT_DIR = "/home/yinian/verifiable_agent/checkpoints/sft_qwen3b"


class VerificationSFTDataset(Dataset):
    """Chat-format SFT dataset. Only compute loss on assistant tokens."""

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

        # Apply chat template to get full text
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize full conversation
        full_ids = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            return_tensors="pt", add_special_tokens=False
        )
        input_ids = full_ids["input_ids"].squeeze(0)
        attention_mask = full_ids["attention_mask"].squeeze(0)

        # Find where assistant response starts - mask loss on everything before it
        # Get the prompt (everything except assistant response)
        prompt_messages = messages[:-1]  # system + user
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(
            prompt_text, truncation=True, max_length=self.max_length,
            return_tensors="pt", add_special_tokens=False
        )
        prompt_len = prompt_ids["input_ids"].shape[1]

        # Labels: -100 for prompt tokens (no loss), actual ids for assistant tokens
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--lora", action="store_true", default=False,
                        help="Use LoRA instead of full fine-tune")
    parser.add_argument("--lora-r", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset from {args.data}...")
    full_dataset = VerificationSFTDataset(args.data, tokenizer, args.max_length)

    # Split train/eval
    n_eval = max(10, int(len(full_dataset) * args.eval_ratio))
    n_train = len(full_dataset) - n_eval
    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_eval],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {n_train}, Eval: {n_eval}")

    print(f"Loading model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if args.lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Effective batch size = batch_size * grad_accum * num_gpus
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=args.bf16,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting SFT training...")
    trainer.train()

    # Save final model
    final_path = os.path.join(args.output, "final")
    print(f"Saving model to {final_path}...")
    if args.lora:
        model.save_pretrained(final_path)
    else:
        trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    print("SFT training complete!")
    print(f"Model saved to: {final_path}")


if __name__ == "__main__":
    main()
